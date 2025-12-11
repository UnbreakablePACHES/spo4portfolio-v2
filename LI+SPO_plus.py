# ======================================
# Imports & Setup
# ======================================
import torch
import pandas as pd
import numpy as np
import optuna
from torch.optim import Adam
from dateutil.relativedelta import relativedelta

from DataPipeline.DataBuilder import build_dataset
from models.PortfolioModel import PortfolioModel
from models.LinearInferencer import LinearPredictorTorch
from pyepo.func.surrogate import SPOPlus

# ======================================
# Build Monthly Dataset
# ======================================
def build_quarterly_dataset(tickers, data_dir, oracle_df, start_month, num_quarters):
    x_list = []
    y_list = []
    for i in range(num_quarters):
        q_start = start_month + relativedelta(months=3 * i)
        q_end = q_start + relativedelta(months=3) - pd.Timedelta(days=1)

        features_df, _ = build_dataset(
            tickers=tickers,
            data_dir=data_dir,
            start_date=str(q_start.date()),
            end_date=str(q_end.date())
        )
        features_df.index = pd.to_datetime(features_df.index).normalize()

        x = features_df.values  # 所有天合并为一个样本

        mask = (oracle_df.index >= q_start) & (oracle_df.index <= q_end)
        y = oracle_df.loc[mask].mean().values  # 平均 oracle 策略

        x_list.append(x)
        y_list.append(y)

    return x_list, y_list
# ======================================
# Training Function
# ======================================
def train_one_epoch(predictor, x_list, y_list, optimizer, spo_loss_fn, optmodel, device):
    predictor.train()
    total_loss = 0.0
    for X_month, c_true_avg in zip(x_list, y_list):
        x_tensor = torch.tensor(X_month, dtype=torch.float32).to(device)  # [T, D]
        c_true = torch.tensor(c_true_avg, dtype=torch.float32).to(device)  # [A]

        optimizer.zero_grad()
        c_hat_all = predictor(x_tensor)  # [T, A]
        c_hat = c_hat_all.mean(dim=0, keepdim=True)  # [1, A]
        c_true = c_true.unsqueeze(0)  # [1, A]

        optmodel.setObj(c_true.detach().cpu().numpy().squeeze())
        z_star_np, obj_val = optmodel.solve()
        z_star = torch.tensor(z_star_np, dtype=torch.float32, device=device).unsqueeze(0)
        true_obj = torch.tensor(obj_val, dtype=torch.float32, device=device).unsqueeze(0)

        loss = spo_loss_fn(c_hat, c_true, z_star, true_obj)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(x_list)

# ======================================
# Optuna Objective
# ======================================
def objective(trial, tickers, oracle_df, train_start, device):
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    num_epochs = trial.suggest_int("num_epochs", 24, 36)

    num_assets = len(tickers)
    features_df, _ = build_dataset(
        tickers=tickers,
        data_dir="data/FeatureData",
        start_date=str(train_start.date()),
        end_date=str(train_start + relativedelta(months=12) - pd.Timedelta(days=1))
    )
    input_dim = features_df.shape[1] // num_assets

    x_list, y_list = build_quarterly_dataset(
        tickers=tickers,
        data_dir="data/FeatureData",
        oracle_df=oracle_df,
        start_month=train_start,
        num_quarters=4  # 4 个季度（即 1 年训练期）
    )

    predictor = LinearPredictorTorch(input_dim * num_assets, num_assets).to(device)
    optmodel = PortfolioModel(n_assets=num_assets, budget=1.0)
    spo_loss_fn = SPOPlus(optmodel, processes=1, solve_ratio=1.0, reduction="mean")
    optimizer = Adam(predictor.parameters(), lr=lr)

    for epoch in range(num_epochs):
        loss = train_one_epoch(predictor, x_list, y_list, optimizer, spo_loss_fn, optmodel, device)

    trial.set_user_attr("model", predictor.state_dict())
    trial.set_user_attr("params", {"lr": lr, "num_epochs": num_epochs})
    return loss

# ======================================
# Main Rolling Loop (example usage)
# ======================================
if __name__ == "__main__":
    tickers = ["EEM", "EFA", "JPXN", "SPY", "XLK", "VTI", "AGG", "DBC"]
    num_assets = len(tickers)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    oracle_df = pd.read_csv("data/DailyOracle/oracle_weights_with_fee.csv", index_col=0)
    oracle_df.index = pd.to_datetime(oracle_df.index).normalize()

    return_df = pd.read_csv("data/DailyReturn/DailyReturn_8tickers.csv", index_col=0)
    return_df.index = pd.to_datetime(return_df.index).normalize()
    return_df.columns = [col.replace("_return", "") for col in return_df.columns]  # match ticker names

    start_month = pd.to_datetime("2016-01-01")
    results = []

    for i in range(108):
        infer_start = start_month + relativedelta(months=i)
        train_start = infer_start - relativedelta(years=1)
        infer_end = infer_start + pd.offsets.MonthEnd(0)

        print(f"\n📅 {infer_start.strftime('%Y-%m')}: 训练期 {train_start.date()} ~ {(infer_start - pd.Timedelta(days=1)).date()}，推断期 {infer_start.date()} ~ {infer_end.date()}")

        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, tickers, oracle_df, train_start, device), n_trials=10)

        best_state_dict = study.best_trial.user_attrs["model"]
        features_df, _ = build_dataset(
            tickers=tickers,
            data_dir="data/FeatureData",
            start_date=str(infer_start.date()),
            end_date=str(infer_end.date())
        )
        input_dim = features_df.shape[1] // num_assets

        predictor = LinearPredictorTorch(input_dim * num_assets, num_assets).to(device)
        predictor.load_state_dict(best_state_dict)
        predictor.eval()

        x_tensor = torch.tensor(features_df.values, dtype=torch.float32).to(device)
        with torch.no_grad():
            c_hat = predictor(x_tensor).mean(dim=0).cpu().numpy()

        optmodel = PortfolioModel(n_assets=num_assets, budget=1.0)
        optmodel.setObj(c_hat)
        z_star, _ = optmodel.solve()

        try:
            arith_return_month = np.expm1(return_df.loc[infer_start:infer_end, tickers].values)
            daily_return = arith_return_month @ z_star
            monthly_return = np.prod(1 + daily_return) - 1
        except Exception as e:
            print(f"⚠️ 无法计算 {infer_start.strftime('%Y-%m')} 的组合收益：{e}")
            monthly_return = np.nan

        # 👇 使用复利但累计收益率从 0 开始
        prev_cum = 0.0 if i == 0 else results[-1]["CumulativeReturn"]
        cumulative_return = (1 + prev_cum) * (1 + monthly_return) - 1

        results.append({
            "Month": infer_start.strftime("%Y-%m"),
            "PortfolioWeights": list(z_star),
            "MonthlyReturn": monthly_return,
            "CumulativeReturn": cumulative_return
        })

        print(f"组合权重: {np.round(z_star, 3)}，月收益: {monthly_return:.4f}，累计收益: {cumulative_return:.4f}")

    df_result = pd.DataFrame(results)
    df_result.to_csv("result/8_ticker_1ytrain1yinfer/LP+SPO_plus.csv", index=False)
    print("\n✅ 全部月份处理完成,结果保存为:LI+SPO_plus.csv")
