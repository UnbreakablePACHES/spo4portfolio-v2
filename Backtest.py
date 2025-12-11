import torch
import yaml
import os
import numpy as np
import pandas as pd
from datetime import datetime,timedelta
from dateutil.relativedelta import relativedelta

# --- 引入组件 ---
from DataPipeline.factory import build_dataloader
from DataPipeline.DataBuilder import build_dataset
from models.factory import build_model
from portfolio.factory import build_portfolio_model
from losses.factory import build_loss
from optimizers.factory import build_optimizer
from utils.logging import Logger

# 【新增】引入 plotting 模块
from utils.plotting import plot_backtest_results 

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def rolling_backtest(config_path: str = "configs/spo_plus_linear.yaml"):
    # 1. 加载 Config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "backtest" not in cfg:
        raise ValueError("Config file missing 'backtest' section!")
    
    bt_cfg = cfg["backtest"]
    exp_name = cfg["experiment"]["name"] + "_rolling"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    exp_name = f"{exp_name}_{timestamp}"
    save_dir = os.path.join("outputs", exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    logger = Logger(save_dir)
    logger.log(f"Rolling Backtest Config Loaded: {bt_cfg}")

    device = torch.device(cfg["trainer"].get("device", "cpu"))
    
    # 2. 解析时间轴
    start_test_date = pd.Timestamp(bt_cfg["test_start"])
    end_test_date = pd.Timestamp(bt_cfg["test_end"])
    
    freq = bt_cfg.get("rebalance_freq", "MS")
    window_months = bt_cfg.get("train_window_size", 12)
    initial_capital = bt_cfg.get("initial_capital", 1.0)
    
    rebalance_dates = pd.date_range(start=start_test_date, end=end_test_date, freq=freq)
    
    all_weights = []
    all_returns_dfs = [] 
    
    # 3. 滚动循环
    for i, current_date in enumerate(rebalance_dates):
        # (保持原有的训练窗口计算逻辑不变...)
        train_end_dt = current_date - timedelta(days=1)
        train_start_dt = train_end_dt - relativedelta(months=window_months) + timedelta(days=1)
        
        if i < len(rebalance_dates) - 1:
            test_end_dt = rebalance_dates[i+1] - timedelta(days=1)
        else:
            test_end_dt = current_date + pd.offsets.MonthEnd(0)

        str_train_start = train_start_dt.strftime("%Y-%m-%d")
        str_train_end = train_end_dt.strftime("%Y-%m-%d")
        str_test_start = current_date.strftime("%Y-%m-%d")
        str_test_end = test_end_dt.strftime("%Y-%m-%d")

        logger.log(f"\n>>> Rebalance Date: {str_test_start}")
        logger.log(f"    Train Window: {str_train_start} -> {str_train_end}")

        # --- 动态设置训练数据范围 ---
        cfg["data"]["train_start"] = str_train_start
        cfg["data"]["train_end"] = str_train_end
        set_seed(cfg["experiment"]["seed"])

        # --- Train ---
        try:
            train_loader = build_dataloader(cfg)
        except Exception as e:
            logger.log(f"    [Error] Dataloader failed: {e}")
            continue

        model = build_model(cfg).to(device)
        portfolio_model = build_portfolio_model(cfg)
        loss_fn = build_loss(cfg, portfolio_model)
        optimizer = build_optimizer(cfg, model)
        
        model.train()
        for epoch in range(1, cfg["trainer"]["epochs"] + 1):
            for batch in train_loader:
                features = batch["features"].to(device)
                c_true = -batch["cost"].to(device)
                
                loss = loss_fn(-model(features), c_true) # c_pred = -r_pred
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # --- Inference ---
        model.eval()
        df_features, df_labels = build_dataset(
            tickers=cfg["data"]["etfs"],
            data_dir=cfg["data"]["root"],
            start_date=str_test_start,
            end_date=str_test_end,
            dropna=True
        )

        if df_features.empty:
            continue

        # 预测权重
        feature_row = df_features.iloc[0].values
        num_assets = len(cfg["data"]["etfs"])
        input_dim = feature_row.shape[0] // num_assets
        x_input = torch.tensor(feature_row, dtype=torch.float32).reshape(1, num_assets, input_dim).to(device)

        with torch.no_grad():
            w_opt, _ = portfolio_model.solve((-model(x_input))[0].cpu().numpy())
        
        # 记录
        period_df = pd.DataFrame({"daily_return": df_labels.values @ w_opt}, index=df_labels.index)
        all_returns_dfs.append(period_df)
        
        w_record = {"Date": str_test_start}
        for idx, ticker in enumerate(cfg["data"]["etfs"]):
            w_record[ticker] = w_opt[idx]
        all_weights.append(w_record)
        
        logger.log(f"    Weights: {np.round(w_opt, 3)}")

    # 4. 结果汇总与画图
    if all_returns_dfs:
        full_perf_df = pd.concat(all_returns_dfs)
        full_perf_df.index.name = "Date"
        
        # 计算 NAV
        full_perf_df["nav"] = initial_capital * np.exp(full_perf_df["daily_return"].cumsum())
        
        full_perf_df.to_csv(os.path.join(save_dir, "rolling_performance.csv"))
        if all_weights:
            pd.DataFrame(all_weights).set_index("Date").to_csv(os.path.join(save_dir, "rolling_weights.csv"))

        # 调用 utils 中的画图函数
        logger.log("Generating plots using utils.plotting...")
        plot_backtest_results(full_perf_df, save_dir)

    logger.log("Backtest Finished.")

if __name__ == "__main__":
    rolling_backtest()