import torch
import yaml
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# --- 引入组件 ---
from DataPipeline.factory import build_dataloader
from DataPipeline.DataBuilder import build_dataset
from models.factory import build_model
from portfolio.factory import build_portfolio_model
from losses.factory import build_loss
from optimizers.factory import build_optimizer
from utils.logging import Logger, log_experiment_setup, log_training_epoch
from utils.plotting import plot_backtest_results 
from utils.analysis import (
    extract_linear_importance, 
    plot_feature_importance_heatmap, 
    calculate_turnover, 
    plot_weights_area, 
    plot_turnover
)
from utils.tuner import run_optuna_tuning

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
    exp_name = cfg["experiment"]["name"] + "_rolling_optuna" # 改个名字区分
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    exp_name = f"{exp_name}_{timestamp}"
    save_dir = os.path.join("outputs", exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    logger = Logger(save_dir)
    sample_feat_path = os.path.join(cfg["data"]["root"], f"{cfg['data']['etfs'][0]}.csv")
    sample_df = pd.read_csv(sample_feat_path)
    # 假设第一列是 Date，后面是特征。如果第一列是 index 可以调整。
    # 这里假设你的特征数据除了 Date 以外全是特征
    feature_importance_history = [] # 用于存储每个月的特征重要度
    logger.log(f"Rolling Backtest (w/ Optuna) Config Loaded.")
    feature_names = cfg["data"]["features"] 
    logger.log(f"Feature Names ({len(feature_names)}): {feature_names}")
    log_experiment_setup(logger, cfg)

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

        # 设置基础时间
        cfg["data"]["train_start"] = str_train_start
        cfg["data"]["train_end"] = str_train_end
        
        # 强制更新 num_assets (防止报错)
        cfg["model"]["params"]["num_assets"] = len(cfg["data"]["etfs"])
        
        # =============================================================
        # === 【核心步骤】运行 Optuna 寻找当前窗口的最佳参数 ===
        # =============================================================
        # 注意：这里 n_trials=10，如果觉得太慢可以改成 5
        best_params = run_optuna_tuning(
            base_cfg=cfg,
            train_start_str=str_train_start,
            train_end_str=str_train_end,
            n_trials=10, 
            logger=logger
        )
        
        # === 将最佳参数应用到当前 Config ===
        cfg["optimizer"]["params"]["lr"] = best_params["lr"]
        cfg["trainer"]["epochs"] = best_params["epochs"]
        
        # =============================================================
        # === 正式训练 (使用最佳参数在 完整窗口 上重训) ===
        # =============================================================
        set_seed(cfg["experiment"]["seed"]) # 重置种子确保可复现
        
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
        total_epochs = cfg["trainer"]["epochs"] # 这里已经是 Optuna 选出来的最佳轮数了

        for epoch in range(1, total_epochs + 1):
            epoch_loss_sum = 0.0
            batch_count = 0
            
            last_pred_cost = None
            last_true_cost = None

            for batch in train_loader:
                features = batch["features"].to(device)
                c_true = -batch["cost"].to(device)
                
                pred_cost = -model(features)
                loss = loss_fn(pred_cost, c_true) 
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss_sum += loss.item()
                batch_count += 1
                
                last_pred_cost = pred_cost
                last_true_cost = c_true

            # 计算平均 Loss
            avg_loss = epoch_loss_sum / batch_count if batch_count > 0 else 0.0
            
            # 日志监控 (正式训练时我们打印 Loss 和 Regret)
            log_training_epoch(
                logger=logger,
                epoch=epoch,
                total_epochs=total_epochs,
                avg_loss=avg_loss,
                pred_cost=last_pred_cost,
                true_cost=last_true_cost,
                portfolio_model=portfolio_model
            )
        
        # --- Inference (保持不变) ---
        model.eval()
        if hasattr(model, "linear"): # 确保是线性模型
            imp_dict = extract_linear_importance(model, feature_names, len(cfg["data"]["etfs"]))
            imp_dict["Date"] = str_test_start
            feature_importance_history.append(imp_dict)
        df_features, df_labels = build_dataset(
            tickers=cfg["data"]["etfs"],
            data_dir=cfg["data"]["root"],
            start_date=str_test_start,
            end_date=str_test_end,
            dropna=True,
            feature_list=feature_names
)

        if df_features.empty:
            continue

        feature_row = df_features.iloc[0].values
        num_assets = len(cfg["data"]["etfs"])
        input_dim = feature_row.shape[0] // num_assets
        x_input = torch.tensor(feature_row, dtype=torch.float32).reshape(1, num_assets, input_dim).to(device)

        with torch.no_grad():
            w_opt, _ = portfolio_model.solve((-model(x_input))[0].cpu().numpy())
        
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
        full_perf_df["nav"] = initial_capital * np.exp(full_perf_df["daily_return"].cumsum())
        
        full_perf_df.to_csv(os.path.join(save_dir, "rolling_performance.csv"))
        if all_weights:
            pd.DataFrame(all_weights).set_index("Date").to_csv(os.path.join(save_dir, "rolling_weights.csv"))

        logger.log("Generating plots using utils.plotting...")
        plot_backtest_results(full_perf_df, save_dir)

# 1. 特征重要度分析
    if feature_importance_history:
        plot_feature_importance_heatmap(feature_importance_history, save_dir)

    # 2. 权重和换手率分析
    if all_weights:
        weights_df = pd.DataFrame(all_weights).set_index("Date")
        
        # >>> 【新增这一行】将索引转换为 Datetime 对象 <<<
        weights_df.index = pd.to_datetime(weights_df.index)
        
        # 确保数据是 float
        weights_df = weights_df.astype(float)
        
        # 保存
        weights_df.to_csv(os.path.join(save_dir, "rolling_weights.csv"))

        # 画堆积图
        plot_weights_area(weights_df, save_dir)
        
        # 计算并画换手率
        turnover = calculate_turnover(weights_df)
        plot_turnover(turnover, save_dir)
        
        avg_to = turnover.mean()
        logger.log(f"Average Monthly Turnover: {avg_to:.4f}")

    logger.log("Backtest Finished.")

if __name__ == "__main__":
    rolling_backtest()