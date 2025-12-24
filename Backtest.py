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
from utils.logging import Logger, log_experiment_setup
from utils.plotting import plot_backtest_results
from utils.analysis import (
    extract_linear_importance,
    plot_feature_importance_heatmap,
    calculate_turnover,
    plot_weights_area,
    plot_turnover,
    calculate_performance_metrics,
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

    # 区分实验名称
    model_type = cfg["model"]["type"]
    exp_name = cfg["experiment"]["name"] + f"_{model_type}_rolling"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    exp_name = f"{exp_name}_{timestamp}"
    save_dir = os.path.join("outputs", exp_name)
    os.makedirs(save_dir, exist_ok=True)

    logger = Logger(save_dir)
    logger.log(f"Rolling Backtest Config Loaded. Model Type: {model_type}")

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
    feature_importance_history = []

    # 3. 滚动循环
    for i, current_date in enumerate(rebalance_dates):
        train_end_dt = current_date - timedelta(days=1)
        train_start_dt = (
            train_end_dt - relativedelta(months=window_months) + timedelta(days=1)
        )

        if i < len(rebalance_dates) - 1:
            test_end_dt = rebalance_dates[i + 1] - timedelta(days=1)
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

        # 强制更新 num_assets
        cfg["model"]["params"]["num_assets"] = len(cfg["data"]["etfs"])

        # =============================================================
        # === Optuna 调参 (Softmax 模式下也同样适用) ===
        # =============================================================
        best_params = run_optuna_tuning(
            base_cfg=cfg,
            train_start_str=str_train_start,
            train_end_str=str_train_end,
            n_trials=10,
            logger=logger,
            seed=cfg["experiment"].get("seed", 42),
        )

        cfg["optimizer"]["params"]["lr"] = best_params["lr"]
        cfg["trainer"]["epochs"] = best_params["epochs"]

        # =============================================================
        # === 正式训练 ===
        # =============================================================
        set_seed(cfg["experiment"]["seed"])

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
        total_epochs = cfg["trainer"]["epochs"]

        for epoch in range(1, total_epochs + 1):
            epoch_loss_sum = 0.0
            batch_count = 0

            for batch in train_loader:
                features = batch["features"].to(device)
                c_true_input = batch["cost"].to(device)  # Cost (负收益)

                # === 新增：获取 prev_weights (如果 Loader 提供了) ===
                w_prev = batch.get("prev_weights")
                if w_prev is not None:
                    w_prev = w_prev.to(device)

                optimizer.zero_grad()

                # === 训练逻辑 ===
                if model_type == "softmax":
                    weights = model(features)
                    loss = loss_fn(weights, c_true_input)
                else:
                    # Linear
                    pred_return = model(features)
                    # Loss 需要预测成本 (-pred_return)
                    # 并传入 prev_weights (如果是 None，SPOPlusLoss 会自动处理为无手续费模式)
                    loss = loss_fn(-pred_return, c_true_input, prev_weights=w_prev)

                loss.backward()
                optimizer.step()

                epoch_loss_sum += loss.item()
                batch_count += 1

            avg_loss = epoch_loss_sum / batch_count if batch_count > 0 else 0.0

            if epoch == total_epochs:
                logger.log(f"    [Train] Epoch {epoch}: Loss={avg_loss:.6f}")

        # =============================================================
        # === Inference (推断) ===
        # =============================================================
        model.eval()

        if hasattr(model, "linear") and model_type == "linear":
            imp_dict = extract_linear_importance(
                model, feature_names, len(cfg["data"]["etfs"])
            )
            imp_dict["Date"] = str_test_start
            feature_importance_history.append(imp_dict)

        # === 【修复】这里使用 _ 接收第3个返回值 (prev_weights)，但在推断时不使用 Oracle 权重 ===
        df_features, df_labels, _ = build_dataset(
            tickers=cfg["data"]["etfs"],
            data_dir=cfg["data"]["root"],
            start_date=str_test_start,
            end_date=str_test_end,
            dropna=True,
            feature_list=feature_names,
        )

        if df_features.empty:
            continue

        feature_row = df_features.iloc[0].values
        num_assets = len(cfg["data"]["etfs"])
        input_dim = feature_row.shape[0] // num_assets
        x_input = (
            torch.tensor(feature_row, dtype=torch.float32)
            .reshape(1, num_assets, input_dim)
            .to(device)
        )

        # === 准备 w_current (上期持仓) 用于 Lasso ===
        if len(all_weights) > 0:
            # 取上一次回测结果的权重
            last_w_dict = all_weights[-1]
            w_current = np.array([last_w_dict[t] for t in cfg["data"]["etfs"]])
        else:
            w_current = np.zeros(num_assets)

        with torch.no_grad():
            model_out = model(x_input)

            if model_type == "softmax":
                w_opt = model_out[0].cpu().numpy()
            else:
                pred_return = model_out
                # 检查是否是带手续费的模型 (有 optimize 方法)
                if hasattr(portfolio_model, "optimize"):
                    # 新模型：必须传入 w_current
                    w_opt = portfolio_model.optimize(
                        pred_return[0].cpu().numpy(), w_current
                    )
                else:
                    # 旧模型：solve
                    w_opt, _ = portfolio_model.solve(pred_return[0].cpu().numpy())

        period_df = pd.DataFrame(
            {"daily_return": df_labels.values @ w_opt}, index=df_labels.index
        )
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
        full_perf_df["nav"] = initial_capital * np.exp(
            full_perf_df["daily_return"].cumsum()
        )

        full_perf_df.to_csv(os.path.join(save_dir, "rolling_performance.csv"))
        if all_weights:
            pd.DataFrame(all_weights).set_index("Date").to_csv(
                os.path.join(save_dir, "rolling_weights.csv")
            )

        logger.log("Generating plots using utils.plotting...")
        plot_backtest_results(full_perf_df, save_dir)

        metrics = calculate_performance_metrics(full_perf_df, save_dir)
        logger.log("-" * 30)
        logger.log(">>> Final Performance Metrics:")
        logger.log(f"    Annualized Return: {metrics['Annualized Return']:.2%}")
        logger.log(f"    Annualized Volatility: {metrics['Annualized Volatility']:.2%}")
        logger.log(f"    Sharpe Ratio:      {metrics['Sharpe Ratio']:.4f}")
        logger.log(f"    Max Drawdown:      {metrics['Max Drawdown']:.2%}")
        logger.log("-" * 30)

    if feature_importance_history:
        plot_feature_importance_heatmap(feature_importance_history, save_dir)

    if all_weights:
        weights_df = pd.DataFrame(all_weights).set_index("Date")
        weights_df.index = pd.to_datetime(weights_df.index)
        weights_df = weights_df.astype(float)

        weights_df.to_csv(os.path.join(save_dir, "rolling_weights.csv"))
        plot_weights_area(weights_df, save_dir)
        turnover = calculate_turnover(weights_df)
        plot_turnover(turnover, save_dir)
        avg_to = turnover.mean()
        logger.log(f"Average Monthly Turnover: {avg_to:.4f}")

    logger.log("Backtest Finished.")


if __name__ == "__main__":
    rolling_backtest("configs/spo_plus_linear.yaml")
