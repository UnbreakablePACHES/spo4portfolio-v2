import torch
import yaml
import os
import numpy as np
import pandas as pd
from datetime import timedelta

from DataPipeline.factory import build_dataloader
from DataPipeline.DataBuilder import build_dataset  # 直接导入 dataset 构建函数以便获取测试数据
from models.factory import build_model
from portfolio.factory import build_portfolio_model
from losses.factory import build_loss
from optimizers.factory import build_optimizer
from utils.logging import Logger

# 设置随机种子
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def rolling_run(config_path: str = "configs/spo_plus_linear.yaml"):
    # ================================
    # 1. 基础配置加载
    # ================================
    with open(config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    exp_name = base_cfg["experiment"]["name"] + "_rolling"
    save_dir = os.path.join("outputs", exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    logger = Logger(save_dir)
    logger.log(f"Rolling Backtest Started: {exp_name}")

    device = torch.device(base_cfg["trainer"].get("device", "cpu"))
    
    # ================================
    # 2. 定义回测时间轴 (例如回测 2024 全年)
    # ================================
    # 注意：请确保这些日期在你的 CSV 数据范围内
    start_test_date = pd.Timestamp("2024-01-01") 
    end_test_date = pd.Timestamp("2024-12-31")
    
    # 生成每月的开始日期 (Month Start)
    test_months = pd.date_range(start_test_date, end_test_date, freq='MS')

    # 用于存储每月的权重和收益
    all_weights = []
    all_returns = []

    # ================================
    # 3. 滚动循环
    # ================================
    for current_month in test_months:
        # -------------------------------------------
        # A. 计算时间窗口
        # -------------------------------------------
        # 训练集结束时间 = 当前测试月开始的前一天
        train_end_dt = current_month - timedelta(days=1)
        # 训练集开始时间 = 训练结束时间向前推 1 年
        train_start_dt = train_end_dt - pd.DateOffset(years=1) + timedelta(days=1)
        
        # 测试月结束时间 (用于截取当月数据算收益)
        test_month_end = current_month + pd.offsets.MonthEnd(0)

        str_train_start = train_start_dt.strftime("%Y-%m-%d")
        str_train_end = train_end_dt.strftime("%Y-%m-%d")
        str_test_month = current_month.strftime("%Y-%m")

        logger.log(f"\n>>> Processing Month: {str_test_month}")
        logger.log(f"    Train Window: {str_train_start} -> {str_train_end}")

        # -------------------------------------------
        # B. 动态更新 Config 并构建环境
        # -------------------------------------------
        # ⚠️ 关键：每次循环更新训练时间段
        base_cfg["data"]["train_start"] = str_train_start
        base_cfg["data"]["train_end"] = str_train_end
        
        # 每次重新设置种子，保证可复现性
        set_seed(base_cfg["experiment"]["seed"])

        # 1. 构建 Dataloader (训练集)
        train_loader = build_dataloader(base_cfg)

        # 2. 构建模型 (每次都是新模型，从头训练)
        model = build_model(base_cfg).to(device)
        
        # 3. 构建优化器和损失函数
        portfolio_model = build_portfolio_model(base_cfg) # Gurobi 模型
        loss_fn = build_loss(base_cfg, portfolio_model)
        optimizer = build_optimizer(base_cfg, model)

        # -------------------------------------------
        # C. 训练模型 (Train Loop)
        # -------------------------------------------
        model.train()
        epochs = base_cfg["trainer"]["epochs"]
        
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for batch in train_loader:
                features = batch["features"].to(device)
                c_true = -batch["cost"].to(device) # cost = -return

                # Forward
                r_pred = model(features)
                c_pred = -r_pred

                # Loss
                loss = loss_fn(c_pred, c_true)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # (可选) 打印最后一个 epoch 的 loss
            if epoch == epochs:
                logger.log(f"    Final Train Loss (Epoch {epoch}): {total_loss:.6f}")

        # -------------------------------------------
        # D. 预测与调仓 (Rebalance)
        # -------------------------------------------
        model.eval()

        # 获取测试月的数据 (用于提取特征进行预测，以及计算真实收益)
        # 这里直接调用 DataBuilder 获取 DataFrame
        df_features, df_labels = build_dataset(
            tickers=base_cfg["data"]["etfs"],
            data_dir=base_cfg["data"]["root"],
            start_date=current_month.strftime("%Y-%m-%d"),
            end_date=test_month_end.strftime("%Y-%m-%d"),
            dropna=True
        )

        if df_features.empty:
            logger.log(f"    [Warning] No data found for {str_test_month}, skipping...")
            continue

        # == 策略执行 ==
        # 假设在月初（测试数据的第一天）根据当时的特征决定当月的权重
        # 取第一行数据作为 Input
        first_day_feature = df_features.iloc[0].values 
        num_assets = len(base_cfg["data"]["etfs"])
        input_dim = first_day_feature.shape[0] // num_assets

        # 构造输入 Tensor (Batch=1)
        x_input = torch.tensor(first_day_feature, dtype=torch.float32).reshape(1, num_assets, input_dim).to(device)

        # 预测 Cost
        with torch.no_grad():
            r_pred_ts = model(x_input)
            c_pred_np = (-r_pred_ts)[0].cpu().numpy()

        # 求解优化问题得到权重 w
        w_opt, _ = portfolio_model.solve(c_pred_np)
        
        logger.log(f"    Decided Weights: {np.round(w_opt, 4)}")
        
        # -------------------------------------------
        # E. 计算当月绩效
        # -------------------------------------------
        # 当月每一天的真实收益率
        daily_returns = df_labels.values # (T, N)
        # 投资组合日收益 = 资产日收益 @ 权重 (假设当月权重不变)
        portfolio_daily_ret = daily_returns @ w_opt
        
        # 记录
        monthly_cum_ret = np.sum(portfolio_daily_ret) # 简单累加对数收益，或者用 exp 计算复利
        all_returns.extend(portfolio_daily_ret)
        
        # 保存权重记录
        record = {"Date": current_month.strftime("%Y-%m-%d")}
        for idx, ticker in enumerate(base_cfg["data"]["etfs"]):
            record[ticker] = w_opt[idx]
        all_weights.append(record)
        
        logger.log(f"    Monthly Return: {monthly_cum_ret:.4f}")

    # ================================
    # 4. 保存最终结果
    # ================================
    # 保存权重 CSV
    weights_df = pd.DataFrame(all_weights).set_index("Date")
    weights_path = os.path.join(save_dir, "rolling_weights.csv")
    weights_df.to_csv(weights_path)
    logger.log(f"Rolling weights saved to {weights_path}")

    # 保存收益曲线 CSV
    returns_df = pd.DataFrame({"daily_return": all_returns})
    returns_df["cumulative_return"] = returns_df["daily_return"].cumsum()
    returns_path = os.path.join(save_dir, "rolling_performance.csv")
    returns_df.to_csv(returns_path)
    logger.log(f"Performance saved to {returns_path}")

if __name__ == "__main__":
    rolling_run()