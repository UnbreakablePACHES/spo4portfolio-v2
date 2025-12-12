import optuna
import copy
import torch
import numpy as np
from datetime import datetime, timedelta

# 引入项目组件 (假设运行目录在根目录)
from DataPipeline.factory import build_dataloader
from models.factory import build_model
from portfolio.factory import build_portfolio_model
from losses.factory import build_loss
from optimizers.factory import build_optimizer

def run_optuna_tuning(base_cfg, train_start_str, train_end_str, n_trials=10, logger=None):
    """
    在给定的时间窗口内运行 Optuna 调参。
    
    原理：
    为了找到泛化能力最好的参数，我们将传入的训练窗口 (Train Window) 再次拆分为：
    1. 子训练集 (Sub-Train): 前 80% 时间
    2. 验证集 (Validation): 后 20% 时间
    
    Optuna 会在 Sub-Train 上训练，在 Validation 上打分 (Minimize SPO Loss)。
    """
    
    # 1. 解析时间并划分 验证集 (Validation Split)
    start_dt = datetime.strptime(train_start_str, "%Y-%m-%d")
    end_dt = datetime.strptime(train_end_str, "%Y-%m-%d")
    total_days = (end_dt - start_dt).days
    
    # 留出最后 20% 的天数作为验证集，或者至少留 1 个月
    val_days = max(30, int(total_days * 0.2)) 
    sub_train_end_dt = end_dt - timedelta(days=val_days)
    val_start_dt = sub_train_end_dt + timedelta(days=1)
    
    str_sub_train_end = sub_train_end_dt.strftime("%Y-%m-%d")
    str_val_start = val_start_dt.strftime("%Y-%m-%d")
    
    if logger:
        logger.log(f"    [Tuning] Split for Optuna: Train({train_start_str} -> {str_sub_train_end}) | Val({str_val_start} -> {train_end_str})")

    # 2. 准备两份 Config (一份读子训练集，一份读验证集)
    # 必须深拷贝，否则会修改原始配置
    sub_train_cfg = copy.deepcopy(base_cfg)
    sub_train_cfg["data"]["train_start"] = train_start_str
    sub_train_cfg["data"]["train_end"] = str_sub_train_end
    # 确保 num_assets 正确
    sub_train_cfg["model"]["params"]["num_assets"] = len(base_cfg["data"]["etfs"])
    
    val_cfg = copy.deepcopy(base_cfg)
    val_cfg["data"]["train_start"] = str_val_start
    val_cfg["data"]["train_end"] = train_end_str
    val_cfg["model"]["params"]["num_assets"] = len(base_cfg["data"]["etfs"])

    # 3. 定义 Optuna 目标函数
    def objective(trial):
        # === 定义搜索空间 ===
        lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
        epochs = trial.suggest_int("epochs", 24, 36) # 调参时轮数可以少一点，节省时间
        
        # 将参数注入配置
        trial_cfg = copy.deepcopy(sub_train_cfg)
        trial_cfg["optimizer"]["params"]["lr"] = lr
        trial_cfg["trainer"]["epochs"] = epochs
        
        # 构建组件
        device = torch.device(trial_cfg["trainer"].get("device", "cpu"))
        
        try:
            # 建立 DataLoader
            t_loader = build_dataloader(trial_cfg)
            v_loader = build_dataloader(val_cfg)
            
            # 建立模型
            model = build_model(trial_cfg).to(device)
            optimizer = build_optimizer(trial_cfg, model)
            
            # Loss 需要 PortfolioModel
            port_model = build_portfolio_model(trial_cfg)
            loss_fn = build_loss(trial_cfg, port_model)
            
            # === 训练 (Sub-Train) ===
            model.train()
            for _ in range(epochs):
                for batch in t_loader:
                    feats = batch["features"].to(device)
                    cost = -batch["cost"].to(device)
                    
                    optimizer.zero_grad()
                    pred = -model(feats)
                    loss = loss_fn(pred, cost)
                    loss.backward()
                    optimizer.step()
            
            # === 验证 (Validation) ===
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in v_loader:
                    feats = batch["features"].to(device)
                    cost = -batch["cost"].to(device)
                    
                    pred = -model(feats)
                    # 使用 SPO+ Loss 作为验证指标
                    v_loss = loss_fn(pred, cost)
                    val_loss_sum += v_loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss_sum / val_batches if val_batches > 0 else float('inf')
            return avg_val_loss
            
        except Exception as e:
            # 如果这组参数导致报错（比如梯度爆炸），返回无穷大
            return float('inf')

    # 4. 开始搜索
    # 禁用 Optuna 的进度条输出，防止日志刷屏
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    if logger:
        logger.log(f"    [Tuning] Best Params: {study.best_params}")
        
    return study.best_params