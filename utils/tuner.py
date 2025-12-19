import optuna
import copy
import torch
import numpy as np
import random
from datetime import datetime, timedelta

# 引入特定采样器用于固定种子
from optuna.samplers import TPESampler

from DataPipeline.factory import build_dataloader
from models.factory import build_model
from portfolio.factory import build_portfolio_model
from losses.factory import build_loss
from optimizers.factory import build_optimizer


# === 新增：本地的种子设置函数 ===
def set_trial_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_optuna_tuning(
    base_cfg, train_start_str, train_end_str, n_trials=10, logger=None, seed=42
):
    """
    增加了 seed 参数，用于固定 Optuna 的随机性
    """

    # 1. 解析时间并划分 验证集
    start_dt = datetime.strptime(train_start_str, "%Y-%m-%d")
    end_dt = datetime.strptime(train_end_str, "%Y-%m-%d")
    total_days = (end_dt - start_dt).days

    val_days = max(30, int(total_days * 0.2))
    sub_train_end_dt = end_dt - timedelta(days=val_days)
    val_start_dt = sub_train_end_dt + timedelta(days=1)

    str_sub_train_end = sub_train_end_dt.strftime("%Y-%m-%d")
    str_val_start = val_start_dt.strftime("%Y-%m-%d")

    if logger:
        logger.log(
            f"    [Tuning] Split for Optuna: Train({train_start_str} -> {str_sub_train_end}) | Val({str_val_start} -> {train_end_str})"
        )

    # 2. 准备 Config
    sub_train_cfg = copy.deepcopy(base_cfg)
    sub_train_cfg["data"]["train_start"] = train_start_str
    sub_train_cfg["data"]["train_end"] = str_sub_train_end
    sub_train_cfg["model"]["params"]["num_assets"] = len(base_cfg["data"]["etfs"])

    val_cfg = copy.deepcopy(base_cfg)
    val_cfg["data"]["train_start"] = str_val_start
    val_cfg["data"]["train_end"] = train_end_str
    val_cfg["model"]["params"]["num_assets"] = len(base_cfg["data"]["etfs"])

    # 3. 定义目标函数
    def objective(trial):
        # === 【关键点 1】在 Trial 内部重置种子 ===
        # 这样确保无论 Optuna 试第几次，模型权重的初始化都是一样的
        # 从而公平地比较 lr 和 epochs 的效果，而不是比较运气
        set_trial_seed(seed)

        # 定义搜索空间
        lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
        epochs = trial.suggest_int("epochs", 20, 40)  # 稍微放宽一点范围

        trial_cfg = copy.deepcopy(sub_train_cfg)
        trial_cfg["optimizer"]["params"]["lr"] = lr
        trial_cfg["trainer"]["epochs"] = epochs

        device = torch.device(trial_cfg["trainer"].get("device", "cpu"))

        try:
            # 建立 DataLoader (set_seed 也会影响这里的 shuffle)
            t_loader = build_dataloader(trial_cfg)
            v_loader = build_dataloader(val_cfg)

            model = build_model(trial_cfg).to(device)
            optimizer = build_optimizer(trial_cfg, model)
            port_model = build_portfolio_model(trial_cfg)
            loss_fn = build_loss(trial_cfg, port_model)

            # 训练
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

            # 验证
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in v_loader:
                    feats = batch["features"].to(device)
                    cost = -batch["cost"].to(device)
                    pred = -model(feats)
                    v_loss = loss_fn(pred, cost)
                    val_loss_sum += v_loss.item()
                    val_batches += 1

            return val_loss_sum / val_batches if val_batches > 0 else float("inf")

        except Exception:
            return float("inf")

    # 4. 开始搜索
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # === 【关键点 2】固定 Optuna 的采样器种子 ===
    # 这样 Optuna 每次运行建议的 lr 序列都是一模一样的
    sampler = TPESampler(seed=seed)

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    if logger:
        logger.log(f"    [Tuning] Best Params: {study.best_params}")

    return study.best_params
