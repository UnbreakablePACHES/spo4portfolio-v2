import optuna
import copy
import torch
import numpy as np
import random
from datetime import datetime, timedelta

from optuna.samplers import TPESampler

from DataPipeline.factory import build_dataloader
from models.factory import build_model
from portfolio.factory import build_portfolio_model
from losses.factory import build_loss
from optimizers.factory import build_optimizer


def set_trial_seed(seed):
    """Seed Python, NumPy, and torch for deterministic trials."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_optuna_tuning(
    base_cfg, train_start_str, train_end_str, n_trials=10, logger=None, seed=42
):
    """Run Optuna search to select learning rate and epochs.

    Args:
        base_cfg: Base configuration dictionary to clone per trial.
        train_start_str: Training period start date (``YYYY-MM-DD``).
        train_end_str: Training period end date (``YYYY-MM-DD``).
        n_trials: Number of Optuna trials to run.
        logger: Optional logger for progress messages.
        seed: Random seed for reproducible search.

    Returns:
        Dictionary of best hyperparameters returned by Optuna.
    """

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

    sub_train_cfg = copy.deepcopy(base_cfg)
    sub_train_cfg["data"]["train_start"] = train_start_str
    sub_train_cfg["data"]["train_end"] = str_sub_train_end
    sub_train_cfg["model"]["params"]["num_assets"] = len(base_cfg["data"]["etfs"])

    val_cfg = copy.deepcopy(base_cfg)
    val_cfg["data"]["train_start"] = str_val_start
    val_cfg["data"]["train_end"] = train_end_str
    val_cfg["model"]["params"]["num_assets"] = len(base_cfg["data"]["etfs"])

    def objective(trial):
        set_trial_seed(seed)

        lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
        epochs = trial.suggest_int("epochs", 20, 40)  # 稍微放宽一点范围

        trial_cfg = copy.deepcopy(sub_train_cfg)
        trial_cfg["optimizer"]["params"]["lr"] = lr
        trial_cfg["trainer"]["epochs"] = epochs

        device = torch.device(trial_cfg["trainer"].get("device", "cpu"))

        try:
            t_loader = build_dataloader(trial_cfg)
            v_loader = build_dataloader(val_cfg)

            model = build_model(trial_cfg).to(device)
            optimizer = build_optimizer(trial_cfg, model)
            port_model = build_portfolio_model(trial_cfg)
            loss_fn = build_loss(trial_cfg, port_model)

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

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=seed)

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    if logger:
        logger.log(f"    [Tuning] Best Params: {study.best_params}")

    return study.best_params
