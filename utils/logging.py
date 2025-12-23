import os
import torch
import numpy as np
from datetime import datetime


class Logger:
    def __init__(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.log_path = os.path.join(save_dir, "train.log")

    def log(self, msg):
        """Append a timestamped message to stdout and the log file."""
        time_str = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        line = f"{time_str} {msg}"
        print(line)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def log_experiment_setup(logger, cfg):
    """Log experiment metadata such as assets and hyperparameters."""
    etf_list = cfg["data"]["etfs"]
    logger.log(f"--------------------------------------------------")
    logger.log(f"Target Assets ({len(etf_list)}): {etf_list}")
    logger.log(
        f"Training Params: Epochs={cfg['trainer']['epochs']}, LR={cfg['optimizer']['params']['lr']}, Device={cfg['trainer'].get('device', 'cpu')}"
    )
    logger.log(f"Predict Model: {cfg['model']['type']}")
    logger.log(f"Loss: {cfg['loss']['type']}")
    logger.log(f"--------------------------------------------------")


def log_training_epoch(
    logger,
    epoch,
    total_epochs,
    avg_loss,
    pred_cost=None,
    true_cost=None,
    portfolio_model=None,
):
    """Log per-epoch training metrics and occasional regret estimates."""

    should_log_regret = epoch == 1 or epoch == total_epochs or epoch % 10 == 0

    if not should_log_regret:
        return

    regret_msg = "N/A"

    if should_log_regret and portfolio_model is not None and pred_cost is not None:
        try:
            with torch.no_grad():
                cp_sample = pred_cost[0].detach().cpu().numpy()
                ct_sample = true_cost[0].detach().cpu().numpy()

                w_pred, _ = portfolio_model.solve(cp_sample)

                realized_cost = ct_sample @ w_pred

                _, oracle_cost = portfolio_model.solve(ct_sample)

                regret_val = realized_cost - oracle_cost
                regret_msg = f"{regret_val:.6f}"
        except Exception as e:
            regret_msg = f"Err({str(e)})"

    logger.log(
        f"        Epoch {epoch:03d}/{total_epochs} | Loss: {avg_loss:.6f} | Regret(Sample): {regret_msg}"
    )
