import os
import torch
import numpy as np
from datetime import datetime


class Logger:
    def __init__(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.log_path = os.path.join(save_dir, "train.log")

    def log(self, msg):
        time_str = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        line = f"{time_str} {msg}"
        print(line)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def log_experiment_setup(logger, cfg):
    """
    打印实验的基础配置信息，如 ETF 列表和训练参数
    """
    etf_list = cfg["data"]["etfs"]
    logger.log(f"--------------------------------------------------")
    logger.log(f"Target Assets ({len(etf_list)}): {etf_list}")
    logger.log(
        f"Training Params: Epochs={cfg['trainer']['epochs']}, LR={cfg['optimizer']['params']['lr']}, Device={cfg['trainer'].get('device', 'cpu')}"
    )
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
    """
    监控训练过程：
    1. 始终显示 Epoch 和 Loss
    2. 定期 (每10轮 或 首尾轮) 计算并显示 Regret (基于 Batch 中的一个样本抽样)
    """

    # 判断是否需要打印详细的 Regret 信息 (首轮、尾轮、每10轮)
    should_log_regret = epoch == 1 or epoch == total_epochs or epoch % 10 == 0

    if not should_log_regret:
        # 如果不需要详细打印，可以选择跳过，或者只在每隔几个 epoch 打印简略 loss
        # 这里为了日志清爽，我们只在特定轮次打印
        return

    regret_msg = "N/A"

    # 如果传入了模型和数据，就开始计算 Regret
    if should_log_regret and portfolio_model is not None and pred_cost is not None:
        try:
            with torch.no_grad():
                # 1. 取 Batch 中的第一个样本 (抽样计算以节省时间)
                # pred_cost 和 true_cost 都在 GPU 上，需要转成 numpy
                cp_sample = pred_cost[0].detach().cpu().numpy()
                ct_sample = true_cost[0].detach().cpu().numpy()

                # 2. 模型预测的决策 w_pred
                w_pred, _ = portfolio_model.solve(cp_sample)

                # 3. 该决策在真实环境下的 Cost (Realized Cost)
                realized_cost = ct_sample @ w_pred

                # 4. 真实环境下的最优决策 (Oracle Cost)
                _, oracle_cost = portfolio_model.solve(ct_sample)

                # 5. Regret = 实际 - 最优
                regret_val = realized_cost - oracle_cost
                regret_msg = f"{regret_val:.6f}"
        except Exception as e:
            regret_msg = f"Err({str(e)})"

    # 打印格式化日志
    logger.log(
        f"        Epoch {epoch:03d}/{total_epochs} | Loss: {avg_loss:.6f} | Regret(Sample): {regret_msg}"
    )
