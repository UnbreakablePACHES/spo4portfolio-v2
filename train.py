import torch
import yaml
import os
import numpy as np
import random

from DataPipeline.factory import build_dataloader
from models.factory import build_model
from portfolio.factory import build_portfolio_model
from losses.factory import build_loss
from optimizers.factory import build_optimizer

from utils.logging import Logger
from utils.recorders import LossRecorder, WeightRecorder, RegretRecorder
from utils.plotting import plot_curve
from datetime import datetime

# =========== 全局 Seed 控制（确保可复现） ===========
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(config_path: str = "configs/spo_plus_linear.yaml"):

    # ================================
    # 加载 config
    # ================================
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ================================
    # 创建保存目录 + Logger
    # ================================
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    exp_name = cfg["experiment"]["name"]
    save_dir = os.path.join("outputs", exp_name + "_" + timestamp)
    os.makedirs(save_dir, exist_ok=True)

    logger = Logger(save_dir)
    logger.log(f"Experiment: {exp_name}")

    # ================================
    # 设备 & 随机种子
    # ================================
    
    device = torch.device(cfg["trainer"].get("device", "cpu"))
    seed = cfg["experiment"].get("seed", 123)
    set_seed(seed)

    logger.log(f"Device: {device}")
    logger.log(f"Seed: {seed}")

    # ================================
    # 构建 dataloader / model / loss / optimizer
    # ================================
    dataloader = build_dataloader(cfg)
    logger.log("Dataloader built.")

    model = build_model(cfg).to(device)
    logger.log("Model built.")

    portfolio_model = build_portfolio_model(cfg)
    logger.log("Portfolio model built.")

    loss_fn = build_loss(cfg, portfolio_model)
    optimizer = build_optimizer(cfg, model)
    logger.log("Loss + Optimizer built.")

    # ========== Recorder ==========
    loss_recorder = LossRecorder(save_dir)
    weight_recorder = WeightRecorder(save_dir, num_assets=cfg["portfolio"]["params"]["num_assets"])
    regret_recorder = RegretRecorder(save_dir)

    epochs = cfg["trainer"]["epochs"]

    # ================================
    # Training Loop
    # ================================
    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        num_batches = 0
        regrets = []

        for batch in dataloader:
            features = batch["features"].to(device)
            r_true = batch["cost"].to(device)   # 这里其实是 log_return
            c_true = -r_true                    # ✅ 统一成“要被最小化的 cost”

            # ---- 预测 ----
            r_pred = model(features)            # 模型输出预测收益
            c_pred = -r_pred                    # ✅ 转成 cost

            # ---- SPO+ loss ----
            loss = loss_fn(c_pred, c_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # ---- Regret 计算（基于 true cost / pred cost）----
            c_pred_np = c_pred[0].detach().cpu().numpy()
            c_true_np = c_true[0].detach().cpu().numpy()

            # predicted decision
            w_pred, obj_pred = portfolio_model.solve(c_pred_np)
            # oracle decision
            w_true, obj_true = portfolio_model.solve(c_true_np)

            # 对于“最小化问题”，regret = obj_pred - obj_true
            regret = obj_pred - obj_true
            regrets.append(regret)

        avg_loss = total_loss / num_batches
        avg_regret = sum(regrets) / len(regrets)

        loss_recorder.add(epoch, avg_loss)
        regret_recorder.add(epoch, avg_regret)
        weight_recorder.add(epoch, w_pred)

        msg = f"Epoch {epoch}/{epochs} | avg_loss={avg_loss:.6f} | avg_regret={avg_regret:.6f}"
        print(msg)
        logger.log(msg)

    # ================================
    # 保存模型
    # ================================
    model_path = os.path.join(save_dir, "model_final.pt")
    torch.save(model.state_dict(), model_path)
    logger.log(f"Training finished. Model saved to: {model_path}")

    # ================================
    # 绘制曲线（loss / regret）
    # ================================
    plot_curve(os.path.join(save_dir, "losses.csv"),
               os.path.join(save_dir, "loss.png"),
               title="Training Loss")

    plot_curve(os.path.join(save_dir, "regrets.csv"),
               os.path.join(save_dir, "regret.png"),
               title="Training Regret")

    logger.log("Plots saved (loss, regret).")


if __name__ == "__main__":
    train()

