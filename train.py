import torch
import yaml

from DataPipeline.factory import build_dataloader
from models.factory import build_model
from portfolio.factory import build_portfolio_model
from losses.factory import build_loss
from optimizers.factory import build_optimizer


def train(config_path: str = "configs/spo_plus_linear.yaml"):
    # 1. 读 config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. 设备 & 随机种子
    device_str = cfg.get("trainer", {}).get("device", "cpu")
    device = torch.device(device_str)
    seed = cfg.get("experiment", {}).get("seed", 42)
    torch.manual_seed(seed)

    # 3. 构建各个组件
    dataloader = build_dataloader(cfg)
    model = build_model(cfg).to(device)
    portfolio_model = build_portfolio_model(cfg)
    loss_fn = build_loss(cfg, portfolio_model=portfolio_model)
    optimizer = build_optimizer(cfg, model)

    epochs = cfg["trainer"]["epochs"]

    # 4. 训练循环
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            features = batch["features"].to(device)  # (B, N, F)
            cost = batch["cost"].to(device)         # (B, N)

            # 前向 + loss
            c_pred = -model(features)
            loss = loss_fn(c_pred, cost)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.6f}")


if __name__ == "__main__":
    train()
