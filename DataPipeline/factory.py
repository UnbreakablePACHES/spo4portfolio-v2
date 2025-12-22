import torch
from torch.utils.data import DataLoader
from .DataBuilder import build_dataset
from .Dataloader import PortfolioDataset


def build_dataloader(cfg):
    dcfg = cfg["dataloader"]
    dtype = dcfg["type"]
    params = dcfg["params"]

    if dtype == "monthly_window":
        target_features = cfg["data"].get("features", None)

        label_win = cfg["data"].get("label_window", 1)

        features_df, labels_df, prev_weights_df = build_dataset(
            tickers=cfg["data"]["etfs"],
            data_dir=cfg["data"]["root"],
            start_date=cfg["data"]["train_start"],
            end_date=cfg["data"]["train_end"],
            dropna=True,
            feature_list=target_features,
            label_window=label_win,  # <--- 传入这个参数
        )

        num_assets = len(cfg["data"]["etfs"])

        # ② 构造 Dataset（使用 Dataloader.py 中的 PortfolioDataset）
        dataset = PortfolioDataset(
            features_df,
            labels_df,
            num_assets,
            prev_weights_df=prev_weights_df,  # 传入可能为 None 的值
        )

        # === 关键：根据 dataset 是否包含 prev 来决定 collate_fn ===
        if dataset.w_prev is not None:

            def collate_fn(batch):
                xs, ys, ws = zip(*batch)
                return {
                    "features": torch.stack(xs),
                    "cost": torch.stack(ys),
                    "prev_weights": torch.stack(ws),  # 新增 Key
                }

        else:
            # 旧逻辑，保持完全不变
            def collate_fn(batch):
                xs, ys = zip(*batch)
                return {"features": torch.stack(xs), "cost": torch.stack(ys)}

        # ④ 返回 PyTorch DataLoader
        return DataLoader(
            dataset,
            batch_size=params.get("batch_size", 1),
            shuffle=params.get("shuffle", False),
            collate_fn=collate_fn,
        )

    else:
        raise ValueError(f"Unknown dataloader type: {dtype}")
