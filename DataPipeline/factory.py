import torch
from torch.utils.data import DataLoader
from .DataBuilder import build_dataset
from .Dataloader import PortfolioDataset


def build_dataloader(cfg):

    # 读取 config
    dcfg = cfg["dataloader"]
    dtype = dcfg["type"]
    params = dcfg["params"]

    if dtype == "monthly_window":

        # ① 调用 DataBuilder 构建 features_df, labels_df
        features_df, labels_df = build_dataset(
            tickers=cfg["data"]["etfs"],
            data_dir=cfg["data"]["root"],
            start_date=cfg["data"]["train_start"],
            end_date=cfg["data"]["train_end"],
            dropna=True
        )

        num_assets = len(cfg["data"]["etfs"])

        # ② 构造 Dataset（使用 Dataloader.py 中的 PortfolioDataset）
        dataset = PortfolioDataset(
            features_df=features_df,
            labels_df=labels_df,
            num_assets=num_assets
        )

        # ③ collate_fn：把返回的 (x,y) 封装成 dict
        def collate_fn(batch):
            xs, ys = zip(*batch)
            xs = torch.stack(xs)  # (B, N, F)
            ys = torch.stack(ys)  # (B, N)
            return {"features": xs, "cost": ys}

        # ④ 返回 PyTorch DataLoader
        return DataLoader(
            dataset,
            batch_size=params.get("batch_size", 1),
            shuffle=params.get("shuffle", False),
            collate_fn=collate_fn
        )

    else:
        raise ValueError(f"Unknown dataloader type: {dtype}")

