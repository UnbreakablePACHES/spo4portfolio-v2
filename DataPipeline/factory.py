import torch
from torch.utils.data import DataLoader
from .DataBuilder import build_dataset
from .Dataloader import PortfolioDataset


def build_dataloader(cfg):
    """Build a ``DataLoader`` according to configuration settings.

    Args:
        cfg: Experiment configuration dictionary containing ``data`` and
            ``dataloader`` fields.

    Returns:
        torch.utils.data.DataLoader configured for portfolio data.

    Raises:
        ValueError: If an unknown dataloader type is requested.
    """

    dcfg = cfg["dataloader"]
    dtype = dcfg["type"]
    params = dcfg["params"]

    if dtype == "monthly_window":
        target_features = cfg["data"].get("features", None)
        label_win = cfg["data"].get("label_window", 1)

        features_df, labels_df = build_dataset(
            tickers=cfg["data"]["etfs"],
            data_dir=cfg["data"]["root"],
            start_date=cfg["data"]["train_start"],
            end_date=cfg["data"]["train_end"],
            dropna=True,
            feature_list=target_features,
            label_window=label_win,  # <--- 传入这个参数
        )

        num_assets = len(cfg["data"]["etfs"])
        dataset = PortfolioDataset(
            features_df=features_df, labels_df=labels_df, num_assets=num_assets
        )

        def collate_fn(batch):
            xs, ys = zip(*batch)
            xs = torch.stack(xs)  # (B, N, F)
            ys = torch.stack(ys)  # (B, N)
            return {"features": xs, "cost": ys}

        return DataLoader(
            dataset,
            batch_size=params.get("batch_size", 1),
            shuffle=params.get("shuffle", False),
            collate_fn=collate_fn,
        )

    else:
        raise ValueError(f"Unknown dataloader type: {dtype}")
