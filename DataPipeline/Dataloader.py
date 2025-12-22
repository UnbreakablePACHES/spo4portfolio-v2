import torch
from torch.utils.data import Dataset
import pandas as pd


class PortfolioDataset(Dataset):
    # 将 prev_weights_df 设为可选 (None)
    def __init__(self, features_df, labels_df, num_assets, prev_weights_df=None):
        num_features = features_df.shape[1] // num_assets
        self.X = features_df.values.reshape(-1, num_assets, num_features)
        self.y = labels_df.values

        # 只有传入了 prev_weights_df 才处理
        if prev_weights_df is not None:
            self.w_prev = prev_weights_df.values
        else:
            self.w_prev = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)

        # 如果有 w_prev，返回 3个值；否则返回 2个值
        if self.w_prev is not None:
            w = torch.tensor(self.w_prev[idx], dtype=torch.float32)
            return x, y, w
        else:
            return x, y
