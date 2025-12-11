import torch
from torch.utils.data import Dataset
import pandas as pd


class PortfolioDataset(Dataset):
    def __init__(self, features_df, labels_df, num_assets):
        """
        features_df: shape = (T, N*F)
        labels_df:   shape = (T, N)
        """
        num_features = features_df.shape[1] // num_assets

        self.X = features_df.values.reshape(-1, num_assets, num_features)
        self.y = labels_df.values

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)  # (N, F)
        y = torch.tensor(self.y[idx], dtype=torch.float32)  # (N,)
        return x, y


