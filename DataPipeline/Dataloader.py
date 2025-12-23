import torch
from torch.utils.data import Dataset
import pandas as pd


class PortfolioDataset(Dataset):
    """Dataset wrapper that reshapes flat features into asset-feature blocks."""

    def __init__(self, features_df, labels_df, num_assets):
        """Initialize the dataset.

        Args:
            features_df: DataFrame shaped ``(T, N * F)`` containing flattened
                asset features.
            labels_df: DataFrame shaped ``(T, N)`` containing target costs or
                returns per asset.
            num_assets: Number of assets represented in the columns.
        """
        num_features = features_df.shape[1] // num_assets

        self.X = features_df.values.reshape(-1, num_assets, num_features)
        self.y = labels_df.values

    def __len__(self):
        """Return the total number of samples."""
        return len(self.X)

    def __getitem__(self, idx):
        """Fetch a single sample.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple ``(features, labels)`` with shapes ``(N, F)`` and ``(N,)``
            respectively.
        """
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y
