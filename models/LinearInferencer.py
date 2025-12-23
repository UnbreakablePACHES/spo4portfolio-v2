import torch.nn as nn


class LinearInferencer(nn.Module):
    """Simple linear predictor producing per-asset scores."""

    def __init__(self, num_assets, input_dim):
        super().__init__()
        self.num_assets = num_assets
        self.input_dim = input_dim

        self.linear = nn.Linear(num_assets * input_dim, num_assets)

    def forward(self, x):
        """Run the linear layer on flattened features.

        Args:
            x: Tensor of shape ``(batch_size, num_assets, input_dim)``.

        Returns:
            Tensor of shape ``(batch_size, num_assets)`` containing predicted
            returns or costs per asset.
        """
        batch_size = x.size(0)
        x = x.reshape(batch_size, self.num_assets * self.input_dim)
        return self.linear(x)
