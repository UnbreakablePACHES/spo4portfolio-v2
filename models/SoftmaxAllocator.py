import torch
import torch.nn as nn


class SoftmaxAllocator(nn.Module):
    """Deep network that outputs portfolio weights via softmax."""

    def __init__(self, num_assets, input_dim, hidden_layers=None, dropout_rate=0.0):
        """Create the allocator network.

        Args:
            num_assets: Number of tradable assets.
            input_dim: Number of features per asset.
            hidden_layers: Optional list of hidden layer widths. Empty or
                ``None`` keeps the model linear.
            dropout_rate: Dropout rate applied after each hidden layer.
        """
        super().__init__()

        hidden_layers = hidden_layers or []
        layers = []
        current_dim = num_assets * input_dim

        layers.append(nn.BatchNorm1d(current_dim))

        if hidden_layers:
            for h_dim in hidden_layers:
                layers.append(nn.Linear(current_dim, h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout_rate))
                current_dim = h_dim

        layers.append(nn.Linear(current_dim, num_assets))
        layers.append(nn.Softmax(dim=1))

        self.net = nn.Sequential(*layers)
        self.num_assets = num_assets
        self.input_dim = input_dim

    def forward(self, x):
        """Generate normalized weights from input features.

        Args:
            x: Tensor of shape ``(batch_size, num_assets, input_dim)``.

        Returns:
            Tensor of shape ``(batch_size, num_assets)`` representing portfolio
            weights that sum to one.
        """
        batch_size = x.size(0)
        x_flat = x.reshape(batch_size, -1)

        weights = self.net(x_flat)

        weights = torch.nan_to_num(weights, nan=1.0 / self.num_assets)
        weights = torch.clamp(weights, min=1e-8, max=1.0)

        return weights
