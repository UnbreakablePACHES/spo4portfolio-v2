import torch
import torch.nn as nn


class MaxReturnLoss(nn.Module):
    """Loss that minimizes portfolio cost (maximizes return)."""

    def __init__(self):
        super().__init__()

    def forward(self, weights, true_cost):
        """Compute mean portfolio cost.

        Args:
            weights: Portfolio weights ``(batch_size, num_assets)``.
            true_cost: Actual costs ``(batch_size, num_assets)`` (negative
                returns).

        Returns:
            Scalar tensor representing average portfolio cost.
        """
        portfolio_cost = (weights * true_cost).sum(dim=1)
        return portfolio_cost.mean()


class MaxSharpeLoss(nn.Module):
    """Loss that maximizes Sharpe ratio using batch statistics."""

    def __init__(self, risk_free_rate=0.0):
        super().__init__()
        self.risk_free_rate = risk_free_rate

    def forward(self, weights, true_cost):
        """Compute negative Sharpe ratio.

        Args:
            weights: Portfolio weights ``(batch_size, num_assets)``.
            true_cost: Actual costs ``(batch_size, num_assets)`` (negative
                returns).

        Returns:
            Scalar tensor representing ``-Sharpe`` for minimization.
        """
        portfolio_returns = -(weights * true_cost).sum(dim=1)

        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std(unbiased=False) + 1e-8

        sharpe_ratio = (mean_return - self.risk_free_rate) / std_return

        return -sharpe_ratio
