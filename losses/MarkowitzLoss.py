import torch
import torch.nn as nn


class MaxReturnLoss(nn.Module):
    """
    最大化收益 Loss (等价于最小化 Portfolio Cost)
    """

    def __init__(self):
        super().__init__()

    def forward(self, weights, true_cost):
        """
        weights: (batch_size, num_assets)
        true_cost: (batch_size, num_assets) - 真实成本 (负收益)
        """
        # Portfolio Cost = w^T * c
        # 目标是最小化 Cost (即最大化收益)
        portfolio_cost = (weights * true_cost).sum(dim=1)
        return portfolio_cost.mean()


class MaxSharpeLoss(nn.Module):
    """
    最大化夏普比率 Loss
    注意：这里的 std 是基于 Batch 的波动率计算的。
    """

    def __init__(self, risk_free_rate=0.0):
        super().__init__()
        self.risk_free_rate = risk_free_rate

    def forward(self, weights, true_cost):
        """
        weights: (batch_size, num_assets)
        true_cost: (batch_size, num_assets) - 真实成本 (负收益)
        """
        # 1. 将 Cost 转换为 Return: r = -c
        # (Batch, Assets) * (Batch, Assets) -> (Batch,)
        # 注意这里有一个负号
        portfolio_returns = -(weights * true_cost).sum(dim=1)

        # 2. 计算 Batch 内的均值和标准差
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std(unbiased=False) + 1e-8

        # 3. 计算 Sharpe
        sharpe_ratio = (mean_return - self.risk_free_rate) / std_return

        # 4. 目标是最大化 Sharpe，即最小化 -Sharpe
        return -sharpe_ratio
