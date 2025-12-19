import torch
import torch.nn as nn

class SoftmaxLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, weights, true_cost):
        """
        weights: (batch_size, num_assets) - 模型输出的权重
        true_cost: (batch_size, num_assets) - 真实成本 (即负收益)
        """
        # 计算组合成本: w * c
        # (batch, assets) * (batch, assets) -> sum(dim=1) -> (batch, )
        portfolio_cost = (weights * true_cost).sum(dim=1)
        
        # 目标是最小化平均成本 (即最大化平均收益)
        return portfolio_cost.mean()