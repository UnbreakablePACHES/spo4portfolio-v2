import torch.nn as nn


class SoftmaxSPOLoss(nn.Module):
    def __init__(self, temperature=1.0):
        """
        Args:
            temperature (float): 温度系数。
                越小 (e.g. 0.1) 越接近 argmax/argmin (硬决策)；
                越大 (e.g. 10.0) 分布越平滑 (均匀分布)。
        """
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=1)

    def allocator(self, cost_vec):
        """
        求解器代理函数 g(y)。
        因为是最小化成本，所以权重倾向于成本小的资产。
        Weight ~ exp(-Cost / T)
        """
        # 注意这里取负号，因为 Softmax 默认是把大的数值变大概率，
        # 而我们要的是成本越小权重越大。
        return self.softmax(-cost_vec / self.temperature)

    def forward(self, pred_cost, true_cost):
        """
        pred_cost: (B, N) 预测成本
        true_cost: (B, N) 真实成本
        """
        # 1. 计算 Oracle 权重 (基于真实成本的软最优解) w*
        #    这是我们希望模型达到的目标分布
        oracle_weights = self.allocator(true_cost)

        # 2. 计算 预测 权重 (基于预测成本的软决策) ŵ
        pred_weights = self.allocator(pred_cost)

        # 3. 计算 Regret (后悔值/额外成本)
        #    Regret = 预测决策的真实成本 - 最优决策的真实成本
        #    我们希望最小化这个差值 (Regret >= 0)
        #    公式: (ŵ - w*) * c_true

        # 这里的 true_cost 必须对应 (B, N)
        portfolio_cost_pred = (pred_weights * true_cost).sum(dim=1)
        portfolio_cost_oracle = (oracle_weights * true_cost).sum(dim=1)

        regret = portfolio_cost_pred - portfolio_cost_oracle

        return regret.mean()
