import gurobipy as gp
import numpy as np
from pyepo.model.grb.grbmodel import optGrbModel


class MarkowitzModel(optGrbModel):
    def __init__(self, n_assets, risk_factor=100, budget=1.0, lb=0.0, ub=1.0):
        self.n_assets = n_assets
        self.risk_factor = risk_factor

        # 兜底默认值
        if lb is None:
            lb = 0.0
        if ub is None:
            ub = 1.0

        m = gp.Model("markowitz")
        m.setParam("OutputFlag", 0)

        # 变量 x: 各资产权重
        x = m.addVars(n_assets, lb=lb, ub=ub, vtype=gp.GRB.CONTINUOUS, name="x")

        # 约束: 权重之和 = Budget (1.0)
        m.addConstr(gp.quicksum(x[i] for i in range(n_assets)) == budget)

        self.m = m
        self.x = x
        super().__init__()

    def _getModel(self):
        return self.m, self.x

    def solve(self, pred_return, cov_matrix):
        """
        Args:
            pred_return: 预测收益率向量 (1D array)
            cov_matrix: 协方差矩阵 (2D array)
        """
        # 如果协方差没传进来，或者全为0，就无法做方差优化
        if cov_matrix is None:
            # 这种情况下只能做 Max Return (退化为线性)
            pass

        # === 构建目标函数: Min(Risk - Return) ===
        # Risk = lambda * w^T * Sigma * w
        risk_term = gp.QuadExpr()
        if cov_matrix is not None:
            for i in range(self.n_assets):
                for j in range(self.n_assets):
                    if abs(cov_matrix[i, j]) > 1e-9:
                        risk_term.add(
                            self.x[i] * self.x[j], self.risk_factor * cov_matrix[i, j]
                        )

        # Return = r^T * w (因为是Minimize，所以用 -Return)
        return_term = gp.LinExpr()
        for i in range(self.n_assets):
            # 注意: pred_return 必须是标量 float
            return_term.add(self.x[i], -float(pred_return[i]))

        self.m.setObjective(risk_term + return_term, gp.GRB.MINIMIZE)
        self.m.optimize()

        if self.m.Status == gp.GRB.OPTIMAL:
            return [self.x[i].x for i in range(self.n_assets)], self.m.objVal
        else:
            # 求解失败兜底：均匀分布
            return [1.0 / self.n_assets] * self.n_assets, 0.0
