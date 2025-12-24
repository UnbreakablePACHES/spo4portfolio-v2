import gurobipy as gp
import numpy as np
from pyepo.model.grb.grbmodel import optGrbModel


class PortfolioModelWithFee(optGrbModel):
    # 1. 新增 gamma_l2 参数 (默认为 0，即不开启)
    def __init__(self, n_assets, gamma_l1=0.003, gamma_l2=0.42, budget=1.0, threads=1):
        self.n_assets = n_assets
        self.gamma_l1 = gamma_l1  # 原来的 gamma，现在明确叫 l1
        self.gamma_l2 = gamma_l2  # 新增的 l2

        # 为了兼容旧代码，保留 gamma 属性
        self.gamma = gamma_l1

        self._model = gp.Model()
        self._model.setParam("OutputFlag", 0)

        if threads is not None:
            self._model.setParam("Threads", threads)

        self.x = self._model.addVars(
            n_assets, lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name="x"
        )

        # L1 辅助变量 z (用于模拟绝对值)
        self.z = self._model.addVars(
            n_assets, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="z"
        )

        self.budget_constr = self._model.addConstr(
            gp.quicksum(self.x[i] for i in range(n_assets)) == budget, name="budget"
        )

        self.z_constrs = []
        super().__init__()

    def _getModel(self):
        return self._model, self.x

    def setObj(self, cost_vec, prev_weight=None):
        if prev_weight is None:
            prev_weight = np.zeros(self.n_assets)

        # 1. 更新 L1 约束 (z >= |x - prev|)
        for constr in self.z_constrs:
            self._model.remove(constr)
        self.z_constrs.clear()

        for i in range(self.n_assets):
            c1 = self._model.addConstr(self.z[i] >= self.x[i] - prev_weight[i])
            c2 = self._model.addConstr(self.z[i] >= -(self.x[i] - prev_weight[i]))
            self.z_constrs.extend([c1, c2])

        # 2. 构建目标函数
        #    Obj = Expected Return - L1_Fee - L2_Penalty

        if hasattr(cost_vec, "detach"):
            cost_vec = cost_vec.detach().cpu().numpy()

        expected_return = gp.quicksum(
            cost_vec[i] * self.x[i] for i in range(self.n_assets)
        )

        l1_cost = self.gamma_l1 * gp.quicksum(self.z[i] for i in range(self.n_assets))

        # 新增：L2 惩罚项 gamma_l2 * sum((x - prev)^2)
        # Gurobi 支持直接写二次项
        l2_cost = 0.0
        if self.gamma_l2 > 0:
            l2_cost = self.gamma_l2 * gp.quicksum(
                (self.x[i] - prev_weight[i]) * (self.x[i] - prev_weight[i])
                for i in range(self.n_assets)
            )

        # Maximize (Return - Costs)
        self._model.setObjective(-expected_return - l1_cost - l2_cost, gp.GRB.MAXIMIZE)

    def optimize(self, cost_vec, prev_weight):
        self.setObj(cost_vec, prev_weight)
        self._model.optimize()

        if self._model.Status == gp.GRB.OPTIMAL:
            sol = [self.x[i].X for i in range(self.n_assets)]
        else:
            sol = (
                list(prev_weight)
                if prev_weight is not None
                else [1.0 / self.n_assets] * self.n_assets
            )

        return sol
