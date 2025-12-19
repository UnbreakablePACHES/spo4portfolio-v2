import gurobipy as gp
from pyepo.model.grb.grbmodel import optGrbModel

import gurobipy as gp
from pyepo.model.grb.grbmodel import optGrbModel


class PortfolioModelWithFee:
    def __init__(self, n_assets, gamma=0.003, budget=1.0):
        self.n_assets = n_assets
        self.gamma = gamma
        self._model = gp.Model()
        self._model.setParam("OutputFlag", 0)  # 静默模式

        # 投资比例变量 x_i ∈ [0, 1]
        self.x = self._model.addVars(
            n_assets, lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name="x"
        )

        # 手续费辅助变量 z_i ≥ |x_i - prev_i|
        self.z = self._model.addVars(
            n_assets, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="z"
        )

        # 预算约束
        self.budget_constr = self._model.addConstr(
            gp.quicksum(self.x[i] for i in range(n_assets)) == budget, name="budget"
        )

        # 保存手续费相关约束以便后续删除更新
        self.z_constrs = []

    def setObj(self, cost_vec, prev_weight):
        # 清除旧的手续费相关约束（保留预算约束）
        for constr in self.z_constrs:
            self._model.remove(constr)
        self.z_constrs.clear()

        # 添加新的 z_i 约束：z_i ≥ |x_i - prev_i|
        for i in range(self.n_assets):
            c1 = self._model.addConstr(self.z[i] >= self.x[i] - prev_weight[i])
            c2 = self._model.addConstr(self.z[i] >= -(self.x[i] - prev_weight[i]))
            self.z_constrs.extend([c1, c2])

        # 构建目标函数：期望收益 - γ × 手续费
        expected_return = gp.quicksum(
            cost_vec[i] * self.x[i] for i in range(self.n_assets)
        )
        transaction_cost = self.gamma * gp.quicksum(
            self.z[i] for i in range(self.n_assets)
        )
        self._model.setObjective(expected_return - transaction_cost, gp.GRB.MAXIMIZE)

    def optimize(self, cost_vec, prev_weight):
        self.setObj(cost_vec, prev_weight)
        self._model.optimize()
        sol = [self.x[i].X for i in range(self.n_assets)]
        return sol
