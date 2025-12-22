import gurobipy as gp
import numpy as np
from pyepo.model.grb.grbmodel import optGrbModel


class PortfolioModelWithFee(optGrbModel):
    def __init__(self, n_assets, gamma=0.003, budget=1.0, threads=1):
        self.n_assets = n_assets
        self.gamma = gamma

        # 初始化 Gurobi 模型
        self._model = gp.Model()
        self._model.setParam("OutputFlag", 0)  # 静默模式

        # 设置线程数 (对于大量小规模求解，单核通常更快)
        if threads is not None:
            self._model.setParam("Threads", threads)

        # 投资比例变量 x_i ∈ [0, 1]
        self.x = self._model.addVars(
            n_assets, lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name="x"
        )

        # 手续费辅助变量 z_i (用于模拟绝对值 |x_i - prev_i|)
        self.z = self._model.addVars(
            n_assets, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="z"
        )

        # 预算约束 sum(x) == budget
        self.budget_constr = self._model.addConstr(
            gp.quicksum(self.x[i] for i in range(n_assets)) == budget, name="budget"
        )

        # 保存手续费相关约束以便后续删除更新
        self.z_constrs = []

        # PyEPO 父类初始化
        super().__init__()

    def _getModel(self):
        """PyEPO 接口要求的 helper 方法"""
        return self._model, self.x

    def setObj(self, cost_vec, prev_weight=None):
        """
        设置目标函数。

        关键修复：将 prev_weight 设为可选参数。
        当 PyEPO 内部调用 setObj(c) 时，prev_weight 为 None。
        此时我们将其设为全 0，等价于忽略手续费变化量（惩罚项变为常数）。
        """
        # 1. 兼容性处理
        if prev_weight is None:
            prev_weight = np.zeros(self.n_assets)

        # 2. 清除旧的手续费相关约束
        for constr in self.z_constrs:
            self._model.remove(constr)
        self.z_constrs.clear()

        # 3. 添加新的 z_i 约束：z_i ≥ |x_i - prev_i|
        for i in range(self.n_assets):
            c1 = self._model.addConstr(self.z[i] >= self.x[i] - prev_weight[i])
            c2 = self._model.addConstr(self.z[i] >= -(self.x[i] - prev_weight[i]))
            self.z_constrs.extend([c1, c2])

        # 4. 构建目标函数
        #    Maximize: Expected Return - Transaction Costs
        #    Expected Return = sum(cost_vec[i] * x[i])
        #    (注意：如果 cost_vec 是成本，需要在外部取反；如果是收益率，直接用)

        # 处理可能的 tensor 输入
        if hasattr(cost_vec, "detach"):
            cost_vec = cost_vec.detach().cpu().numpy()

        expected_return = gp.quicksum(
            cost_vec[i] * self.x[i] for i in range(self.n_assets)
        )
        transaction_cost = self.gamma * gp.quicksum(
            self.z[i] for i in range(self.n_assets)
        )

        self._model.setObjective(expected_return - transaction_cost, gp.GRB.MAXIMIZE)

    def optimize(self, cost_vec, prev_weight):
        """
        自定义的优化接口，显式要求 prev_weight
        """
        self.setObj(cost_vec, prev_weight)
        self._model.optimize()

        if self._model.Status == gp.GRB.OPTIMAL:
            sol = [self.x[i].X for i in range(self.n_assets)]
        else:
            # 兜底
            sol = (
                list(prev_weight)
                if prev_weight is not None
                else [1.0 / self.n_assets] * self.n_assets
            )

        return sol
