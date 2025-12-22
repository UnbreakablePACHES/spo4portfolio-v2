import gurobipy as gp
from pyepo.model.grb.grbmodel import optGrbModel


class PortfolioModelWithFee(optGrbModel):
    def __init__(self, n_assets, gamma=0.005, budget=1.0):
        self.n_assets = n_assets
        self.gamma = gamma

        # 初始化 Gurobi 模型
        self._model = gp.Model()
        self._model.setParam("OutputFlag", 0)  # 静默模式

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

        # PyEPO 父类初始化 (虽非必须，但保持一致性)
        super().__init__()

    def _getModel(self):
        """PyEPO 接口要求的 helper 方法"""
        return self._model, self.x

    def setObj(self, cost_vec, prev_weight):
        # 1. 清除旧的手续费相关约束
        #    (注意：频繁 add/remove 在大规模求解时可能影响性能，但对于日频交易通常没问题)
        for constr in self.z_constrs:
            self._model.remove(constr)
        self.z_constrs.clear()

        # 2. 添加新的 z_i 约束：z_i ≥ |x_i - prev_i|
        #    即 z_i ≥ x_i - prev_i  且  z_i ≥ -(x_i - prev_i)
        #    这样在最小化目标函数中，z_i 会被压至 |x_i - prev_i|
        for i in range(self.n_assets):
            c1 = self._model.addConstr(self.z[i] >= self.x[i] - prev_weight[i])
            c2 = self._model.addConstr(self.z[i] >= -(self.x[i] - prev_weight[i]))
            self.z_constrs.extend([c1, c2])

        # 3. 构建目标函数
        #    Maximize: Expected Return - Transaction Costs
        #    Expected Return = sum(cost_vec[i] * x[i])  <-- 假设传入的 cost_vec 是收益率
        #    Transaction Cost = gamma * sum(z[i])
        expected_return = gp.quicksum(
            cost_vec[i] * self.x[i] for i in range(self.n_assets)
        )
        transaction_cost = self.gamma * gp.quicksum(
            self.z[i] for i in range(self.n_assets)
        )

        self._model.setObjective(expected_return - transaction_cost, gp.GRB.MAXIMIZE)

    def optimize(self, cost_vec, prev_weight):
        """
        执行求解
        Args:
            cost_vec: 预期收益向量 (注意符号: 必须是收益，如果是 Cost 请先取反)
            prev_weight: 上一期持仓权重
        Returns:
            sol: 最优权重列表
        """
        self.setObj(cost_vec, prev_weight)
        self._model.optimize()

        # 检查求解状态
        if self._model.Status == gp.GRB.OPTIMAL:
            sol = [self.x[i].X for i in range(self.n_assets)]
        else:
            # 兜底策略：如果求解失败（极少情况），保持上一期仓位或平仓
            # 这里简单返回全 0 或者上一期权重，避免程序崩溃
            print(
                f"Warning: Portfolio optimization failed with status {self._model.Status}"
            )
            sol = list(prev_weight)  # 或者返回全0: [0.0] * self.n_assets

        return sol
