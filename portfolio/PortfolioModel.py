import gurobipy as gp
from pyepo.model.grb.grbmodel import optGrbModel


class PortfolioModel(optGrbModel):
    def __init__(self, n_assets, budget=1.0, lb=0.0, ub=1.0, threads=None):
        """
        Args:
            n_assets: 资产数量
            budget: 总预算
            lb: 单个资产权重下限 (默认为 0.0)
            ub: 单个资产权重上限 (默认为 1.0)
            threads: Gurobi 线程数 (可选)
        """
        self.n_assets = n_assets

        # =========================================================
        # === 【修复】处理 None 值，防止 Gurobi 报错 ===
        # =========================================================
        if lb is None:
            lb = 0.0
        if ub is None:
            ub = 1.0
        # =========================================================

        m = gp.Model()
        m.setParam("OutputFlag", 0)

        # 如果传入了线程数设置 (用于 Optuna 并行加速)
        if threads is not None:
            m.setParam("Threads", threads)

        x = m.addVars(n_assets, lb=lb, ub=ub, vtype=gp.GRB.CONTINUOUS, name="x")
        m.addConstr(gp.quicksum(x[i] for i in range(n_assets)) == budget)

        m.modelSense = gp.GRB.MINIMIZE

        self.m = m
        self.x = x

        super().__init__()

    def _getModel(self):
        return self.m, self.x

    def setObj(self, cost_vec):
        if hasattr(cost_vec, "detach"):
            cost_vec = cost_vec.detach().cpu().numpy()
        for i, c in enumerate(cost_vec):
            self.x[i].setAttr("obj", float(c))

    def solve(self, cost_vec=None):
        """
        cost_vec=None → SPOPlus 内部调用
        cost_vec=向量 → 用户调用（如求 true_sol）
        """
        if cost_vec is not None:
            self.setObj(cost_vec)

        self.m.optimize()

        # 处理可能的无解情况
        if self.m.Status == gp.GRB.OPTIMAL:
            sol = [self.x[i].x for i in range(self.n_assets)]
            obj = self.m.objVal
        else:
            # 如果求解失败，返回均匀分布作为兜底
            sol = [1.0 / self.n_assets] * self.n_assets
            obj = 0.0

        return sol, obj
