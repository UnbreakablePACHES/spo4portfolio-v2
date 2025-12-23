import gurobipy as gp
from pyepo.model.grb.grbmodel import optGrbModel


class PortfolioModel(optGrbModel):
    def __init__(self, n_assets, budget=1.0, lb=0.0, ub=1.0, threads=None):
        """Create a basic long-only portfolio optimization model.

        Args:
            n_assets: Number of assets in the universe.
            budget: Total allocation budget constraint.
            lb: Lower bound per-asset weight.
            ub: Upper bound per-asset weight.
            threads: Optional thread count for Gurobi.
        """
        self.n_assets = n_assets

        if lb is None:
            lb = 0.0
        if ub is None:
            ub = 1.0

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
        """Optimize the portfolio for the provided cost vector.

        Args:
            cost_vec: Optional cost vector to set before solving. When ``None``
                the existing objective is used (useful for SPO+ calls).

        Returns:
            Tuple of ``(solution, objective_value)`` where solution is a list of
            asset weights.
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
