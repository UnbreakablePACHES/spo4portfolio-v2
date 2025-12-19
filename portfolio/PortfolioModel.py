import gurobipy as gp
from pyepo.model.grb.grbmodel import optGrbModel


class PortfolioModel(optGrbModel):
    def __init__(self, n_assets, budget=1.0, lb=0.0, ub=1.0):
        self.n_assets = n_assets

        m = gp.Model()
        m.setParam("OutputFlag", 0)

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
        sol = [self.x[i].x for i in range(self.n_assets)]
        return sol, self.m.objVal
