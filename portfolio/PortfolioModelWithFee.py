import gurobipy as gp


class PortfolioModelWithFee:
    """Portfolio optimization model with linear transaction costs."""

    def __init__(self, n_assets, gamma=0.003, budget=1.0):
        """Initialize the model.

        Args:
            n_assets: Number of assets in the universe.
            gamma: Transaction-cost penalty coefficient.
            budget: Total allocation budget constraint.
        """

        self.n_assets = n_assets
        self.gamma = gamma
        self._model = gp.Model()
        self._model.setParam("OutputFlag", 0)

        self.x = self._model.addVars(
            n_assets, lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name="x"
        )

        self.z = self._model.addVars(
            n_assets, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="z"
        )

        self.budget_constr = self._model.addConstr(
            gp.quicksum(self.x[i] for i in range(n_assets)) == budget, name="budget"
        )

        self.z_constrs = []

    def setObj(self, cost_vec, prev_weight):
        """Update the objective with transaction costs and prior weights.

        Args:
            cost_vec: Expected return or cost vector for each asset.
            prev_weight: Previous allocation used to compute turnover costs.
        """

        for constr in self.z_constrs:
            self._model.remove(constr)
        self.z_constrs.clear()

        for i in range(self.n_assets):
            c1 = self._model.addConstr(self.z[i] >= self.x[i] - prev_weight[i])
            c2 = self._model.addConstr(self.z[i] >= -(self.x[i] - prev_weight[i]))
            self.z_constrs.extend([c1, c2])

        expected_return = gp.quicksum(
            cost_vec[i] * self.x[i] for i in range(self.n_assets)
        )
        transaction_cost = self.gamma * gp.quicksum(
            self.z[i] for i in range(self.n_assets)
        )
        self._model.setObjective(expected_return - transaction_cost, gp.GRB.MAXIMIZE)

    def optimize(self, cost_vec, prev_weight):
        """Solve the optimization problem for the provided cost vector.

        Args:
            cost_vec: Expected return or cost vector for each asset.
            prev_weight: Previous allocation used to compute turnover costs.

        Returns:
            List of optimized asset weights.
        """

        self.setObj(cost_vec, prev_weight)
        self._model.optimize()
        sol = [self.x[i].X for i in range(self.n_assets)]
        return sol
