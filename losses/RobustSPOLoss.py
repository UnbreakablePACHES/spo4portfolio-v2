import torch
from pyepo.func import SPOPlus


class NaiveRobustSPOLoss(torch.nn.Module):
    def __init__(self, opt_model, radius=0.01, num_samples=4):
        """Monte Carlo robust SPO+ loss around ground-truth costs.

        Args:
            opt_model: pyepo optimization model used to solve oracle decisions.
            radius: L-inf perturbation radius applied to true cost vectors.
            num_samples: Number of perturbations to sample per batch for the
                worst-case approximation.
        """
        super().__init__()
        self.opt_model = opt_model
        self.loss_fn = SPOPlus(opt_model)
        self.radius = radius
        self.num_samples = num_samples

    def _solve_batch(self, cost_batch):
        """Solve the portfolio problem for each cost vector in the batch."""
        sols = []
        objs = []
        for row in cost_batch:
            sol, obj = self.opt_model.solve(row.detach().cpu().numpy())
            sols.append(sol)
            objs.append(obj)
        device = cost_batch.device
        sols = torch.tensor(sols, dtype=torch.float32, device=device)
        objs = torch.tensor(objs, dtype=torch.float32, device=device)
        return sols, objs

    def forward(self, pred_cost, true_cost):
        """Compute worst-case SPO+ over random perturbations of true costs."""
        scenario_losses = []
        for _ in range(self.num_samples):
            noise = (torch.rand_like(true_cost) * 2 - 1) * self.radius
            perturbed_cost = (true_cost + noise).detach()

            true_sols, true_objs = self._solve_batch(perturbed_cost)
            scenario_loss = self.loss_fn(pred_cost, perturbed_cost, true_sols, true_objs)
            scenario_losses.append(scenario_loss)

        return torch.stack(scenario_losses).max()
