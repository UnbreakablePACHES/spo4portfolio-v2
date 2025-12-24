import torch
from pyepo.func import SPOPlus


class NaiveRobustSPOLoss(torch.nn.Module):
    def __init__(self, opt_model, rho=0.01, gamma=None, num_samples=4):
        """Monte Carlo robust SPO+ loss around ground-truth costs.

        The uncertainty set follows ``U(ĉ) = {ĉ ∘ (1 + ζ): ||ζ||_∞ ≤ ρ, ||ζ||_1
        ≤ Γ}``, where ``ρ`` caps the per-asset relative error and ``Γ`` budgets
        the simultaneous deviation across all assets. When ``Γ`` is ``None``,
        only the ``ℓ∞`` cap is enforced.

        Args:
            opt_model: pyepo optimization model used to solve oracle decisions.
            rho: L-inf perturbation radius ``ρ`` applied to relative costs.
            gamma: Optional L1 budget ``Γ`` limiting total simultaneous
                deviation.
            num_samples: Number of perturbations to sample per batch for the
                worst-case approximation.
        """
        super().__init__()
        self.opt_model = opt_model
        self.loss_fn = SPOPlus(opt_model)
        self.rho = rho
        self.gamma = gamma
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
        """Compute ``x^*_{RO}(ĉ) = argmin_{x∈X} max_{c∈U(ĉ)} c^T x`` and
        approximate ``max_{c∈U(ĉ)} ĉ^T(x(ĉ) - x^*(ĉ))`` with samples.

        We sample perturbations ``ζ`` satisfying ``||ζ||_∞ ≤ ρ`` (and ``||ζ||_1
        ≤ Γ`` when provided), build perturbed costs ``ĉ ∘ (1 + ζ)``, and use
        their oracle solutions inside the SPO+ surrogate before taking the
        worst-case scenario loss.
        """
        scenario_losses = []
        for _ in range(self.num_samples):
            noise = (torch.rand_like(true_cost) * 2 - 1) * self.rho
            if self.gamma is not None:
                l1_norm = noise.abs().sum(dim=1, keepdim=True)
                scale = torch.minimum(
                    torch.tensor(1.0, device=true_cost.device),
                    self.gamma / (l1_norm + 1e-12),
                )
                noise = noise * scale

            perturbed_cost = (true_cost * (1 + noise)).detach()

            true_sols, true_objs = self._solve_batch(perturbed_cost)
            scenario_loss = self.loss_fn(
                pred_cost, perturbed_cost, true_sols, true_objs
            )
            scenario_losses.append(scenario_loss.mean())

        return torch.stack(scenario_losses).max()
