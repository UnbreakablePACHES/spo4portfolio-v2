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
        """
        修正后的逻辑：
        计算 perturbed_pred 在真实环境 true_cost 下的 SPO+ Loss，
        取所有采样中最差（最大）的那个 Loss 作为当前 batch 的 Loss。
        """
        scenario_losses = []
        for _ in range(self.num_samples):
            # 1. 生成扰动噪声
            noise = (torch.rand_like(pred_cost) * 2 - 1) * self.rho
            if self.gamma is not None:
                l1_norm = noise.abs().sum(dim=1, keepdim=True)
                scale = torch.minimum(
                    torch.tensor(1.0, device=pred_cost.device),
                    self.gamma / (l1_norm + 1e-12),
                )
                noise = noise * scale

            # 2. 扰动预测值 (假设不确定性在预测侧)
            perturbed_pred_cost = pred_cost * (1 + noise)

            # 3. 关键修正：计算 Loss 必须基于 true_cost
            # 注意：SPOPlus 的参数通常是 (pred, true, true_sol, true_obj)
            # 我们需要计算的是：如果我预测了 perturbed_pred_cost，在面对 true_cost 时会有多大后悔？

            # 为了计算 SPO+，我们需要真实成本下的最优解 (缓存以加速)
            # 如果 SPOPlus 类内部没有缓存机制，这里需要手动解 true_cost
            # 但通常 SPOPlus 的 forward 接收 true_sols 和 true_objs 是对应 true_cost 的

            # 下面这一步是在解 true_cost 的最优解（Ground Truth 的最优解）
            # 这部分通常在循环外做一次即可，因为 true_cost 在循环中是不变的！
            # 但为了适配你的代码结构，我先写在循环里，优化时应移到循环外。
            true_sols, true_objs = self._solve_batch(true_cost)

            # 计算 perturbed_pred 和 true_cost 之间的损失
            scenario_loss = self.loss_fn(
                perturbed_pred_cost, true_cost, true_sols, true_objs
            )
            scenario_losses.append(scenario_loss.mean())

        # 取最差情况（Max Loss）进行优化
        return torch.stack(scenario_losses).max()
