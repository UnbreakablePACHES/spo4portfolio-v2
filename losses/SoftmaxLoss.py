import torch.nn as nn


class SoftmaxSPOLoss(nn.Module):
    def __init__(self, temperature=1.0):
        """Create the loss module.

        Args:
            temperature: Softmax temperature controlling how sharp allocations
                are. Lower values approximate hard decisions.
        """
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=1)

    def allocator(self, cost_vec):
        """Map a cost vector to soft portfolio weights.

        Args:
            cost_vec: Tensor of shape ``(batch, assets)`` containing costs.

        Returns:
            Tensor of soft weights emphasizing lower-cost assets.
        """
        return self.softmax(-cost_vec / self.temperature)

    def forward(self, pred_cost, true_cost):
        """Compute SPO-style regret between predicted and oracle weights.

        Args:
            pred_cost: Predicted cost tensor ``(batch, assets)``.
            true_cost: Ground-truth cost tensor ``(batch, assets)``.

        Returns:
            Scalar tensor representing mean regret.
        """
        oracle_weights = self.allocator(true_cost)

        pred_weights = self.allocator(pred_cost)
        portfolio_cost_pred = (pred_weights * true_cost).sum(dim=1)
        portfolio_cost_oracle = (oracle_weights * true_cost).sum(dim=1)

        regret = portfolio_cost_pred - portfolio_cost_oracle

        return regret.mean()
