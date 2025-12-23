import torch
from pyepo.func import SPOPlus


class SPOPlusLoss(torch.nn.Module):
    def __init__(self, opt_model):
        super().__init__()
        self.opt_model = opt_model
        self.loss_fn = SPOPlus(opt_model)

    def forward(self, pred_cost, true_cost):
        """Compute SPO+ loss for the batch.

        Args:
            pred_cost: Predicted cost tensor ``(batch, assets)``.
            true_cost: Ground-truth cost tensor ``(batch, assets)``.

        Returns:
            Scalar tensor containing the SPO+ loss.
        """
        B, N = pred_cost.shape
        true_sols = []
        true_objs = []
        for i in range(B):
            ct_np = true_cost[i].detach().cpu().numpy()
            sol, obj = self.opt_model.solve(ct_np)
            true_sols.append(sol)
            true_objs.append(obj)

        true_sols = torch.tensor(
            true_sols, dtype=torch.float32, device=pred_cost.device
        )  # (B, N)
        true_objs = torch.tensor(
            true_objs, dtype=torch.float32, device=pred_cost.device
        )  # (B,)

        loss = self.loss_fn(pred_cost, true_cost, true_sols, true_objs)
        return loss
