# losses/SPOPlusLoss.py
import torch
from pyepo.func import SPOPlus

class SPOPlusLoss(torch.nn.Module):
    def __init__(self, opt_model):
        super().__init__()
        self.opt_model = opt_model
        self.loss_fn = SPOPlus(opt_model)

    def forward(self, pred_cost, true_cost):
        """
        pred_cost: (B, N) 预测的 cost（注意如果模型输出的是收益，要先取负）
        true_cost: (B, N) 真实的 cost（同样应该是负的 log_return 或啥的）
        """
        B, N = pred_cost.shape

        # 1️⃣ 先用 Gurobi 对每个样本的 true_cost 求 oracle 解
        true_sols = []
        true_objs = []
        for i in range(B):
            ct_np = true_cost[i].detach().cpu().numpy()
            sol, obj = self.opt_model.solve(ct_np)
            true_sols.append(sol)
            true_objs.append(obj)

        true_sols = torch.tensor(true_sols, dtype=torch.float32, device=pred_cost.device)  # (B, N)
        true_objs = torch.tensor(true_objs, dtype=torch.float32, device=pred_cost.device)  # (B,)

        # 2️⃣ 一次性把整批数据喂给 SPOPlus
        loss = self.loss_fn(pred_cost, true_cost, true_sols, true_objs)
        return loss







