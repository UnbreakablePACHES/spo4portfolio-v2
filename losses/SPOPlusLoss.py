import torch
import torch.nn as nn
import numpy as np


class SPOPlusLoss(nn.Module):
    def __init__(self, opt_model):
        super().__init__()
        self.opt_model = opt_model
        # 注意：不再需要 self.legacy_loss = pyepo.func.SPOPlus
        # 我们将手动实现统一的计算逻辑，确保符号绝对正确

    def forward(self, pred_cost, true_cost, prev_weights=None):
        """
        SPO+ Loss = Obj(w_true, c_spo) - Obj(w_spo, c_spo)
        确保结果 >= 0。

        Args:
            pred_cost: (B, N) 预测成本 (通常是 -预测收益)
            true_cost: (B, N) 真实成本 (通常是 -真实收益)
            prev_weights: (B, N) or None. 上期持仓。
        """
        B, N = pred_cost.shape
        device = pred_cost.device

        # 1. 构造 SPO+ 扰动成本
        # c_spo = 2 * pred - true
        cost_spo = 2 * pred_cost - true_cost

        # 2. 准备数据
        spo_cost_np = cost_spo.detach().cpu().numpy()
        true_cost_np = true_cost.detach().cpu().numpy()

        # 处理 prev_weights 兼容性
        if prev_weights is None:
            # 如果是旧 Config，假设无手续费，即 prev_weight 为全 0
            prev_weights_np = np.zeros((B, N))
            # 对应的 Tensor 也要是全 0
            prev_weights_t = torch.zeros_like(pred_cost)
        else:
            prev_weights_np = prev_weights.detach().cpu().numpy()
            prev_weights_t = prev_weights

        true_sols = []
        spo_sols = []

        # 3. 逐样本求解 (Solver Loop)
        for i in range(B):
            # -------------------------------------------------------------
            # 关键：统一使用 "收益" 视角调用 Solver
            # 我们的 Solver 是 Maximize (Return - Fee)
            # 这里的 Cost 是 "负收益"，所以传入 -cost 作为 Return
            # -------------------------------------------------------------

            # A) 求解 True Oracle (基于真实成本)
            # 目标: Minimize true_cost + fee <=> Maximize -true_cost - fee
            if hasattr(self.opt_model, "optimize"):
                w_true = self.opt_model.optimize(-true_cost_np[i], prev_weights_np[i])
            else:
                # 兼容旧的 PortfolioModel (只有 solve)
                self.opt_model.setObj(true_cost_np[i])
                w_true, _ = self.opt_model.solve()

            true_sols.append(w_true)

            # B) 求解 SPO Oracle (基于扰动成本)
            # 目标: Minimize spo_cost + fee <=> Maximize -spo_cost - fee
            if hasattr(self.opt_model, "optimize"):
                w_spo = self.opt_model.optimize(-spo_cost_np[i], prev_weights_np[i])
            else:
                self.opt_model.setObj(spo_cost_np[i])
                w_spo, _ = self.opt_model.solve()

            spo_sols.append(w_spo)

        # 转回 Tensor
        true_sols = torch.tensor(
            true_sols, dtype=torch.float32, device=device
        )  # (B, N)
        spo_sols = torch.tensor(spo_sols, dtype=torch.float32, device=device)  # (B, N)

        # 4. 计算 Loss
        # 公式: Loss = Obj_spo(w_true) - Obj_spo(w_spo)
        # Obj(w, c) = c^T w + gamma * ||w - w_prev||_1
        # 注意：这里的 Obj 是“成本目标”，我们要最小化它。
        # w_spo 是该目标下的最优解，所以 Obj(w_spo) 必然 <= Obj(w_true)
        # 因此 Loss 必然 >= 0。

        # 获取 gamma (如果没有则为 0)
        gamma = getattr(self.opt_model, "gamma", 0.0)

        # 计算交易费部分 (L1 Lasso)
        # 如果 prev_weights 为 0 (Legacy模式)，这里计算的是 w 的 L1 范数，
        # 但因为 sum(w)=1 且 w>=0，所以 L1 范数恒等于 1，这就变成了常数，不影响梯度。
        fee_true = gamma * torch.abs(true_sols - prev_weights_t).sum(dim=1)
        fee_spo = gamma * torch.abs(spo_sols - prev_weights_t).sum(dim=1)

        # 计算线性成本部分 (c^T w)
        # cost_spo 带有梯度，w 是常数
        linear_true = (cost_spo * true_sols).sum(dim=1)
        linear_spo = (cost_spo * spo_sols).sum(dim=1)

        # 总目标值
        obj_true = linear_true + fee_true
        obj_spo = linear_spo + fee_spo

        # 计算均值 Loss
        loss = (obj_true - obj_spo).mean()

        return loss
