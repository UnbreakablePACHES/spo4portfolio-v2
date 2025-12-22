import torch
import torch.nn as nn
import numpy as np
import pyepo


class SPOPlusLoss(nn.Module):
    def __init__(self, opt_model):
        super().__init__()
        self.opt_model = opt_model
        # 保留原有的 PyEPO 模块，用于不带手续费的旧代码复现
        self.legacy_loss = pyepo.func.SPOPlus(opt_model)

    def forward(self, pred_cost, true_cost, prev_weights=None):
        """
        Args:
            pred_cost: (B, N)
            true_cost: (B, N)
            prev_weights: (B, N) Optional.
                          如果为 None，执行标准 SPO+ (兼容旧 Config);
                          如果不为 None，执行带 Lasso 的 Custom SPO+.
        """
        B, N = pred_cost.shape
        device = pred_cost.device

        # ==========================================================
        # 分支 1：兼容模式 (prev_weights 为 None)
        # 适用于运行 spo_plus_linear.yaml 等不带手续费配置的情况
        # ==========================================================
        if prev_weights is None:
            # 1. 先用 Gurobi 对每个样本的 true_cost 求 oracle 解
            true_sols = []
            true_objs = []
            true_cost_np = true_cost.detach().cpu().numpy()

            for i in range(B):
                # 【关键修复】分两步调用，防止报错 "solve() takes 1 positional argument"
                # 无论 opt_model 是自定义的还是 PyEPO 原生的，这种写法都兼容
                self.opt_model.setObj(true_cost_np[i])
                sol, obj = self.opt_model.solve()

                true_sols.append(sol)
                true_objs.append(obj)

            true_sols = torch.tensor(true_sols, dtype=torch.float32, device=device)
            true_objs = torch.tensor(
                true_objs, dtype=torch.float32, device=device
            ).squeeze()

            # 调用 PyEPO 原生库计算 Loss
            return self.legacy_loss(pred_cost, true_cost, true_sols, true_objs)

        # ==========================================================
        # 分支 2：新功能模式 (带手续费 Lasso)
        # 适用于 DataPipeline 返回了 prev_weights 的情况
        # ==========================================================
        else:
            # 1. 构造 SPO+ 扰动成本: c_spo = 2*pred - true
            cost_spo = 2 * pred_cost - true_cost

            # 2. 准备 Numpy 数据
            spo_cost_np = cost_spo.detach().cpu().numpy()
            true_cost_np = true_cost.detach().cpu().numpy()

            # 确定 prev_weights 不为 None 后再 detach
            prev_weights_np = prev_weights.detach().cpu().numpy()

            true_sols = []
            spo_sols = []

            # 3. 逐样本求解优化问题
            for i in range(B):
                # 求解 True Oracle: maximize return - fee (传入 -true_cost 作为 return)
                # 使用 optimize 方法（PortfolioModelWithFee 特有）
                w_true = self.opt_model.optimize(-true_cost_np[i], prev_weights_np[i])
                true_sols.append(w_true)

                # 求解 SPO Oracle: maximize return - fee (传入 -spo_cost 作为 return)
                w_spo = self.opt_model.optimize(-spo_cost_np[i], prev_weights_np[i])
                spo_sols.append(w_spo)

            true_sols = torch.tensor(true_sols, dtype=torch.float32, device=device)
            spo_sols = torch.tensor(spo_sols, dtype=torch.float32, device=device)

            # 4. 手动计算 Loss
            # Loss = Obj(w_true) - Obj(w_spo) (Minimize cost form)

            # 获取 gamma，如果模型没有 gamma 属性（旧模型混用），默认为 0
            gamma = getattr(self.opt_model, "gamma", 0.0)

            # 交易费项
            fee_true = gamma * torch.abs(true_sols - prev_weights).sum(dim=1)
            fee_spo = gamma * torch.abs(spo_sols - prev_weights).sum(dim=1)

            # 线性项 (注意: cost_spo * x)
            linear_true = (cost_spo * true_sols).sum(dim=1)
            linear_spo = (cost_spo * spo_sols).sum(dim=1)

            obj_true = linear_true + fee_true
            obj_spo = linear_spo + fee_spo

            loss = (obj_true - obj_spo).mean()

            return loss
