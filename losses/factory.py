import torch.nn as nn
from .SPOPlusLoss import SPOPlusLoss
from .SoftmaxLoss import SoftmaxSPOLoss
from .MarkowitzLoss import MaxReturnLoss, MaxSharpeLoss


def build_loss(cfg, portfolio_model=None):
    """
    根据 config["loss"]["type"] 构建 loss
    """
    lcfg = cfg["loss"]
    ltype = lcfg["type"]

    if ltype == "spo_plus":
        if portfolio_model is None:
            raise ValueError("SPOPlusLoss requires a portfolio_model")
        return SPOPlusLoss(portfolio_model)
    elif ltype == "softmax_spo":  # 新增这个分支
        # 可以从 config 读取 temperature，默认为 1.0
        temp = cfg["loss"].get("params", {}).get("temperature", 1.0)
        return SoftmaxSPOLoss(temperature=temp)

    # 预留给未来的 robust 损失
    elif ltype == "robust_ro":
        raise NotImplementedError("RO loss not implemented yet")

    # Baselines
    elif ltype == "max_return":
        return MaxReturnLoss()
    elif ltype == "max_sharpe":
        return MaxSharpeLoss()
    elif ltype == "mse":
        return nn.MSELoss()

    else:
        raise ValueError(f"Unknown loss type: {ltype}")
