from .SPOPlusLoss import SPOPlusLoss

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

    # 预留给未来的 robust 损失
    elif ltype == "robust_ro":
        raise NotImplementedError("RO loss not implemented yet")

    elif ltype == "robust_topk":
        raise NotImplementedError("top-k loss not implemented yet")

    elif ltype == "robust_knn":
        raise NotImplementedError("k-NN loss not implemented yet")

    else:
        raise ValueError(f"Unknown loss type: {ltype}")
