from .SPOPlusLoss import SPOPlusLoss
from .SoftmaxLoss import SoftmaxSPOLoss
from .RobustSPOLoss import NaiveRobustSPOLoss
from .MarkowitzLoss import MaxReturnLoss, MaxSharpeLoss


def build_loss(cfg, portfolio_model=None):
    """Construct the loss function specified in configuration.

    Args:
        cfg: Configuration dictionary containing ``loss`` settings.
        portfolio_model: Optional portfolio model required by some losses.

    Returns:
        Initialized loss module.

    Raises:
        ValueError: If the loss type is unsupported or missing dependencies.
        NotImplementedError: For placeholder robust losses.
    """
    lcfg = cfg["loss"]
    ltype = lcfg["type"]

    if ltype == "spo_plus":
        if portfolio_model is None:
            raise ValueError("SPOPlusLoss requires a portfolio_model")
        return SPOPlusLoss(portfolio_model)

    elif ltype == "softmax_spo":
        temp = cfg["loss"].get("params", {}).get("temperature", 1.0)
        return SoftmaxSPOLoss(temperature=temp)

    elif ltype == "robust_ro":
        if portfolio_model is None:
            raise ValueError("NaiveRobustSPOLoss requires a portfolio_model")
        params = lcfg.get("params", {})
        rho = params.get("rho", params.get("radius", 0.01))
        gamma = params.get("gamma", None)
        num_samples = params.get("num_samples", 4)
        return NaiveRobustSPOLoss(
            portfolio_model, rho=rho, gamma=gamma, num_samples=num_samples
        )

    elif ltype == "robust_topk":
        raise NotImplementedError("top-k loss not implemented yet")

    elif ltype == "robust_knn":
        raise NotImplementedError("k-NN loss not implemented yet")

    elif ltype == "max_return":
        return MaxReturnLoss()
    elif ltype == "max_sharpe":
        return MaxSharpeLoss()

    else:
        raise ValueError(f"Unknown loss type: {ltype}")
