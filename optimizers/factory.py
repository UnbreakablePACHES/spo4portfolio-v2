import torch


def build_optimizer(cfg, model):
    """Create an optimizer from configuration.

    Args:
        cfg: Configuration dictionary containing optimizer settings.
        model: Model whose parameters will be optimized.

    Returns:
        torch.optim.Optimizer instance.

    Raises:
        ValueError: If the optimizer type is unknown.
    """

    ocfg = cfg["optimizer"]
    otype = ocfg["type"]
    params = ocfg.get("params", {})

    if otype == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=params.get("lr", 1e-3),
            weight_decay=params.get("weight_decay", 0.0),
        )

    elif otype == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=params.get("lr", 0.01),
            momentum=params.get("momentum", 0.0),
            weight_decay=params.get("weight_decay", 0.0),
        )

    elif otype == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=params.get("lr", 1e-3),
            weight_decay=params.get("weight_decay", 0.01),
        )

    else:
        raise ValueError(f"Unknown optimizer type: {otype}")
