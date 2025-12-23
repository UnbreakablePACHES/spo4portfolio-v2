import torch.nn as nn
from .LinearInferencer import LinearInferencer
from .SoftmaxAllocator import SoftmaxAllocator


def build_model(cfg):
    """Construct a model instance from configuration.

    Args:
        cfg: Configuration dictionary containing ``model`` and ``data`` keys.

    Returns:
        nn.Module: Initialized model ready for training.

    Raises:
        ValueError: If the requested model type is unknown.
    """
    mcfg = cfg["model"]
    mtype = mcfg["type"]
    params = mcfg.get("params", {})

    if mtype == "linear":

        feature_list = cfg["data"].get("features", [])
        input_dim = len(feature_list)

        if "input_dim" in params:
            input_dim = params["input_dim"]

        return LinearInferencer(
            num_assets=len(cfg["data"]["etfs"]), input_dim=input_dim
        )

    elif mtype == "softmax":
        # 自动推断 input_dim
        feature_list = cfg["data"].get("features", [])
        input_dim = len(feature_list)

        return SoftmaxAllocator(
            num_assets=len(cfg["data"]["etfs"]),
            input_dim=input_dim,
            hidden_layers=params.get("hidden_layers", []),
            dropout_rate=params.get("dropout_rate", 0.0),
        )

    else:
        raise ValueError(f"Unknown model type: {mtype}")
