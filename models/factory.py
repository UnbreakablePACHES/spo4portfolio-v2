import torch.nn as nn
from .LinearInferencer import LinearInferencer
from .SoftmaxAllocator import SoftmaxAllocator


def build_model(cfg):
    mcfg = cfg["model"]
    mtype = mcfg["type"]
    params = mcfg.get("params", {})  # 获取 params，如果没有则为空字典

    if mtype == "linear":
        # 自动推断 input_dim
        feature_list = cfg["data"].get("features", [])
        input_dim = len(feature_list)

        # 兼容旧配置：如果 config 里硬编码了 input_dim，优先使用
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
            # ====================================================
            # 【核心修复】这里必须改用 hidden_layers，不能再传 hidden_dim
            # ====================================================
            hidden_layers=params.get("hidden_layers", []),
            dropout_rate=params.get("dropout_rate", 0.0),
        )

    else:
        raise ValueError(f"Unknown model type: {mtype}")
