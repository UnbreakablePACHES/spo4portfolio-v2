from .LinearInferencer import LinearInferencer
from .SoftmaxAllocator import SoftmaxAllocator
# 如果未来有其他模型，也在这里 import
# from .LSTMInferencer import LSTMInferencer
# from .TransformerInferencer import TransformerInferencer


def build_model(cfg):
    """
    根据 config["model"]["type"] 构建模型
    """
    mcfg = cfg["model"]
    mtype = mcfg["type"]
    params = mcfg["params"]

    if mtype == "linear":
        # === 【自动推断】 ===
        # 从 data.features 列表获取长度，不再依赖 model.params.input_dim
        # 这样 YAML 里就不用手写 '7' 了
        feat_list = cfg["data"].get("features", [])
        auto_input_dim = len(feat_list)

        return LinearInferencer(
            num_assets=len(cfg["data"]["etfs"]),
            input_dim=auto_input_dim,
        )

    elif mtype == "softmax":  
        return SoftmaxAllocator(
            num_assets=len(cfg["data"]["etfs"]),
            input_dim=len(cfg["data"]["features"]),
            hidden_dim=params.get("hidden_dim", None)
        )

    else:
        raise ValueError(f"Unknown model type: {mtype}")
