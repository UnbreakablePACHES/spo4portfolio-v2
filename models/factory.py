from .LinearInferencer import LinearInferencer
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
        return LinearInferencer(
            num_assets=len(cfg["data"]["etfs"]),
            input_dim=params["input_dim"],
        )

    # 未来可以这样扩展：
    # elif mtype == "lstm":
    #     return LSTMInferencer(**params)

    # elif mtype == "transformer":
    #     return TransformerInferencer(**params)

    else:  
        raise ValueError(f"Unknown model type: {mtype}")
