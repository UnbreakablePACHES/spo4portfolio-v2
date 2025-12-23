from .PortfolioModel import PortfolioModel
from .PortfolioModelWithFee import PortfolioModelWithFee


def build_portfolio_model(cfg):
    """Instantiate a portfolio optimization model from configuration.

    Args:
        cfg: Configuration dictionary containing portfolio settings.

    Returns:
        PortfolioModel or PortfolioModelWithFee depending on configuration.

    Raises:
        ValueError: If an unknown portfolio type is provided.
    """
    pcfg = cfg["portfolio"]
    ptype = pcfg["type"]
    params = pcfg["params"]

    # 动态获取资产数量
    real_num_assets = len(cfg["data"]["etfs"])

    # ========== 训练阶段：使用 SPO+ 的基本 PortfolioModel ==========
    if ptype == "basic":
        return PortfolioModel(
            n_assets=real_num_assets,
            budget=params.get("budget", 1.0),
            ub=params.get("ub", None),
            lb=params.get("lb", None),
        )

    # ========== 回测阶段：加交易费用的 PortfolioModel ==========
    elif ptype == "with_fee":
        return PortfolioModelWithFee(
            n_assets=real_num_assets,
            gamma=params.get("gamma", 0.003),
            budget=params.get("budget", 1.0),
        )

    else:
        raise ValueError(f"Unknown portfolio type: {ptype}")
