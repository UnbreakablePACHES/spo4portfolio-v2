from .PortfolioModel import PortfolioModel
from .PortfolioModelWithFee import PortfolioModelWithFee


def build_portfolio_model(cfg):
    """
    根据 config["portfolio"] 构建组合优化模型
    """
    # 兼容两种写法: 1. cfg["portfolio"]["type"]  2. cfg["portfolio"]["name"]
    pcfg = cfg["portfolio"]
    ptype = pcfg.get("type", pcfg.get("name", "MeanVariance"))
    params = pcfg.get("params", {})

    # 获取资产数量 (通常由 Backtest 动态注入到 params 中)
    # 如果 params 里没有，尝试从 cfg["model"]["params"] 里找
    n_assets = params.get("n_assets")
    if n_assets is None:
        n_assets = cfg["model"]["params"].get("num_assets")

    if n_assets is None:
        raise ValueError("n_assets not found in config params.")

    # === 分支 1: 带手续费模型 (with_fee) ===
    # 无论是配置文件显式指定，还是为了跑实验临时切换
    if ptype == "with_fee" or "gamma" in params or "gamma_l1" in params:
        # 参数映射逻辑：
        # 1. 预算
        budget = params.get("budget", 1.0)

        # 2. 线程 (建议默认为 1)
        threads = params.get("threads", 1)

        # 3. L1 交易费系数 (兼容 'gamma' 和 'gamma_l1' 两种写法)
        # 优先读 gamma_l1，没有读 gamma，再没有默认 0.003
        gamma_l1 = params.get("gamma_l1", params.get("gamma", 0.003))

        # 4. L2 换仓惩罚系数 (默认为 0)
        gamma_l2 = params.get("gamma_l2", 0.0)

        return PortfolioModelWithFee(
            n_assets=n_assets,
            gamma_l1=gamma_l1,
            gamma_l2=gamma_l2,
            budget=budget,
            threads=threads,
        )

    # === 分支 2: 标准均值方差 (basic) ===
    elif ptype == "basic":
        return PortfolioModel(
            n_assets=n_assets,
            budget=params.get("budget", 1.0),
            lb=params.get("lb", 0.0),
            ub=params.get("ub", 1.0),
            threads=params.get("threads", 1),  # 也可以加上 threads
        )

    else:
        raise ValueError(f"Unknown portfolio model type: {ptype}")
