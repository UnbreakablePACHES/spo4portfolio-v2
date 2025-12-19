import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def calculate_turnover(weights_df):
    """
    计算换手率 (Turnover)
    Turnover = sum(|w_t - w_{t-1}|) / 2
    """
    # 简单的换手率计算：前后两期权重差的绝对值之和
    diff = weights_df.diff().abs().sum(axis=1).fillna(0.0)
    # 通常双边交易量除以2为单边换手率
    turnover = diff / 2.0
    return turnover


def plot_weights_area(weights_df, save_dir):
    """
    画出权重的堆积面积图 (Stacked Area Plot)
    """
    plt.figure(figsize=(12, 6))

    # 绘制堆积图
    weights_df.plot.area(figsize=(12, 6), cmap="tab20", alpha=0.8)

    plt.title("Portfolio Allocation Over Time")
    plt.ylabel("Weight")
    plt.xlabel("Date")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "analysis_weights_area.png"))
    plt.close()


def plot_turnover(turnover_series, save_dir):
    """
    画出换手率变化图
    """
    plt.figure(figsize=(10, 5))
    turnover_series.plot(kind="bar", color="orange", alpha=0.7)
    plt.title(f"Monthly Turnover (Avg: {turnover_series.mean():.4f})")
    plt.ylabel("Turnover Rate")

    # 如果x轴太密，简化显示
    if len(turnover_series) > 20:
        plt.xticks(
            ticks=np.arange(0, len(turnover_series), len(turnover_series) // 10),
            labels=turnover_series.index[:: len(turnover_series) // 10].strftime(
                "%Y-%m"
            ),
            rotation=45,
        )

    plt.grid(axis="y", linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "analysis_turnover.png"))
    plt.close()


def extract_linear_importance(model, feature_names, num_assets):
    """
    从 LinearInferencer 中提取特征重要度

    原理：
    Linear层权重形状为 (num_assets, num_assets * input_dim)
    我们想知道第 k 个特征 (feature_k) 对所有资产预测的平均贡献度。
    """
    # 获取权重矩阵 (N_out, N_in) -> (Num_Assets, Num_Assets * Feats)
    W = model.linear.weight.detach().cpu().numpy()

    # 也就是有 num_assets 个特征，重复了 num_assets 次
    input_dim = len(feature_names)

    # 容器：存储每个特征的“平均绝对权重值”
    importance_dict = {name: 0.0 for name in feature_names}

    # 遍历每个特征 k
    for k, feat_name in enumerate(feature_names):
        # 找到矩阵中所有对应这个特征的列
        # 索引是 k, k + input_dim, k + 2*input_dim, ...
        col_indices = [k + j * input_dim for j in range(num_assets)]

        # 提取这些列的权重
        w_cols = W[:, col_indices]

        # 计算平均绝对值 (Mean Absolute Weight) 作为重要度
        importance_dict[feat_name] = np.mean(np.abs(w_cols))

    return importance_dict


def plot_feature_importance_heatmap(importance_history, save_dir):
    """
    绘制特征重要度随时间变化的 Heatmap
    importance_history: List of dicts [{'Date':..., 'Feat1':...}, ...]
    """
    df = pd.DataFrame(importance_history).set_index("Date")

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.T, cmap="viridis", annot=False, fmt=".2f")
    plt.title("Feature Importance Evolution (Linear Weights)")
    plt.xlabel("Rebalance Date")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "analysis_feature_importance.png"))
    plt.close()

    # 保存 CSV 方便查看
    df.to_csv(os.path.join(save_dir, "analysis_feature_importance.csv"))


def calculate_performance_metrics(df_results, save_dir=None):
    """
    计算策略的核心评价指标：
    1. 年化收益率 (Annualized Return)
    2. 年化波动率 (Annualized Volatility)
    3. 夏普比率 (Sharpe Ratio)
    4. 最大回撤 (Max Drawdown)

    参数:
    - df_results: 必须包含 'daily_return' (log return) 和 'nav' 列
    """
    # 1. 准备数据
    # log returns 用于计算累积收益方便，但在计算波动率时通常也可以直接用
    log_rets = df_results["daily_return"]
    nav = df_results["nav"]

    # 假设一年 252 个交易日
    TRADING_DAYS = 252

    # 2. 计算指标
    # A. 年化收益率 (基于复利)
    # 总收益 = nav_end / nav_start - 1
    # 年化 = (1 + 总收益)^(252/N) - 1  或者直接用 log_return 均值 * 252
    mean_daily_ret = log_rets.mean()
    annualized_return = mean_daily_ret * TRADING_DAYS
    # 转换为百分比显示的简单收益率近似值: exp(ann_ret) - 1
    annualized_return_simple = np.exp(annualized_return) - 1

    # B. 年化波动率
    daily_std = log_rets.std()
    annualized_vol = daily_std * np.sqrt(TRADING_DAYS)

    # C. 夏普比率 (假设无风险利率 Rf=0)
    # Sharpe = E[Rp - Rf] / sigma
    sharpe_ratio = (
        (annualized_return_simple) / annualized_vol if annualized_vol != 0 else 0
    )

    # D. 最大回撤 (MDD)
    # 滚动最大值
    rolling_max = nav.cummax()
    # 回撤 = (当前值 - 历史最高) / 历史最高
    drawdown = (nav - rolling_max) / rolling_max
    max_drawdown = drawdown.min()  # 这是一个负数，例如 -0.20

    # 3. 汇总结果
    metrics = {
        "Annualized Return": annualized_return_simple,
        "Annualized Volatility": annualized_vol,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
    }

    # 4. 如果指定了保存路径，保存到 txt
    if save_dir:
        import json

        metrics_path = os.path.join(save_dir, "performance_metrics.txt")
        with open(metrics_path, "w") as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n")

    return metrics
