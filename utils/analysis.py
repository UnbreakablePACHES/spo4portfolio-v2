import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def calculate_turnover(weights_df):
    """Compute single-sided turnover from weight history.

    Args:
        weights_df: DataFrame of weights indexed by date.

    Returns:
        Series of turnover values per period.
    """
    diff = weights_df.diff().abs().sum(axis=1).fillna(0.0)
    turnover = diff / 2.0
    return turnover


def plot_weights_area(weights_df, save_dir):
    """Plot a stacked area chart of portfolio weights over time."""
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
    """Visualize turnover as a bar chart."""
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
    """Extract mean absolute weight importance from a linear model.

    Args:
        model: Trained ``LinearInferencer`` instance.
        feature_names: List of feature names in per-asset order.
        num_assets: Number of assets represented in the weight matrix.

    Returns:
        Dict mapping feature name to average absolute weight magnitude.
    """
    W = model.linear.weight.detach().cpu().numpy()
    input_dim = len(feature_names)

    importance_dict = {name: 0.0 for name in feature_names}

    for k, feat_name in enumerate(feature_names):
        col_indices = [k + j * input_dim for j in range(num_assets)]
        w_cols = W[:, col_indices]
        importance_dict[feat_name] = np.mean(np.abs(w_cols))

    return importance_dict


def plot_feature_importance_heatmap(importance_history, save_dir):
    """Plot heatmap of feature importance over time."""
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
    """Calculate annualized performance metrics from backtest results.

    Args:
        df_results: DataFrame containing ``daily_return`` (log return) and
            ``nav`` columns.
        save_dir: Optional directory to persist metrics to disk.

    Returns:
        Dict with annualized return, volatility, Sharpe ratio, and max drawdown.
    """
    log_rets = df_results["daily_return"]
    nav = df_results["nav"]

    TRADING_DAYS = 252

    mean_daily_ret = log_rets.mean()
    annualized_return = mean_daily_ret * TRADING_DAYS
    annualized_return_simple = np.exp(annualized_return) - 1

    daily_std = log_rets.std()
    annualized_vol = daily_std * np.sqrt(TRADING_DAYS)

    sharpe_ratio = (
        (annualized_return_simple) / annualized_vol if annualized_vol != 0 else 0
    )

    rolling_max = nav.cummax()
    drawdown = (nav - rolling_max) / rolling_max
    max_drawdown = drawdown.min()  # 这是一个负数，例如 -0.20

    metrics = {
        "Annualized Return": annualized_return_simple,
        "Annualized Volatility": annualized_vol,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
    }

    if save_dir:
        import json

        metrics_path = os.path.join(save_dir, "performance_metrics.txt")
        with open(metrics_path, "w") as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n")

    return metrics
