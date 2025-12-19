import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import os
import csv


def plot_curve(csv_path, save_path, title="Loss Curve"):
    """
    绘制训练过程中的 Loss 或 Regret 曲线 (原有的函数)
    """
    epochs = []
    values = []

    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            epochs.append(int(row[0]))
            values.append(float(row[1]))

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, values, marker="o")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


def plot_backtest_results(df_results, save_dir):
    """
    绘制回测结果图：
    1. 累计净值曲线 (Line Chart)
    2. 月度收益分布 (Bar Chart)
    """
    # 设置样式
    plt.style.use("ggplot")

    # 1. 准备数据
    nav = df_results["nav"]

    # 计算月度收益
    # 注意：这里假设 df_results["daily_return"] 是对数收益率
    monthly_log_ret = df_results["daily_return"].resample("M").sum()
    monthly_simple_ret = np.exp(monthly_log_ret) - 1

    # 2. 创建画布 (上下两张子图)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [2, 1]}
    )

    # --- 子图1: 累计净值 ---
    ax1.plot(nav.index, nav.values, color="tab:blue", linewidth=2, label="Strategy NAV")
    ax1.set_title("Cumulative Return (NAV)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Net Asset Value", fontsize=12)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # --- 子图2: 月度收益 ---
    colors = ["tab:red" if x < 0 else "tab:green" for x in monthly_simple_ret.values]

    # 绘制柱状图
    ax2.bar(
        monthly_simple_ret.index,
        monthly_simple_ret.values,
        width=20,
        color=colors,
        alpha=0.8,
        label="Monthly Return",
    )

    # ==========================================
    # 【修改重点】X轴标签防重叠处理
    # ==========================================
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # 动态判断数据长度，决定标签间隔
    num_months = len(monthly_simple_ret)
    if num_months <= 12:
        interval = 1  # 1年以内：逐月显示
    elif num_months <= 36:
        interval = 3  # 3年以内：每季度显示
    elif num_months <= 60:
        interval = 6  # 5年以内：每半年显示
    else:
        interval = 12  # 超长回测：每年显示

    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))

    # 旋转45度，并略微缩小字体
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=10)
    # ==========================================

    ax2.set_title("Monthly Returns", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Return", fontsize=12)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # 3. 保存图片
    plot_path = os.path.join(save_dir, "backtest_report.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Backtest plots saved to: {plot_path}")
