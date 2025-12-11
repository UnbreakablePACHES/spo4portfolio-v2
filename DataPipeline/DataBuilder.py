import pandas as pd

def build_dataset(tickers, data_dir="data/FeatureData", dropna=True,
                  start_date=None, end_date=None):
    """
    构建用于模型训练的数据集

    参数:
    - tickers: List[str],ETF 名称列表
    - data_dir: str,CSV 文件夹路径
    - dropna: 是否去掉缺失值行
    - start_date: str,起始日期(如 "2015-01-01")
    - end_date: str,结束日期(如 "2022-12-31")

    返回:
    - merged_feature: (T, N*F)，特征
    - merged_label: (T, N)，标签
    """
    feature_dfs = []
    label_dfs = []

    for ticker in tickers:
        path = f"{data_dir}/{ticker}.csv"
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")

        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        df.drop(columns=["Close"], inplace=True, errors="ignore")

        # 按日期筛选
        if start_date is not None:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df.index <= pd.to_datetime(end_date)]

        df["log_return_input"] = df["log_return"].shift(1)

        feature_cols = [
            "log_return_input", "SMA_10", "price_bias", "RSI_14",
            "MACD_diff", "bollinger_width", "volume_bias"
        ]
        df_feature = df[feature_cols].copy()
        df_feature.columns = [f"{ticker}_{col}" for col in df_feature.columns]
        feature_dfs.append(df_feature)

        df_label = df[["log_return"]].rename(columns={"log_return": ticker})
        label_dfs.append(df_label)

    merged_feature = pd.concat(feature_dfs, axis=1, join="inner")
    merged_label = pd.concat(label_dfs, axis=1, join="inner")

    if dropna:
        merged_feature.dropna(inplace=True)
        merged_label = merged_label.loc[merged_feature.index]

    return merged_feature, merged_label


