import pandas as pd

# 把原来的 build_dataset 签名改一下，增加 feature_list 参数
def build_dataset(tickers, data_dir="data/FeatureData", dropna=True,
                  start_date=None, end_date=None, 
                  feature_list=None): # <--- 【新增参数】
    """
    feature_list: List[str], 需要使用的特征列名列表
    """
    feature_dfs = []
    label_dfs = []

    # 如果没传，给一个默认列表（作为保底，或者直接报错强制要求传）
    if feature_list is None:
        feature_list = [
            "log_return_input", "SMA_10", "price_bias", "RSI_14",
            "MACD_diff", "bollinger_width", "volume_bias"
        ]

    for ticker in tickers:
        path = f"{data_dir}/{ticker}.csv"
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        
        # ... (中间的数据清洗代码保持不变) ...
        # df = df.loc[...] 
        # df.drop(...)
        # 按日期筛选...

        df["log_return_input"] = df["log_return"].shift(1)

        # === 【关键修改】 ===
        # 不再硬编码，而是使用传入的 feature_list
        # 确保这些列在 CSV 里都存在
        available_cols = [c for c in feature_list if c in df.columns]
        if len(available_cols) != len(feature_list):
            missing = set(feature_list) - set(available_cols)
            raise ValueError(f"Ticker {ticker} missing features: {missing}")

        df_feature = df[available_cols].copy()
        
        # 加上 Ticker 前缀，保持原本逻辑
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

