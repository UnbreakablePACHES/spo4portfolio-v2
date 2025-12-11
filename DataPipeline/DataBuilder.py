import pandas as pd

def build_dataset(tickers, data_dir="data/FeatureData", dropna=True,
                  start_date=None, end_date=None, 
                  feature_list=None): # <--- 确保这里有 feature_list 参数
    """
    构建用于模型训练的数据集
    
    参数:
    - tickers: List[str], ETF 名称列表
    - data_dir: str, CSV 文件夹路径
    - dropna: 是否去掉缺失值行
    - start_date: str, 起始日期 (e.g. "2023-01-01")
    - end_date: str, 结束日期
    - feature_list: List[str], 需要使用的特征列名列表 (从 Config 传入)
    """
    feature_dfs = []
    label_dfs = []

    # 如果没传特征列表，给一个默认值防止报错
    if feature_list is None:
        feature_list = [
            "log_return_input", "SMA_10", "price_bias", "RSI_14",
            "MACD_diff", "bollinger_width", "volume_bias"
        ]

    for ticker in tickers:
        path = f"{data_dir}/{ticker}.csv"
        # 1. 读取数据
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")

        # 2. 基础清洗
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        df.drop(columns=["Close"], inplace=True, errors="ignore")

        # ==================================================
        # 【重点】在这里写日期筛选代码
        # 必须写在 for 循环里，读完 csv 之后
        # ==================================================
        if start_date is not None:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df.index <= pd.to_datetime(end_date)]

        # 3. 特征工程 (Shift等)
        # 注意：如果 start_date 卡得太死，Shift 产生的 NaN 会导致第一天数据被 dropna 删掉
        # 这通常是可以接受的，或者你可以让 start_date 稍微提前一点点
        if "log_return" in df.columns:
            df["log_return_input"] = df["log_return"].shift(1)

        # 4. 根据 feature_list 筛选特征列
        # 先判断 CSV 里有没有这些列
        available_cols = [c for c in feature_list if c in df.columns]
        
        # 校验：如果有特征缺失，报错提示
        if len(available_cols) != len(feature_list):
            missing = set(feature_list) - set(available_cols)
            raise ValueError(f"Ticker {ticker} missing features: {missing}")

        df_feature = df[available_cols].copy()
        df_feature.columns = [f"{ticker}_{col}" for col in df_feature.columns]
        feature_dfs.append(df_feature)

        # 5. 提取标签 (Label)
        df_label = df[["log_return"]].rename(columns={"log_return": ticker})
        label_dfs.append(df_label)

    # 合并所有资产的数据
    merged_feature = pd.concat(feature_dfs, axis=1, join="inner")
    merged_label = pd.concat(label_dfs, axis=1, join="inner")

    if dropna:
        merged_feature.dropna(inplace=True)
        merged_label = merged_label.loc[merged_feature.index]

    return merged_feature, merged_label

