import pandas as pd

def build_dataset(tickers, data_dir="data/FeatureData", dropna=True,
                  start_date=None, end_date=None, 
                  feature_list=None, label_window=1): # <--- 确保这里有 feature_list 参数
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

        # ==================================================
        # 【核心修改】Label 生成逻辑
        # ==================================================
        if label_window == 1:
            # 原逻辑：预测下一日收益
            df_label = df[["log_return"]].rename(columns={"log_return": ticker})
        else:
            # 新逻辑：预测未来 N 天累计收益
            # 逻辑：当前行 t 的特征是 t-1 的。我们希望预测从 t-1 收盘持有到 t+19 收盘的收益。
            # 即 sum(r_t, r_{t+1}, ..., r_{t+19})。
            # rolling(20).sum() 在 t+19 时刻的值正是这个 sum。
            # 所以我们需要把 t+19 的值 shift 回 t，偏移量是 -(window-1) = -19。
            
            # 计算滚动和 (Rolling Sum)
            indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=label_window)
            # 或者用更简单的写法:
            # shift(-(N-1)) 是因为 rolling 包含当前行
            rolling_sum = df["log_return"].rolling(window=label_window).sum().shift(-(label_window - 1))
            
            df_label = rolling_sum.to_frame(name=ticker)
        label_dfs.append(df_label)

    # 合并所有资产的数据
    merged_feature = pd.concat(feature_dfs, axis=1, join="inner")
    merged_label = pd.concat(label_dfs, axis=1, join="inner")

    if dropna:
        # 1. 先把特征和标签拼在一起，彻底清洗所有 NaN
        #    这样既能删掉特征缺失的行(通常在开头)，
        #    也能删掉标签缺失的行(通常在结尾，因为做了 shift(-N))
        full_df = pd.concat([merged_feature, merged_label], axis=1)
        full_df.dropna(inplace=True)

        # 2. 重新拆分回特征和标签
        #    注意：这里要确保列名切分正确
        feat_cols = merged_feature.columns
        label_cols = merged_label.columns
        
        merged_feature = full_df[feat_cols]
        merged_label = full_df[label_cols]

    return merged_feature, merged_label

    return merged_feature, merged_label

