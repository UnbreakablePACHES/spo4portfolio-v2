import pandas as pd


def build_dataset(
    tickers,
    data_dir="data/FeatureData",
    dropna=True,
    start_date=None,
    end_date=None,
    feature_list=None,
    label_window=1,
    oracle_weights_df=None,  # <--- 传入 Oracle 权重
):
    feature_dfs = []
    label_dfs = []
    prev_weight_dfs = []  # <--- 1. 初始化列表

    if feature_list is None:
        feature_list = [
            "log_return_input",
            "SMA_10",
            "price_bias",
            "RSI_14",
            "MACD_diff",
            "bollinger_width",
            "volume_bias",
        ]

    for ticker in tickers:
        path = f"{data_dir}/{ticker}.csv"
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")

        # --- 基础清洗 ---
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        df.drop(columns=["Close"], inplace=True, errors="ignore")

        # --- 日期筛选 ---
        if start_date is not None:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df.index <= pd.to_datetime(end_date)]

        # --- 特征工程 ---
        if "log_return" in df.columns:
            df["log_return_input"] = df["log_return"].shift(1)

        # --- 特征列筛选 ---
        available_cols = [c for c in feature_list if c in df.columns]
        if len(available_cols) != len(feature_list):
            missing = set(feature_list) - set(available_cols)
            raise ValueError(f"Ticker {ticker} missing features: {missing}")

        df_feature = df[available_cols].copy()
        df_feature.columns = [f"{ticker}_{col}" for col in df_feature.columns]
        feature_dfs.append(df_feature)

        # --- Label 生成 ---
        if label_window == 1:
            df_label = df[["log_return"]].rename(columns={"log_return": ticker})
        else:
            rolling_sum = (
                df["log_return"]
                .rolling(window=label_window)
                .sum()
                .shift(-(label_window - 1))
            )
            df_label = rolling_sum.to_frame(name=ticker)
        label_dfs.append(df_label)

        # --- 2. 核心修正：Oracle Prev Weights 处理 ---
        if oracle_weights_df is not None:
            # 如果提供了 oracle 文件，提取该 ticker 的权重
            if ticker in oracle_weights_df.columns:
                # 必须 reindex 确保日期对齐，缺失填 0
                s_oracle = oracle_weights_df[ticker].reindex(df.index).fillna(0.0)
                # Shift(1): 今天的 "prev_weight" 是昨天的 "oracle_weight"
                s_prev = s_oracle.shift(1)
                df_prev = s_prev.to_frame(name=ticker)
            else:
                # 如果文件中没有这个 ticker，给全 0
                df_prev = pd.DataFrame(0.0, index=df.index, columns=[ticker])

            prev_weight_dfs.append(df_prev)

    # --- 合并 ---
    merged_feature = pd.concat(feature_dfs, axis=1, join="inner")
    merged_label = pd.concat(label_dfs, axis=1, join="inner")

    # 如果没有 oracle_weights_df，直接走简单逻辑返回
    if oracle_weights_df is None:
        # 这里需要单独处理 dropna
        if dropna:
            full_df = pd.concat([merged_feature, merged_label], axis=1)
            full_df.dropna(inplace=True)
            merged_feature = full_df[merged_feature.columns]
            merged_label = full_df[merged_label.columns]
        return merged_feature, merged_label, None

    # --- 3. 核心修正：如果有 prev_weights，必须一起 dropna ---
    merged_prev = pd.concat(prev_weight_dfs, axis=1, join="inner")

    if dropna:
        # 将 Feature, Label, PrevWeights 全部拼在一起清洗
        # 这样 Shift(1) 产生的 NaN 就会被正确删掉，保持行对齐
        full_df = pd.concat([merged_feature, merged_label, merged_prev], axis=1)
        full_df.dropna(inplace=True)

        # 重新切分
        n_feat = merged_feature.shape[1]
        n_lab = merged_label.shape[1]

        merged_feature = full_df.iloc[:, :n_feat]
        merged_label = full_df.iloc[:, n_feat : n_feat + n_lab]
        merged_prev = full_df.iloc[:, n_feat + n_lab :]

    return merged_feature, merged_label, merged_prev
