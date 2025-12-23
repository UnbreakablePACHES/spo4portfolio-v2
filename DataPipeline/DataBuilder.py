import pandas as pd


def build_dataset(
    tickers,
    data_dir="data/FeatureData",
    dropna=True,
    start_date=None,
    end_date=None,
    feature_list=None,
    label_window=1,
):
    """Build a feature and label dataset for the provided tickers.

    Args:
        tickers: Iterable of asset tickers to load.
        data_dir: Directory containing per-ticker CSV feature files.
        dropna: Whether to drop rows with missing values after alignment.
        start_date: Inclusive start date filter (``YYYY-MM-DD``).
        end_date: Inclusive end date filter (``YYYY-MM-DD``).
        feature_list: Specific feature column names to keep; defaults to a
            baseline list when not provided.
        label_window: Number of future periods to aggregate for the label
            target; ``1`` uses the next-period log return.

    Returns:
        Tuple of ``(features_df, labels_df)`` aligned on dates.
    """
    feature_dfs = []
    label_dfs = []

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

        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        df.drop(columns=["Close"], inplace=True, errors="ignore")
        if start_date is not None:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df.index <= pd.to_datetime(end_date)]
        if "log_return" in df.columns:
            df["log_return_input"] = df["log_return"].shift(1)
        available_cols = [c for c in feature_list if c in df.columns]
        if len(available_cols) != len(feature_list):
            missing = set(feature_list) - set(available_cols)
            raise ValueError(f"Ticker {ticker} missing features: {missing}")

        df_feature = df[available_cols].copy()
        df_feature.columns = [f"{ticker}_{col}" for col in df_feature.columns]
        feature_dfs.append(df_feature)

        if label_window == 1:
            df_label = df[["log_return"]].rename(columns={"log_return": ticker})
        else:
            indexer = pd.api.indexers.FixedForwardWindowIndexer(
                window_size=label_window
            )
            rolling_sum = (
                df["log_return"]
                .rolling(window=label_window)
                .sum()
                .shift(-(label_window - 1))
            )

            df_label = rolling_sum.to_frame(name=ticker)
        label_dfs.append(df_label)

    merged_feature = pd.concat(feature_dfs, axis=1, join="inner")
    merged_label = pd.concat(label_dfs, axis=1, join="inner")

    if dropna:
        full_df = pd.concat([merged_feature, merged_label], axis=1)
        full_df.dropna(inplace=True)

        feat_cols = merged_feature.columns
        label_cols = merged_label.columns

        merged_feature = full_df[feat_cols]
        merged_label = full_df[label_cols]

    return merged_feature, merged_label
