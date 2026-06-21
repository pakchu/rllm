"""Leak-safe market feature engineering shared across RL/VLM pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from preprocessing.indicators import mfi, rsi


CORE_MARKET_FEATURE_COLUMNS = (
    "range_vol",
    "trend_12",
    "trend_24",
    "trend_96",
    "sma12_ratio",
    "sma24_ratio",
    "sma48_ratio",
    "rsi_norm",
    "mfi_norm",
    "bb_z",
    "range_pos",
)


EXTENDED_MARKET_FEATURE_COLUMNS = CORE_MARKET_FEATURE_COLUMNS + (
    "close_zscore_48",
    "return_zscore_48",
    "body_ratio",
    "upper_shadow",
    "lower_shadow",
    "candle_range",
    "body_to_range",
    "shadow_imbalance",
    "volume_ratio",
    "volume_zscore",
    "window_drawdown",
    "trades_ratio",
    "taker_buy_ratio",
    "taker_imbalance",
    "funding_rate",
    "funding_zscore",
    "oi_change",
    "oi_zscore",
    "dxy",
    "dxy_zscore",
    "dxy_momentum",
    "kimchi_premium",
    "kimchi_premium_zscore",
    "kimchi_premium_change",
    "usdkrw_zscore",
    "usdkrw_momentum",
    "dxy_available",
    "kimchi_available",
    "usdkrw_available",
    "external_any_available",
    "weekly_return_1w",
    "weekly_return_4w",
    "weekly_range_1w",
    "weekly_range_pos",
    "weekly_drawdown_4w",
    "htf_4h_return_1",
    "htf_4h_return_4",
    "htf_4h_range_1",
    "htf_4h_range_pos",
    "htf_4h_drawdown_4",
    "htf_1d_return_1",
    "htf_1d_return_4",
    "htf_1d_range_1",
    "htf_1d_range_pos",
    "htf_1d_drawdown_4",
    "htf_3d_return_1",
    "htf_3d_return_4",
    "htf_3d_range_1",
    "htf_3d_range_pos",
    "htf_3d_drawdown_4",
    "htf_1w_return_1",
    "htf_1w_return_4",
    "htf_1w_range_1",
    "htf_1w_range_pos",
    "htf_1w_drawdown_4",
)


def _clean_series(series: pd.Series, *, clip: float | None = None) -> pd.Series:
    out = series.replace([np.inf, -np.inf], np.nan)
    if clip is not None:
        out = out.clip(-float(clip), float(clip))
    return out.fillna(0.0)


def _ratio_to_mean(series: pd.Series, window: int) -> pd.Series:
    avg = series.rolling(window, min_periods=1).mean()
    out = (series - avg) / avg.replace(0.0, np.nan)
    return _clean_series(out)


def _return_over(close: pd.Series, periods: int) -> pd.Series:
    ref = close.shift(max(1, int(periods)))
    out = (close - ref) / ref.replace(0.0, np.nan)
    return _clean_series(out)


def _rolling_zscore(series: pd.Series, window: int, *, clip: float = 5.0) -> pd.Series:
    window = max(1, int(window))
    min_periods = min(window, max(2, window // 3))
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std(ddof=0)
    out = (series - mean) / std.replace(0.0, np.nan)
    return _clean_series(out, clip=clip)


def _optional_column(df: pd.DataFrame, name: str) -> pd.Series | None:
    if name not in df.columns:
        return None
    return df[name].astype(float)


def _completed_timeframe_features(
    market_df: pd.DataFrame,
    *,
    prefix: str,
    resample_rule: str,
    min_source_rows: int,
) -> dict[str, pd.Series]:
    """Previous completed higher-timeframe features aligned to each row.

    The higher-timeframe candle that contains the current row is deliberately
    excluded.  We resample, compute features, shift them by one completed bar,
    then backward as-of join to every market timestamp.  This makes row ``t``
    depend only on higher-timeframe candles completed before ``t``.
    """
    defaults = {
        f"{prefix}_return_1": pd.Series(0.0, index=market_df.index),
        f"{prefix}_return_4": pd.Series(0.0, index=market_df.index),
        f"{prefix}_range_1": pd.Series(0.0, index=market_df.index),
        f"{prefix}_range_pos": pd.Series(0.0, index=market_df.index),
        f"{prefix}_drawdown_4": pd.Series(0.0, index=market_df.index),
    }
    if "date" not in market_df.columns or len(market_df) < int(min_source_rows):
        return defaults

    source = market_df[["date", "open", "high", "low", "close"]].copy()
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    source = source.dropna(subset=["date"]).sort_values("date")
    if source.empty:
        return defaults
    source = source.set_index("date")

    htf = pd.DataFrame(
        {
            "open": source["open"].resample(resample_rule, label="right", closed="right").first(),
            "high": source["high"].resample(resample_rule, label="right", closed="right").max(),
            "low": source["low"].resample(resample_rule, label="right", closed="right").min(),
            "close": source["close"].resample(resample_rule, label="right", closed="right").last(),
        }
    ).dropna()
    if len(htf) < 2:
        return defaults

    prev = htf.shift(1)
    htf_range = (prev["high"] - prev["low"]).replace(0.0, np.nan)
    features = pd.DataFrame(index=htf.index)
    features[f"{prefix}_return_1"] = _clean_series(prev["close"] / prev["open"].replace(0.0, np.nan) - 1.0)
    features[f"{prefix}_return_4"] = _clean_series(
        prev["close"] / prev["close"].shift(4).replace(0.0, np.nan) - 1.0
    )
    features[f"{prefix}_range_1"] = _clean_series(htf_range / prev["close"].replace(0.0, np.nan))
    features[f"{prefix}_range_pos"] = _clean_series(((prev["close"] - prev["low"]) / htf_range) * 2.0 - 1.0)
    htf_peak_4 = prev["close"].rolling(4, min_periods=1).max()
    features[f"{prefix}_drawdown_4"] = _clean_series(1.0 - prev["close"] / htf_peak_4.replace(0.0, np.nan))
    features = features.replace([np.inf, -np.inf], 0.0).fillna(0.0).reset_index(names="date")

    target_dates = pd.DataFrame({"date": pd.to_datetime(market_df["date"], errors="coerce")})
    target_dates["_row"] = np.arange(len(target_dates))
    aligned = pd.merge_asof(
        target_dates.sort_values("date"),
        features.sort_values("date"),
        on="date",
        direction="backward",
    ).sort_values("_row")
    out: dict[str, pd.Series] = {}
    for col in defaults:
        out[col] = pd.Series(aligned[col].fillna(0.0).to_numpy(), index=market_df.index)
    return out


def _completed_multitimeframe_features(market_df: pd.DataFrame) -> dict[str, pd.Series]:
    """Leak-safe completed higher-timeframe features for 4h/1d/3d/1w."""
    specs = (
        ("htf_4h", "4h", 4 * 60 * 4),
        ("htf_1d", "1D", 24 * 60 * 4),
        ("htf_3d", "3D", 3 * 24 * 60 * 4),
        ("htf_1w", "W-SUN", 7 * 24 * 60),
    )
    out: dict[str, pd.Series] = {}
    for prefix, rule, min_rows in specs:
        out.update(
            _completed_timeframe_features(
                market_df,
                prefix=prefix,
                resample_rule=rule,
                min_source_rows=min_rows,
            )
        )
    # Backward-compatible weekly aliases used by edge_state_v6.
    alias_pairs = {
        "weekly_return_1w": "htf_1w_return_1",
        "weekly_return_4w": "htf_1w_return_4",
        "weekly_range_1w": "htf_1w_range_1",
        "weekly_range_pos": "htf_1w_range_pos",
        "weekly_drawdown_4w": "htf_1w_drawdown_4",
    }
    for alias, src in alias_pairs.items():
        out[alias] = out.get(src, pd.Series(0.0, index=market_df.index))
    return out


def build_market_feature_frame(
    market_df: pd.DataFrame,
    *,
    window_size: int = 96,
    zscore_window: int = 48,
    volume_window: int = 48,
) -> pd.DataFrame:
    """
    Build history-only engineered market features.

    All features at row ``t`` depend only on rows ``<= t``.
    """
    if market_df.empty:
        return pd.DataFrame(index=market_df.index)

    close = market_df["close"].astype(float)
    open_ = market_df["open"].astype(float)
    high = market_df["high"].astype(float)
    low = market_df["low"].astype(float)
    volume = market_df["volume"].astype(float)

    roll_high = high.rolling(window_size, min_periods=window_size).max()
    roll_low = low.rolling(window_size, min_periods=window_size).min()
    range_mid = (roll_high + roll_low) / 2.0
    range_span = (roll_high - roll_low).replace(0.0, np.nan)

    feature_map: dict[str, pd.Series] = {}
    feature_map["range_vol"] = _clean_series((roll_high - roll_low) / range_mid.replace(0.0, np.nan))
    feature_map["trend_12"] = _return_over(close, 11)
    feature_map["trend_24"] = _return_over(close, 23)
    feature_map["trend_96"] = _return_over(close, max(window_size - 1, 1))
    feature_map["sma12_ratio"] = _ratio_to_mean(close, 12)
    feature_map["sma24_ratio"] = _ratio_to_mean(close, 24)
    feature_map["sma48_ratio"] = _ratio_to_mean(close, 48)
    feature_map["rsi_norm"] = _clean_series((rsi(close, length=14) - 50.0) / 50.0)
    feature_map["mfi_norm"] = _clean_series((mfi(high, low, close, volume, length=14) - 50.0) / 50.0)

    bb_mean = close.rolling(20, min_periods=1).mean()
    bb_std = close.rolling(20, min_periods=1).std(ddof=0)
    feature_map["bb_z"] = _clean_series((close - bb_mean) / bb_std.replace(0.0, np.nan))
    feature_map["range_pos"] = _clean_series(((close - roll_low) / range_span) * 2.0 - 1.0)

    log_return = np.log(close / close.shift(1).replace(0.0, np.nan))
    feature_map["close_zscore_48"] = _rolling_zscore(close, zscore_window)
    feature_map["return_zscore_48"] = _rolling_zscore(log_return, zscore_window)

    upper_body = np.maximum(open_, close)
    lower_body = np.minimum(open_, close)
    candle_range = (high - low).replace(0.0, np.nan)
    feature_map["body_ratio"] = _clean_series((close - open_) / close.replace(0.0, np.nan))
    feature_map["upper_shadow"] = _clean_series((high - upper_body) / close.replace(0.0, np.nan))
    feature_map["lower_shadow"] = _clean_series((lower_body - low) / close.replace(0.0, np.nan))
    feature_map["candle_range"] = _clean_series((high - low) / close.replace(0.0, np.nan))
    feature_map["body_to_range"] = _clean_series((close - open_).abs() / candle_range)
    feature_map["shadow_imbalance"] = _clean_series(
        feature_map["lower_shadow"] - feature_map["upper_shadow"]
    )

    vol_mean = volume.rolling(volume_window, min_periods=max(5, volume_window // 3)).mean()
    vol_std = volume.rolling(volume_window, min_periods=max(5, volume_window // 3)).std(ddof=0)
    feature_map["volume_ratio"] = _clean_series(np.log1p(volume / vol_mean.replace(0.0, np.nan)))
    feature_map["volume_zscore"] = _clean_series(
        (volume - vol_mean) / vol_std.replace(0.0, np.nan), clip=5.0
    )

    rolling_peak = close.rolling(window_size, min_periods=1).max()
    feature_map["window_drawdown"] = _clean_series(
        1.0 - close / rolling_peak.replace(0.0, np.nan)
    )

    number_of_trades = _optional_column(market_df, "number_of_trades")
    if number_of_trades is not None:
        trades_mean = number_of_trades.rolling(volume_window, min_periods=max(5, volume_window // 3)).mean()
        feature_map["trades_ratio"] = _clean_series(
            np.log1p(number_of_trades / trades_mean.replace(0.0, np.nan))
        )
    else:
        feature_map["trades_ratio"] = pd.Series(0.0, index=market_df.index)

    taker_buy_base = _optional_column(market_df, "taker_buy_base")
    if taker_buy_base is not None:
        taker_buy_ratio = (taker_buy_base / volume.replace(0.0, np.nan)).fillna(0.5)
        feature_map["taker_buy_ratio"] = _clean_series(taker_buy_ratio)
        feature_map["taker_imbalance"] = _clean_series(taker_buy_ratio * 2.0 - 1.0)
    else:
        feature_map["taker_buy_ratio"] = pd.Series(0.5, index=market_df.index)
        feature_map["taker_imbalance"] = pd.Series(0.0, index=market_df.index)

    funding_rate = _optional_column(market_df, "funding_rate")
    if funding_rate is not None:
        feature_map["funding_rate"] = _clean_series(funding_rate, clip=1.0)
        feature_map["funding_zscore"] = _rolling_zscore(funding_rate, volume_window)
    else:
        feature_map["funding_rate"] = pd.Series(0.0, index=market_df.index)
        feature_map["funding_zscore"] = pd.Series(0.0, index=market_df.index)

    open_interest = _optional_column(market_df, "open_interest")
    if open_interest is not None:
        feature_map["oi_change"] = _clean_series(
            np.log(open_interest / open_interest.shift(1).replace(0.0, np.nan))
        )
        feature_map["oi_zscore"] = _rolling_zscore(open_interest, volume_window)
    else:
        feature_map["oi_change"] = pd.Series(0.0, index=market_df.index)
        feature_map["oi_zscore"] = pd.Series(0.0, index=market_df.index)

    optional_external_defaults = {
        "dxy": 0.0,
        "dxy_zscore": 0.0,
        "dxy_momentum": 0.0,
        "kimchi_premium": 0.0,
        "kimchi_premium_zscore": 0.0,
        "kimchi_premium_change": 0.0,
        "usdkrw_zscore": 0.0,
        "usdkrw_momentum": 0.0,
        "dxy_available": 0.0,
        "kimchi_available": 0.0,
        "usdkrw_available": 0.0,
        "external_any_available": 0.0,
    }
    for col, default in optional_external_defaults.items():
        series = _optional_column(market_df, col)
        feature_map[col] = (
            _clean_series(series, clip=5.0)
            if series is not None
            else pd.Series(float(default), index=market_df.index)
        )

    feature_map.update(_completed_multitimeframe_features(market_df))

    frame = pd.DataFrame(feature_map, index=market_df.index)
    return frame.replace([np.inf, -np.inf], 0.0).fillna(0.0)
