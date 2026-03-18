"""Timeframe aggregation and leak-safe window helpers."""

from __future__ import annotations

from typing import Dict

import pandas as pd

_TIMEFRAME_RULES: Dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "1d": "1d",
}

_REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}
_OPTIONAL_SUM_COLUMNS = (
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base",
    "taker_buy_quote",
)


def _normalize_timeframe(timeframe: str) -> str:
    key = timeframe.strip().lower()
    return _TIMEFRAME_RULES.get(key, timeframe)


def aggregate_ohlcv(
    df_1m: pd.DataFrame,
    timeframe: str = "5m",
    drop_incomplete_last_candle: bool = True,
) -> pd.DataFrame:
    """
    Aggregate OHLCV data with left-closed/right-open bins.

    Notes:
        - Uses ``closed='left'`` and ``label='left'``.
        - By default, drops the last aggregated candle per ticker to avoid
          potentially incomplete candles (future-leak prevention policy).
    """
    missing = _REQUIRED_COLUMNS.difference(df_1m.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rule = _normalize_timeframe(timeframe)

    working = df_1m.copy()
    working["date"] = pd.to_datetime(working["date"])
    if "tic" not in working.columns:
        working["tic"] = "UNKNOWN"

    agg_map = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    for col in _OPTIONAL_SUM_COLUMNS:
        if col in working.columns:
            agg_map[col] = "sum"

    aggregated = (
        working.sort_values(["tic", "date"])
        .groupby("tic")
        .resample(rule=rule, on="date", closed="left", label="left")
        .agg(agg_map)
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
        .sort_values(["tic", "date"])
        .reset_index(drop=True)
    )

    if drop_incomplete_last_candle and not aggregated.empty:
        order = aggregated.groupby("tic").cumcount()
        sizes = aggregated.groupby("tic")["date"].transform("size")
        aggregated = aggregated.loc[order < (sizes - 1)].reset_index(drop=True)

    if aggregated.empty:
        return aggregated

    aggregated["day"] = aggregated["date"].dt.dayofweek
    ordered_columns = (
        ["date", "open", "high", "low", "close", "volume"]
        + [c for c in _OPTIONAL_SUM_COLUMNS if c in aggregated.columns]
        + ["tic", "day"]
    )
    return aggregated[ordered_columns]


def make_window(data: pd.DataFrame, t: int, w: int = 96) -> pd.DataFrame:
    """
    Return a leak-safe historical window `[t-w+1, t]`.

    Args:
        data: Time-ordered dataframe.
        t: Current row index.
        w: Window size.
    """
    if w <= 0:
        raise ValueError("w must be > 0")
    if t >= len(data):
        raise ValueError(f"t={t} is out of range for len(data)={len(data)}")
    if t < w - 1:
        raise ValueError(f"insufficient history: t={t}, w={w}")

    window = data.iloc[t - w + 1 : t + 1]
    if len(window) != w:
        raise ValueError(f"invalid window size: expected {w}, got {len(window)}")
    return window

