"""Technical indicator helpers (history-only)."""

from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, length: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(length, min_periods=length).mean()


def bollinger_bands(
    close: pd.Series, length: int = 20, sigma: float = 2.0
) -> pd.DataFrame:
    """Bollinger bands from rolling mean/std."""
    basis = close.rolling(length, min_periods=length).mean()
    dev = close.rolling(length, min_periods=length).std(ddof=0)
    upper = basis + sigma * dev
    lower = basis - sigma * dev
    return pd.DataFrame({"bb_basis": basis, "bb_upper": upper, "bb_lower": lower})


def envelopes(close: pd.Series, length: int = 96, pct: float = 1.0) -> pd.DataFrame:
    """Price envelopes around SMA."""
    mid = close.rolling(length, min_periods=length).mean()
    ratio = pct / 100.0
    upper = mid * (1.0 + ratio)
    lower = mid * (1.0 - ratio)
    return pd.DataFrame({"env_mid": mid, "env_upper": upper, "env_lower": lower})


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder-style EMA approximation)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def mfi(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 14
) -> pd.Series:
    """Money Flow Index."""
    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * volume
    tp_delta = typical_price.diff()
    positive_flow = raw_money_flow.where(tp_delta > 0.0, 0.0)
    negative_flow = raw_money_flow.where(tp_delta < 0.0, 0.0).abs()

    pos_sum = positive_flow.rolling(length, min_periods=length).sum()
    neg_sum = negative_flow.rolling(length, min_periods=length).sum()
    money_ratio = pos_sum / neg_sum
    return 100.0 - (100.0 / (1.0 + money_ratio))

