"""Scalar feature extraction helpers."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from preprocessing.timeframe import make_window
from utils import range_volatility_pct


def compute_range_volatility_pct(window_df: pd.DataFrame) -> float:
    """Compute window range volatility from OHLC window."""
    return float(
        range_volatility_pct(highs=window_df["high"].to_numpy(), lows=window_df["low"].to_numpy())
    )


def extract_scalars(
    window_df: pd.DataFrame, position_size_pct: float, last_entry_price: float
) -> Dict[str, float]:
    """Build scalar observation dict from current window + portfolio state."""
    return {
        "position_size_pct": float(position_size_pct),
        "last_entry_price": float(last_entry_price),
        "range_volatility_pct": compute_range_volatility_pct(window_df),
    }


def build_scalar_frame(
    market_df: pd.DataFrame,
    window_size: int = 96,
    position_size_pct: float = 0.0,
    last_entry_price: float = 0.0,
) -> pd.DataFrame:
    """
    Build scalar features per timestep using only historical window `[t-w+1, t]`.
    """
    rows = []
    for t in range(window_size - 1, len(market_df)):
        window = make_window(market_df, t=t, w=window_size)
        scalars = extract_scalars(window, position_size_pct, last_entry_price)
        rows.append({"date": market_df.loc[t, "date"], **scalars})
    return pd.DataFrame(rows)
