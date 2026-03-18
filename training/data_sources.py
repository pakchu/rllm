"""Market data source utilities for training/preprocessing."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _timeframe_to_timedelta(timeframe: str) -> Optional[pd.Timedelta]:
    """Parse timeframe strings like 1m/5m/15m/1h/1d into Timedelta."""
    key = str(timeframe).strip().lower()
    if not key:
        return None
    if key.endswith("m") and key[:-1].isdigit():
        return pd.Timedelta(minutes=int(key[:-1]))
    if key.endswith("h") and key[:-1].isdigit():
        return pd.Timedelta(hours=int(key[:-1]))
    if key.endswith("d") and key[:-1].isdigit():
        return pd.Timedelta(days=int(key[:-1]))
    return None


def _infer_csv_cadence(df: pd.DataFrame) -> Optional[pd.Timedelta]:
    """Infer dominant sampling cadence from date column using positive diffs."""
    if "date" not in df.columns or len(df) < 3:
        return None
    diffs = (
        pd.to_datetime(df["date"])
        .sort_values()
        .diff()
        .dropna()
        .dt.total_seconds()
    )
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return None
    mode = diffs.mode()
    if mode.empty:
        return None
    return pd.Timedelta(seconds=float(mode.sort_values().iloc[0]))


def _validate_csv_timeframe(
    df: pd.DataFrame,
    timeframe: str,
    input_csv: Optional[str],
) -> None:
    """
    Guard against accidental timeframe mismatch when loading CSV.

    Example:
      - requested timeframe='5m'
      - CSV cadence is 15 minutes
      -> raise ValueError to prevent silent evaluation/training skew.
    """
    expected = _timeframe_to_timedelta(timeframe)
    if expected is None:
        return
    inferred = _infer_csv_cadence(df)
    if inferred is None:
        return
    if abs(inferred - expected) > pd.Timedelta(seconds=1):
        src = input_csv or "<csv>"
        raise ValueError(
            "CSV timeframe mismatch: "
            f"requested timeframe='{timeframe}' ({expected}), "
            f"but inferred cadence from '{src}' is {inferred}."
        )


def make_synthetic_market_df(
    num_rows: int = 5_000,
    start_price: float = 30_000.0,
    drift: float = 0.0,
    regime_amplitude: float = 0.0004,
    regime_period: int = 720,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic 1-minute OHLCV data for smoke workflows."""
    rng = np.random.default_rng(seed)
    t = np.arange(num_rows, dtype=np.float64)
    period = max(1, int(regime_period))
    cyclical_drift = float(regime_amplitude) * np.sin(2.0 * np.pi * t / period)
    returns = rng.normal(loc=drift + cyclical_drift, scale=0.001, size=num_rows)
    prices = start_price * np.exp(np.cumsum(returns))

    open_prices = prices
    close_prices = prices * np.exp(rng.normal(0.0, 0.0005, size=num_rows))
    high_prices = np.maximum(open_prices, close_prices) * (
        1.0 + np.abs(rng.normal(0.0008, 0.0002, size=num_rows))
    )
    low_prices = np.minimum(open_prices, close_prices) * (
        1.0 - np.abs(rng.normal(0.0008, 0.0002, size=num_rows))
    )
    volumes = rng.lognormal(mean=2.0, sigma=0.3, size=num_rows)

    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=num_rows, freq="1min"),
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes,
            "tic": "BTCUSDT",
        }
    )


def _normalize_market_df(df: pd.DataFrame, symbol: str = "BTCUSDT") -> pd.DataFrame:
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    if "tic" not in out.columns:
        out["tic"] = symbol.upper()
    out = out.sort_values(["date", "tic"]).reset_index(drop=True)
    return out


def _apply_date_filters(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Apply optional date range filters to normalized market dataframe."""
    out = df
    if start_date:
        start_ts = pd.to_datetime(start_date)
        out = out[out["date"] >= start_ts]
    if end_date:
        end_ts = pd.to_datetime(end_date)
        # For date-only inputs (YYYY-MM-DD), include entire day.
        if len(end_date.strip()) <= 10:
            end_ts = end_ts + pd.Timedelta(days=1)
            out = out[out["date"] < end_ts]
        else:
            out = out[out["date"] <= end_ts]
    return out.reset_index(drop=True)


def load_market_data(
    source: str = "synthetic",
    input_csv: Optional[str] = None,
    timeframe: str = "1m",
    symbol: str = "BTCUSDT",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    market_type: str = "futures",
    num_rows: int = 8_000,
    synthetic_drift: float = 0.0,
    synthetic_regime_amplitude: float = 0.0004,
    synthetic_regime_period: int = 720,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load market data from synthetic/csv/binance source.

    Notes:
      - `synthetic`: generated at 1m then optionally aggregated to timeframe.
      - `csv`: loaded as-is (expects OHLCV columns with `date`).
      - `binance`: downloads using requested interval directly.
    """
    source_key = source.lower().strip()

    if source_key == "synthetic":
        df = make_synthetic_market_df(
            num_rows=num_rows,
            seed=seed,
            drift=synthetic_drift,
            regime_amplitude=synthetic_regime_amplitude,
            regime_period=synthetic_regime_period,
        )
        if timeframe != "1m":
            from preprocessing.timeframe import aggregate_ohlcv

            df = aggregate_ohlcv(
                df, timeframe=timeframe, drop_incomplete_last_candle=True
            )
    elif source_key == "csv":
        if not input_csv:
            raise ValueError("input_csv is required when source='csv'")
        df = pd.read_csv(input_csv, parse_dates=["date"])
        _validate_csv_timeframe(df, timeframe=timeframe, input_csv=input_csv)
    elif source_key == "binance":
        if not start_date or not end_date:
            raise ValueError(
                "start_date and end_date are required when source='binance'"
            )
        from downloader import download

        df = download(
            start_date=start_date,
            end_date=end_date,
            ticker_list=[symbol],
            time_interval=timeframe,
            market_type=market_type,
        )
    else:
        raise ValueError(f"Unsupported source: {source}")

    normalized = _normalize_market_df(df, symbol=symbol)
    return _apply_date_filters(normalized, start_date=start_date, end_date=end_date)
