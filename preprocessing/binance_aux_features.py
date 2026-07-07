"""Leak-safe Binance USD-M auxiliary feature joins.

This module attaches public futures auxiliary market data (funding history and
premium index klines) to bar data by backward-as-of joins only.  Premium kline
``close`` values are timestamped by ``close_time`` when present so the hourly
premium value is not visible before the premium kline has completed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _coerce_naive_utc(series: pd.Series) -> pd.Series:
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz_convert(None)
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="raise", utc=True).dt.tz_convert(None)


def _normalise_market_dates(market: pd.DataFrame) -> pd.DataFrame:
    if "date" not in market.columns:
        raise ValueError("market frame must contain a date column")
    out = market.copy()
    out["date"] = _coerce_naive_utc(out["date"])
    out["_row"] = np.arange(len(out))
    return out.sort_values("date").reset_index(drop=True)


def _read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path).expanduser(), compression="infer")


def load_funding_history(path: str | Path) -> pd.DataFrame:
    """Load downloader output as ``date, funding_rate`` rows."""
    return normalise_funding_history_frame(_read_csv(path))


def normalise_funding_history_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise funding rows from CSV or live DB query output."""
    if "funding_time" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"funding_time": "date"})
    missing = {"date", "funding_rate"}.difference(df.columns)
    if missing:
        raise ValueError(f"funding frame lacks columns: {sorted(missing)}")
    out = df.loc[:, ["date", "funding_rate"]].copy()
    out["date"] = _coerce_naive_utc(out["date"])
    out["funding_rate"] = pd.to_numeric(out["funding_rate"], errors="coerce")
    return out.dropna(subset=["date", "funding_rate"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def load_premium_index_klines(path: str | Path) -> pd.DataFrame:
    """Load premium index kline output with causal timestamps.

    The Binance endpoint returns OHLC rows.  When ``close_time`` exists we use it
    as the availability timestamp for the row's ``close`` value; otherwise we
    fall back to ``date``.  This avoids exposing an hourly close at the hour open.
    """
    return normalise_premium_index_frame(_read_csv(path))


def normalise_premium_index_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise premium-index kline rows from CSV or live DB query output.

    When ``close_time`` exists, it is used as the causal availability timestamp.
    DB rows may store it as a timestamp, while downloader CSV rows may store it
    as milliseconds since epoch.
    """
    if "premium_index" in df.columns and "date" in df.columns:
        out = df.loc[:, ["date", "premium_index"]].copy()
        out["date"] = _coerce_naive_utc(out["date"])
        out["premium_index"] = pd.to_numeric(out["premium_index"], errors="coerce")
        return out.dropna(subset=["date", "premium_index"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if "close" not in df.columns:
        raise ValueError("premium frame lacks close column")
    if "close_time" in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df["close_time"]):
            dates = _coerce_naive_utc(df["close_time"])
        else:
            numeric_close_time = pd.to_numeric(df["close_time"], errors="coerce")
            if numeric_close_time.notna().any():
                dates = pd.to_datetime(numeric_close_time, unit="ms", utc=True).dt.tz_convert(None)
            else:
                dates = pd.to_datetime(df["close_time"], errors="raise", utc=True).dt.tz_convert(None)
    elif "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="raise", utc=True).dt.tz_convert(None)
    else:
        raise ValueError("premium frame lacks close_time/date column")
    out = pd.DataFrame(
        {
            "date": dates,
            "premium_index": pd.to_numeric(df["close"], errors="coerce"),
        }
    )
    return out.dropna(subset=["date", "premium_index"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _merge_aux(
    market: pd.DataFrame,
    aux: pd.DataFrame,
    *,
    value_cols: list[str],
    tolerance: str | pd.Timedelta | None,
) -> tuple[pd.DataFrame, pd.Series]:
    if aux.empty:
        available = pd.Series(0.0, index=market.index)
        return market.copy(), available
    market_sorted = _normalise_market_dates(market)
    aux_sorted = aux.loc[:, ["date", *value_cols]].copy().sort_values("date")
    rename = {col: f"__aux_{col}" for col in value_cols if col in market_sorted.columns}
    aux_join = aux_sorted.rename(columns=rename)
    joined = pd.merge_asof(
        market_sorted,
        aux_join,
        on="date",
        direction="backward",
        tolerance=pd.Timedelta(tolerance) if tolerance else None,
    )
    availability_source = rename.get(value_cols[0], value_cols[0])
    joined["__aux_available"] = joined[availability_source].notna().astype(float)
    for col in value_cols:
        source = rename.get(col, col)
        if source not in joined.columns:
            continue
        source_values = pd.to_numeric(joined[source], errors="coerce")
        if col in market.columns:
            base = pd.to_numeric(joined[col], errors="coerce")
            joined[col] = source_values.combine_first(base)
            if source != col:
                joined = joined.drop(columns=[source])
        else:
            if source != col:
                joined[col] = source_values
                joined = joined.drop(columns=[source])
            else:
                joined[col] = source_values
    joined = joined.sort_values("_row").reset_index(drop=True)
    available = joined.pop("__aux_available").astype(float)
    joined = joined.drop(columns=["_row"])
    return joined, available.reset_index(drop=True)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    window = max(1, int(window))
    min_periods = min(window, max(2, window // 3))
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std(ddof=0)
    return ((series - mean) / std.replace(0.0, np.nan)).clip(-5.0, 5.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def attach_binance_um_aux_features(
    market: pd.DataFrame,
    *,
    funding_csv: str | Path | None = None,
    premium_csv: str | Path | None = None,
    funding_tolerance: str | pd.Timedelta | None = "12h",
    premium_tolerance: str | pd.Timedelta | None = "2h",
    zscore_window: int = 96,
) -> pd.DataFrame:
    """Attach BTC/USDT futures auxiliary rows to market bars without lookahead."""
    return attach_binance_um_aux_frames(
        market,
        funding_frame=load_funding_history(funding_csv) if funding_csv else None,
        premium_frame=load_premium_index_klines(premium_csv) if premium_csv else None,
        funding_tolerance=funding_tolerance,
        premium_tolerance=premium_tolerance,
        zscore_window=zscore_window,
    )


def attach_binance_um_aux_frames(
    market: pd.DataFrame,
    *,
    funding_frame: pd.DataFrame | None = None,
    premium_frame: pd.DataFrame | None = None,
    funding_tolerance: str | pd.Timedelta | None = "12h",
    premium_tolerance: str | pd.Timedelta | None = "2h",
    zscore_window: int = 96,
) -> pd.DataFrame:
    """Attach DB/DataFrame-sourced Binance USD-M aux rows without lookahead."""
    out = market.copy()
    availability_cols: list[str] = []
    if funding_frame is not None and not funding_frame.empty:
        out, available = _merge_aux(
            out,
            normalise_funding_history_frame(funding_frame),
            value_cols=["funding_rate"],
            tolerance=funding_tolerance,
        )
        out["funding_available"] = available.to_numpy(dtype=float)
        availability_cols.append("funding_available")
    if premium_frame is not None and not premium_frame.empty:
        out, available = _merge_aux(
            out,
            normalise_premium_index_frame(premium_frame),
            value_cols=["premium_index"],
            tolerance=premium_tolerance,
        )
        premium = pd.to_numeric(out["premium_index"], errors="coerce")
        out["premium_index_zscore"] = _rolling_zscore(premium, zscore_window)
        out["premium_index_change"] = premium.diff(max(1, int(zscore_window))).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["premium_available"] = available.to_numpy(dtype=float)
        availability_cols.append("premium_available")
    if availability_cols:
        out["binance_aux_any_available"] = out[availability_cols].max(axis=1)
    return out.replace([np.inf, -np.inf], np.nan)
