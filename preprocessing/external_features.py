"""Leak-safe external macro/premium feature joins.

The wave_trading project already has cached PostgreSQL-derived forex, USDKRW,
and Upbit BTC/KRW bars.  This module reuses those local caches when available
and joins the resulting Dollar Index proxy and Kimchi Premium onto this repo's
BTCUSDT bars without looking forward in time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DXY_WEIGHTS: dict[str, float] = {
    "EURUSD": -0.576,
    "USDJPY": 0.136,
    "GBPUSD": -0.119,
    "USDCAD": 0.091,
    "USDSEK": 0.042,
    "USDCHF": 0.036,
}


def _coerce_naive_utc(series: pd.Series) -> pd.Series:
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz_convert(None)
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="raise", utc=True).dt.tz_convert(None)


def _normalise_dates(df: pd.DataFrame, *, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = _coerce_naive_utc(out[date_col])
    return out.sort_values(date_col).drop_duplicates(date_col, keep="last").reset_index(drop=True)


def _infer_interval(market: pd.DataFrame) -> str:
    if len(market) < 3 or "date" not in market.columns:
        return "1m"
    dates = pd.to_datetime(market["date"], errors="coerce").sort_values()
    delta = dates.diff().dropna().median()
    if pd.isna(delta):
        return "1m"
    minutes = max(1, int(round(delta.total_seconds() / 60.0)))
    return f"{minutes}min"


def _resample_close_by_tic(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    needed = {"date", "tic", "close"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"external bar frame lacks columns: {sorted(missing)}")
    out = df.loc[:, ["date", "tic", "close"]].copy()
    out = out.dropna(subset=["date", "tic", "close"])
    out["date"] = _coerce_naive_utc(out["date"])
    out["close"] = out["close"].astype(float)
    freq = str(interval).replace("m", "min") if str(interval).endswith("m") and not str(interval).endswith("min") else str(interval)
    if freq in {"1m", "1min"}:
        return out.sort_values(["date", "tic"]).reset_index(drop=True)
    return (
        out.set_index("date")
        .groupby("tic")
        .resample(freq)["close"]
        .last()
        .dropna()
        .reset_index()
        .sort_values(["date", "tic"])
        .reset_index(drop=True)
    )


def calculate_dollar_index(forex_bars: pd.DataFrame, *, interval: str = "1min") -> pd.DataFrame:
    """Return a DXY-like index from past/at-time forex closes."""
    bars = _resample_close_by_tic(forex_bars, interval)
    if bars.empty:
        return pd.DataFrame(columns=["date", "dxy"])
    pivot = bars.pivot(index="date", columns="tic", values="close").sort_index().ffill()
    weighted_log_return = pd.Series(0.0, index=pivot.index, dtype=float)
    has_component = False
    for ticker, weight in DXY_WEIGHTS.items():
        if ticker not in pivot.columns:
            continue
        has_component = True
        positive_close = pivot[ticker].where(pivot[ticker] > 0.0)
        weighted_log_return = weighted_log_return.add(
            np.log(positive_close / positive_close.shift(1)) * weight,
            fill_value=0.0,
        )
    if not has_component:
        return pd.DataFrame(columns=["date", "dxy"])
    dxy = (1.0 + weighted_log_return.fillna(0.0)).cumprod() * 100.0
    return pd.DataFrame({"date": dxy.index, "dxy": dxy.astype(float).values})


def calculate_forex_component_features(forex_bars: pd.DataFrame, *, interval: str = "1min") -> pd.DataFrame:
    """Return individual FX component closes for lead-lag feature mining.

    Columns are point-in-time resampled closes named ``fx_<ticker>``.  They are
    later joined with backward-asof semantics, so a market row never sees future
    FX data.
    """
    bars = _resample_close_by_tic(forex_bars, interval)
    if bars.empty:
        return pd.DataFrame(columns=["date"])
    pivot = bars.pivot(index="date", columns="tic", values="close").sort_index().ffill()
    cols = [c for c in DXY_WEIGHTS if c in pivot.columns]
    if not cols:
        return pd.DataFrame(columns=["date"])
    out = pivot.loc[:, cols].rename(columns={c: f"fx_{c.lower()}" for c in cols}).reset_index()
    return out


def calculate_kimchi_premium(
    btcusdt_bars: pd.DataFrame,
    btckrw_bars: pd.DataFrame,
    usdkrw_bars: pd.DataFrame,
    *,
    interval: str = "1min",
) -> pd.DataFrame:
    """Return Kimchi Premium = BTC/KRW / (BTC/USDT * USD/KRW) - 1."""
    btc = _resample_close_by_tic(_with_tic(btcusdt_bars, "BTCUSDT"), interval)
    krw = _resample_close_by_tic(_with_tic(btckrw_bars, "KRW-BTC"), interval)
    fx = _resample_close_by_tic(_with_tic(usdkrw_bars, "USDKRW"), interval)
    btc_s = btc[btc["tic"].eq("BTCUSDT")].set_index("date")["close"].rename("btcusdt")
    krw_s = krw[krw["tic"].eq("KRW-BTC")].set_index("date")["close"].rename("btckrw")
    fx_s = fx[fx["tic"].eq("USDKRW")].set_index("date")["close"].rename("usdkrw")
    joined = pd.concat([btc_s, krw_s, fx_s], axis=1).dropna()
    if joined.empty:
        return pd.DataFrame(columns=["date", "kimchi_premium", "usdkrw", "btckrw"])
    premium = joined["btckrw"] / (joined["btcusdt"] * joined["usdkrw"]) - 1.0
    return pd.DataFrame(
        {
            "date": joined.index,
            "kimchi_premium": premium.astype(float).values,
            "usdkrw": joined["usdkrw"].astype(float).values,
            "btckrw": joined["btckrw"].astype(float).values,
        }
    )


def _with_tic(df: pd.DataFrame, tic: str) -> pd.DataFrame:
    out = df.copy()
    if "tic" not in out.columns:
        out["tic"] = tic
    return out


def backward_asof_external_join(
    market: pd.DataFrame,
    external: pd.DataFrame,
    *,
    tolerance: str | pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Backward-asof join external rows onto market rows; never uses future rows."""
    if external.empty:
        return market.copy()
    market_sorted = _normalise_dates(market)
    external_sorted = _normalise_dates(external)
    tol = pd.Timedelta(tolerance) if tolerance else None
    joined = pd.merge_asof(
        market_sorted,
        external_sorted,
        on="date",
        direction="backward",
        tolerance=tol,
    )
    return joined.sort_values("date").reset_index(drop=True)


def add_external_derived_features(market: pd.DataFrame, *, zscore_window: int = 96, momentum_period: int = 96) -> pd.DataFrame:
    out = market.copy()
    if "dxy" in out.columns:
        dxy = out["dxy"].astype(float)
        out["dxy_zscore"] = _rolling_zscore(dxy, zscore_window)
        out["dxy_momentum"] = _pct_change(dxy, momentum_period)
    if "kimchi_premium" in out.columns:
        kimchi = out["kimchi_premium"].astype(float)
        out["kimchi_premium_zscore"] = _rolling_zscore(kimchi, zscore_window)
        out["kimchi_premium_change"] = kimchi.diff(max(1, int(momentum_period))).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if "usdkrw" in out.columns:
        usdkrw = out["usdkrw"].astype(float)
        out["usdkrw_zscore"] = _rolling_zscore(usdkrw, zscore_window)
        out["usdkrw_momentum"] = _pct_change(usdkrw, momentum_period)
    if "btckrw" in out.columns:
        btckrw = out["btckrw"].astype(float)
        out["btckrw_zscore"] = _rolling_zscore(btckrw, zscore_window)
        out["btckrw_momentum"] = _pct_change(btckrw, momentum_period)
    for col in [c for c in out.columns if str(c).startswith("fx_") and not str(c).endswith(("_zscore", "_momentum", "_available"))]:
        series = out[col].astype(float)
        out[f"{col}_zscore"] = _rolling_zscore(series, zscore_window)
        out[f"{col}_momentum"] = _pct_change(series, momentum_period)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    window = max(1, int(window))
    min_periods = min(window, max(2, window // 3))
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std(ddof=0)
    return ((series - mean) / std.replace(0.0, np.nan)).clip(-5.0, 5.0).fillna(0.0)


def _pct_change(series: pd.Series, period: int) -> pd.Series:
    ref = series.shift(max(1, int(period)))
    return ((series - ref) / ref.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_external_feature_frame(
    market: pd.DataFrame,
    *,
    forex_bars: pd.DataFrame | None = None,
    btckrw_bars: pd.DataFrame | None = None,
    usdkrw_bars: pd.DataFrame | None = None,
    interval: str | None = None,
    include_forex_components: bool = False,
) -> pd.DataFrame:
    interval = interval or _infer_interval(market)
    parts: list[pd.DataFrame] = []
    if forex_bars is not None and not forex_bars.empty:
        parts.append(calculate_dollar_index(forex_bars, interval=interval))
        if include_forex_components:
            parts.append(calculate_forex_component_features(forex_bars, interval=interval))
    if btckrw_bars is not None and usdkrw_bars is not None and not btckrw_bars.empty and not usdkrw_bars.empty:
        parts.append(calculate_kimchi_premium(market, btckrw_bars, usdkrw_bars, interval=interval))
    if not parts:
        return pd.DataFrame(columns=["date"])
    out = parts[0]
    for part in parts[1:]:
        out = pd.merge(out, part, on="date", how="outer")
    return out.sort_values("date").reset_index(drop=True)


def attach_external_features(
    market: pd.DataFrame,
    external: pd.DataFrame,
    *,
    tolerance: str | pd.Timedelta | None = None,
    zscore_window: int = 96,
    momentum_period: int = 96,
) -> pd.DataFrame:
    joined = backward_asof_external_join(market, external, tolerance=tolerance)
    # Preserve source availability before derived features fill missing values with
    # neutral zeros.  Without these flags, missing future macro/premium caches are
    # indistinguishable from true zero momentum/z-score regimes.
    if "dxy" in joined.columns:
        joined["dxy_available"] = joined["dxy"].notna().astype(float)
    if "kimchi_premium" in joined.columns:
        joined["kimchi_available"] = joined["kimchi_premium"].notna().astype(float)
    if "usdkrw" in joined.columns:
        joined["usdkrw_available"] = joined["usdkrw"].notna().astype(float)
    available_cols = [c for c in ("dxy_available", "kimchi_available", "usdkrw_available") if c in joined.columns]
    if available_cols:
        joined["external_any_available"] = joined[available_cols].max(axis=1)
    return add_external_derived_features(joined, zscore_window=zscore_window, momentum_period=momentum_period)


def load_wave_trading_cached_bars(
    wave_trading_root: str | Path,
    tickers: Iterable[str],
) -> pd.DataFrame:
    """Load and merge wave_trading cached bars matching all requested tickers.

    wave_trading caches are date-range keyed.  Choosing only the largest matching
    file silently ignored newer smaller 2026 extension caches.  This loader
    merges all matching cache shards, de-duplicates mirrored ``data`` and
    ``research/data`` files by filename, then de-duplicates bars by
    ``(date, tic)`` using the latest loaded row.
    """
    root = Path(wave_trading_root).expanduser().resolve()
    ticker_set = {str(t) for t in tickers}
    candidates = list((root / "data").glob("*.csv.gz")) + list((root / "research" / "data").glob("*.csv.gz"))
    selected: list[Path] = []
    seen_names: set[str] = set()
    for path in sorted(candidates, key=lambda p: (p.name, str(p))):
        path = path.resolve()
        if path.name in seen_names:
            continue
        try:
            preview = pd.read_csv(path, usecols=["tic"], nrows=200_000)
        except Exception:
            continue
        present = {str(x) for x in preview["tic"].dropna().unique()}
        if ticker_set.issubset(present):
            selected.append(path)
            seen_names.add(path.name)
    if not selected:
        raise FileNotFoundError(f"no wave_trading cache with tickers {sorted(ticker_set)} under {root}")
    frames = [pd.read_csv(path, parse_dates=["date"], compression="gzip") for path in selected]
    out = pd.concat(frames, ignore_index=True)
    out = out[out["tic"].astype(str).isin(ticker_set)].copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise", utc=True).dt.tz_convert(None)
    out = out.sort_values(["date", "tic"]).drop_duplicates(["date", "tic"], keep="last").reset_index(drop=True)
    return out


def attach_wave_trading_external_features(
    market: pd.DataFrame,
    *,
    wave_trading_root: str | Path = "../workspace/wave_trading",
    tolerance: str | pd.Timedelta | None = None,
    interval: str | None = None,
    include_forex_components: bool = False,
) -> pd.DataFrame:
    """Attach DXY/Kimchi features from wave_trading local caches."""
    forex = load_wave_trading_cached_bars(wave_trading_root, DXY_WEIGHTS.keys())
    usdkrw = load_wave_trading_cached_bars(wave_trading_root, ["USDKRW"])
    btckrw = load_wave_trading_cached_bars(wave_trading_root, ["KRW-BTC"])
    external = build_external_feature_frame(
        market,
        forex_bars=forex,
        btckrw_bars=btckrw,
        usdkrw_bars=usdkrw,
        interval=interval,
        include_forex_components=include_forex_components,
    )
    return attach_external_features(market, external, tolerance=tolerance)
