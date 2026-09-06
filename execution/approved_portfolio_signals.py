"""Approved pure signal adapters for portfolio sleeves.

This module intentionally contains no exchange, order, network, or live-config
side effects.  It adapts frozen research formulas into causal production-facing
signals with explicit cutoffs and conservative source-missingness handling.

Parity notes
------------
* ``build_macro_targets`` mirrors the frozen macro-flow production candidate:
  75% dollar_flow6 vol20 plus 25% flow_switch720_long, daily UTC refresh and
  hourly target maintenance.  It uses the same completed-hour market features
  and macro feature formulas as the research code.  DXY availability is mandatory; an unavailable DXY refresh disables only
  that component. Daily-held targets are not silently regated intraday.
  Unused Kimchi/USDKRW inputs are not prerequisites for this approved formula.
* ``score_dollar_short`` mirrors the legacy dollar-rally short gates and the
  144-row/12-hour lifecycle, but fail-closes when explicit source-availability
  flags or required finite inputs are missing.  The legacy audit's "original"
  replay did not require DXY availability at the signal row; this adapter does
  because production execution must preserve source missingness conservatively.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

MACRO_DOLLAR_WEIGHT = 0.75
MACRO_SWITCH_WEIGHT = 0.25
ANNUAL_HOURS = 8766.0

DOLLAR_SHORT_DXY_MOMENTUM_THRESHOLD = 0.0021818982809893497
DOLLAR_SHORT_HTF_1D_RETURN_4_THRESHOLD = 0.016096783732847175
DOLLAR_SHORT_WINDOW_SIZE = 144
DOLLAR_SHORT_HOLD_BARS = 144
DOLLAR_SHORT_STRIDE_BARS = 12
DOLLAR_SHORT_PHASE_OFFSET_BARS = DOLLAR_SHORT_WINDOW_SIZE - 1
DOLLAR_SHORT_HOLD = pd.Timedelta(hours=12)
DOLLAR_SHORT_PHASE_ANCHOR = pd.Timestamp("2019-12-31 15:00:00", tz="UTC")

_MARKET_REQUIRED = ("date", "open", "high", "low", "close")
_MACRO_REQUIRED = _MARKET_REQUIRED + ("quote_asset_volume", "taker_buy_quote", "dxy", "dxy_available")


@dataclass(frozen=True)
class Cutoff:
    """Normalized inclusive data cutoff for causal computations."""

    asof: pd.Timestamp


def _utc_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _normalize_market(market: pd.DataFrame, *, required: tuple[str, ...], asof: Any | None) -> tuple[pd.DataFrame, Cutoff]:
    """Return a UTC-sorted copy limited to rows whose timestamps are <= asof."""

    missing = [col for col in required if col not in market.columns]
    if missing:
        raise ValueError(f"market missing required columns: {missing}")
    out = market.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
    if out["date"].isna().any():
        raise ValueError("market.date contains non-timestamp values")
    if out["date"].duplicated().any():
        raise ValueError("Duplicate market timestamps")
    out = out.sort_values("date").reset_index(drop=True)
    if out.empty:
        raise ValueError("Empty market")
    if (out["date"].astype("int64") % pd.Timedelta("5min").value).any():
        raise ValueError("Off-grid market timestamps")
    cutoff_ts = _utc_timestamp(asof) if asof is not None else pd.Timestamp(out["date"].iloc[-1])
    out = out.loc[out["date"] <= cutoff_ts].reset_index(drop=True)
    if out.empty:
        raise ValueError("market has no rows at or before asof")
    if not out["date"].diff().dropna().eq(pd.Timedelta("5min")).all():
        raise ValueError("Market grid gap")
    for col in required:
        if col == "date":
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for volume_column in ("quote_asset_volume", "taker_buy_quote"):
        if volume_column in required:
            values = out[volume_column].to_numpy(float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Invalid volume source: {volume_column}")
    if "quote_asset_volume" in required and "taker_buy_quote" in required:
        if (out["taker_buy_quote"] > out["quote_asset_volume"] * (1 + 1e-9) + 1e-9).any():
            raise ValueError("Taker volume exceeds total volume")
    prices = out[["open", "high", "low", "close"]].to_numpy(float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Invalid market prices")
    return out, Cutoff(asof=cutoff_ts)


def _finite_all(frame: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> pd.Series:
    if not columns:
        return pd.Series(True, index=frame.index)
    arr = frame.loc[:, list(columns)].to_numpy(dtype=float, copy=False)
    return pd.Series(np.isfinite(arr).all(axis=1), index=frame.index)


def _completed_hourly_market(market: pd.DataFrame) -> pd.DataFrame:
    """Completed hourly OHLCV rows labelled at the hour end, matching research."""

    indexed = market.set_index("date")
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "quote_asset_volume": "sum",
        "taker_buy_quote": "sum",
    }
    h = indexed.resample("1h", label="right", closed="left").agg(agg)
    counts = indexed["close"].resample("1h", label="right", closed="left").count()
    h = h.loc[counts.eq(12)]
    return h


def _base_hourly_features(market: pd.DataFrame) -> pd.DataFrame:
    h = _completed_hourly_market(market)
    close = h["close"].astype(float)
    positive_close = close.where(close > 0.0)
    log_close = np.log(positive_close)
    returns = log_close.diff()
    x = pd.DataFrame(index=h.index)
    for window in (6, 24, 720):
        vol = returns.rolling(window, min_periods=window).std(ddof=0)
        x[f"mom{window}"] = log_close.diff(window) / (vol * np.sqrt(window)).replace(0.0, np.nan)
        numerator = (2.0 * h["taker_buy_quote"] - h["quote_asset_volume"]).rolling(window).sum()
        denominator = h["quote_asset_volume"].rolling(window).sum().replace(0.0, np.nan)
        x[f"flow{window}"] = numerator / denominator
        mean = close.rolling(window).mean()
        std = close.rolling(window).std(ddof=0)
        x[f"z{window}"] = (close - mean) / std.replace(0.0, np.nan)
    x["vol24"] = returns.rolling(24).std(ddof=0)
    return x.replace([np.inf, -np.inf], np.nan)


def _macro_features(raw: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Research-parity macro features: completed-hour last value, delayed 1h."""

    source = raw.copy()
    source["date"] = pd.to_datetime(source["date"], utc=True, errors="coerce").dt.tz_convert(None)
    h = source.set_index("date").resample("1h", label="right", closed="left").last().shift(1)
    out = pd.DataFrame(index=h.index)
    for field, flag in (("dxy", "dxy_available"), ("usdkrw", "usdkrw_available"), ("kimchi_premium", "kimchi_available")):
        if field not in h.columns:
            continue
        valid = h[flag] > 0.5 if flag in h.columns else pd.Series(False, index=h.index)
        values = h[field].where(valid)
        if field != "kimchi_premium":
            values = np.log(values.where(values > 0.0))
        out[f"{field}_valid"] = valid.astype(float)
        for hours in (6, 24):
            out[f"{field}_change{hours}"] = values.diff(hours)
        std = values.rolling(168, min_periods=72).std(ddof=0).replace(0.0, np.nan)
        out[f"{field}_z"] = (values - values.rolling(168, min_periods=72).mean()) / std
    return out.reindex(index).replace([np.inf, -np.inf], np.nan)


def _hold_signal(raw: np.ndarray, hours: int, dates: pd.DatetimeIndex) -> np.ndarray:
    s = pd.Series(np.asarray(raw, dtype=float), index=pd.DatetimeIndex(dates))
    mask = s.index.hour % int(hours) == 0
    return s.where(mask).ffill().fillna(0.0).to_numpy(dtype=float)


def build_macro_targets(market: pd.DataFrame, *, asof: Any | None = None) -> pd.Series:
    """Build approved macro-flow hourly targets indexed by execution time T+5m.

    ``asof`` is wall-clock time: each source bar must have closed by it.  Only complete 12-row UTC hours
    ending at or before that cutoff can produce a signal, and any row missing a
    required finite input or explicit macro availability flag is flat.
    """

    cutoff = _utc_timestamp(asof) if asof is not None else _utc_timestamp(market["date"].iloc[-1]) + pd.Timedelta("5min")
    m, _ = _normalize_market(market, required=_MACRO_REQUIRED, asof=cutoff)
    m = m.loc[m["date"] + pd.Timedelta("5min") <= cutoff].copy()
    if m.empty:
        return pd.Series(dtype=float, name="macro_flow_target")

    calc = m.copy()
    calc["date"] = calc["date"].dt.tz_convert(None)
    x = _base_hourly_features(calc)
    macro = _macro_features(
        calc[[c for c in ("date", "dxy", "usdkrw", "kimchi_premium", "dxy_available", "usdkrw_available", "kimchi_available") if c in calc.columns]],
        x.index,
    )
    x = pd.concat([x, macro], axis=1)

    n = len(x)
    if n == 0:
        return pd.Series(dtype=float, name="macro_flow_target")
    vol24 = x["vol24"].to_numpy(dtype=float)
    size = np.clip(np.divide(0.2, vol24 * np.sqrt(ANNUAL_HOURS), out=np.ones(n), where=vol24 > 0.0), 0.1, 1.0)
    flow = x["flow6"].to_numpy(dtype=float)
    dollar = x["dxy_change6"].to_numpy(dtype=float)
    mom = x["mom720"].to_numpy(dtype=float)
    z = x["z24"].to_numpy(dtype=float)

    dollar_inputs_ok = _finite_all(x, ["vol24", "flow6", "dxy_change6"]) & (
        x.get("dxy_valid", pd.Series(1.0, index=x.index)) > 0.5
    )
    dollar_raw = np.where((np.abs(flow) > 0.02) & (np.sign(flow) * dollar < 0.0), np.sign(flow), 0.0) * size
    dollar_raw = np.where(dollar_inputs_ok.to_numpy(bool), dollar_raw, 0.0)
    dollar_signal = _hold_signal(dollar_raw, 24, x.index)
    trend = np.abs(mom) > 0.75
    direction = np.sign(mom)
    reverse = np.where(np.abs(z) > 1.5, -np.sign(z), 0.0)
    switch_inputs_ok = _finite_all(x, ["mom720", "flow6", "z24"])
    switch_raw = np.maximum(np.where(trend, np.where(direction * flow > 0.0, direction, 0.0), reverse), 0.0)
    switch_raw = np.where(switch_inputs_ok.to_numpy(bool), switch_raw, 0.0)
    switch_signal = _hold_signal(switch_raw, 24, x.index)

    target = MACRO_DOLLAR_WEIGHT * dollar_signal + MACRO_SWITCH_WEIGHT * switch_signal
    target = np.clip(target, -1.0, 1.0)
    execution_index = (x.index + pd.Timedelta(minutes=5)).tz_localize("UTC")
    return pd.Series(target, index=execution_index, name="macro_flow_target")


def _original_144_features(market: pd.DataFrame) -> pd.DataFrame:
    """Subset of the original 144-window feature contract needed by top0.

    ``dxy_momentum`` is consumed from the enriched market frame when present,
    matching ``build_market_feature_frame``.  When only raw DXY is supplied, a
    96-row percentage change fallback matches the external feature default used
    by the historical enrichment path.  The 1D feature mirrors the completed,
    previous daily-candle implementation for ``window_size=144``.
    """

    source = market.copy()
    source["date"] = pd.to_datetime(source["date"], utc=True, errors="coerce")
    out = pd.DataFrame(index=source.index)
    if "dxy_momentum" in source.columns:
        out["dxy_momentum"] = pd.to_numeric(source["dxy_momentum"], errors="coerce")
    else:
        dxy = pd.to_numeric(source["dxy"], errors="coerce") if "dxy" in source.columns else pd.Series(np.nan, index=source.index)
        out["dxy_momentum"] = dxy / dxy.shift(96).replace(0.0, np.nan) - 1.0

    if len(source) < 24 * 60 * 4:
        out["htf_1d_return_4"] = 0.0
        return out.replace([np.inf, -np.inf], np.nan)

    daily = source[["date", "open", "high", "low", "close"]].set_index("date")
    htf = pd.DataFrame(
        {
            "open": daily["open"].resample("1D", label="right", closed="right").first(),
            "high": daily["high"].resample("1D", label="right", closed="right").max(),
            "low": daily["low"].resample("1D", label="right", closed="right").min(),
            "close": daily["close"].resample("1D", label="right", closed="right").last(),
        }
    ).dropna()
    prev = htf.shift(1)
    features = pd.DataFrame({"date": htf.index, "htf_1d_return_4": (prev["close"] / prev["close"].shift(4).replace(0.0, np.nan) - 1.0).to_numpy()})
    aligned = pd.merge_asof(
        pd.DataFrame({"date": source["date"], "_row": np.arange(len(source))}).sort_values("date"),
        features.sort_values("date"),
        on="date",
        direction="backward",
    ).sort_values("_row")
    out["htf_1d_return_4"] = aligned["htf_1d_return_4"].to_numpy()
    return out.replace([np.inf, -np.inf], np.nan)


def _is_legacy_signal_phase(ts: pd.Timestamp) -> bool:
    ts = _utc_timestamp(ts)
    delta = ts - DOLLAR_SHORT_PHASE_ANCHOR
    if delta < pd.Timedelta(0):
        return False
    bars = delta / pd.Timedelta(minutes=5)
    return float(bars).is_integer() and int(bars) % DOLLAR_SHORT_STRIDE_BARS == DOLLAR_SHORT_PHASE_OFFSET_BARS % DOLLAR_SHORT_STRIDE_BARS


def score_dollar_short(market: pd.DataFrame, decision_bar_date: Any) -> dict[str, Any]:
    """Score the frozen legacy dollar-rally short at one decision 5m bar.

    Returns a serializable dict.  The adapter never orders; when active, the
    lifecycle is fixed 1x short entry at ``decision_bar_date + 5m`` and exit 144
    five-minute bars later (12 hours), with no TP/SL.
    """

    decision = _utc_timestamp(decision_bar_date)
    required = ("date", "open", "high", "low", "close", "dxy_momentum", "dxy_available")
    m, _ = _normalize_market(market, required=required, asof=decision)
    pos = m.index[m["date"].eq(decision)]
    if len(pos) != 1:
        return {
            "name": "dollar_rally_short",
            "active": False,
            "position": 0.0,
            "decision_time": decision.isoformat(),
            "reason": "decision_bar_not_available_at_cutoff",
        }
    i = int(pos[0])
    execution = decision + pd.Timedelta(minutes=5)
    exit_time = execution + DOLLAR_SHORT_HOLD
    reasons: list[str] = []
    if not _is_legacy_signal_phase(decision):
        reasons.append("off_global_phase")
    if i < DOLLAR_SHORT_WINDOW_SIZE - 1 or i < 288 * 4:
        reasons.append("insufficient_warmup")
    features = _original_144_features(m)
    dxy_momentum = float(features.at[i, "dxy_momentum"]) if i in features.index else float("nan")
    htf_1d_return_4 = float(features.at[i, "htf_1d_return_4"]) if i in features.index else float("nan")
    if not np.isfinite(dxy_momentum) or not np.isfinite(htf_1d_return_4):
        reasons.append("nonfinite_required_inputs")
    if "dxy_available" in m.columns and not bool(float(m.at[i, "dxy_available"]) > 0.5):
        reasons.append("dxy_unavailable_at_signal")
    gates = {
        "dxy_momentum": {
            "value": dxy_momentum,
            "op": ">=",
            "threshold": DOLLAR_SHORT_DXY_MOMENTUM_THRESHOLD,
            "passed": bool(np.isfinite(dxy_momentum) and dxy_momentum >= DOLLAR_SHORT_DXY_MOMENTUM_THRESHOLD),
        },
        "htf_1d_return_4": {
            "value": htf_1d_return_4,
            "op": ">=",
            "threshold": DOLLAR_SHORT_HTF_1D_RETURN_4_THRESHOLD,
            "passed": bool(np.isfinite(htf_1d_return_4) and htf_1d_return_4 >= DOLLAR_SHORT_HTF_1D_RETURN_4_THRESHOLD),
        },
    }
    if not gates["dxy_momentum"]["passed"]:
        reasons.append("dxy_momentum_below_threshold")
    if not gates["htf_1d_return_4"]["passed"]:
        reasons.append("htf_1d_return_4_below_threshold")
    active = not reasons
    return {
        "name": "dollar_rally_short",
        "active": bool(active),
        "position": -1.0 if active else 0.0,
        "side": "SHORT" if active else "FLAT",
        "decision_time": decision.isoformat(),
        "execution_time": execution.isoformat(),
        "lifecycle": {
            "entry_time": execution.isoformat(),
            "exit_time": exit_time.isoformat(),
            "hold_bars_5m": DOLLAR_SHORT_HOLD_BARS,
            "hold_hours": 12,
            "take_profit": None,
            "stop_loss": None,
        },
        "global_phase": {
            "anchor": DOLLAR_SHORT_PHASE_ANCHOR.isoformat(),
            "signal_minute": 55,
            "entry_minute": 0,
            "matched": _is_legacy_signal_phase(decision),
        },
        "gates": gates,
        "reason": "active" if active else ";".join(dict.fromkeys(reasons)),
    }
