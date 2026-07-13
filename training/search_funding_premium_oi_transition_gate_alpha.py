"""Search a bounded OI-state transition gate on the fixed LR-impact long alpha.

The funding/premium + central ``lr_impact_72`` base rule is immutable.  OI
state thresholds and transition rankings are learned only from 2021-04-15 to
2022-12-31, policies are selected only on 2023/H1/H2 while every source is
physically truncated before 2024, and the frozen Top-10 is replayed unchanged
on 2024, 2025 and 2026 YTD.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.binance_aux_features import (
    attach_binance_um_aux_frames,
    normalise_funding_history_frame,
    normalise_premium_index_frame,
)
from preprocessing.market_features import _completed_timeframe_features
from training.search_funding_premium_external_state_gate_alpha import (
    _file_sha256,
    _frame_hash,
    _manifest_core_hash,
    _read_premium_before,
    _validate_manifest,
)
from training.search_funding_premium_independent_gate_alpha import (
    DAILY_MOMENTUM_THRESHOLD,
    FUNDING_THRESHOLD,
    HOLD_BARS,
    PREMIUM_CHANGE_THRESHOLD,
    STRIDE_BARS,
    TREND_96_THRESHOLD,
)
from training.search_liquidity_recovery_bidirectional_alpha import features as build_liquidity_features
from training.search_positioning_disagreement_alpha import (
    _attach_delayed_metrics,
    _future_extreme,
    _simulate_no_stop,
)
from training.search_positioning_hgb_path_alpha import _feature_hash, _read_before


SELECTION_END = "2024-01-01"
FIT_WINDOW = ("2021-04-15", "2023-01-01")
LR_IMPACT_LOWER = -0.20030301257467914
LR_IMPACT_UPPER = 0.24664964484849766
BASE_ADMISSION_FEATURES = (
    "btc_funding_rate",
    "btc_premium_index_change",
    "btc_trend_96",
    "btc_daily_mom4",
    "btc_lr_impact_72",
)

WINDOWS = {
    "fit": FIT_WINDOW,
    "select_2023": ("2023-01-01", "2024-01-01"),
    "select_2023_h1": ("2023-01-01", "2023-07-01"),
    "select_2023_h2": ("2023-07-01", "2024-01-01"),
    "test_2024": ("2024-01-01", "2025-01-01"),
    "eval_2025": ("2025-01-01", "2026-01-01"),
    "holdout_2026": ("2026-01-01", "2026-06-02"),
    "oos_2024_2026": ("2024-01-01", "2026-06-02"),
}

QUARTER_WINDOWS = {
    "2024Q1": ("2024-01-01", "2024-04-01"),
    "2024Q2": ("2024-04-01", "2024-07-01"),
    "2024Q3": ("2024-07-01", "2024-10-01"),
    "2024Q4": ("2024-10-01", "2025-01-01"),
    "2025Q1": ("2025-01-01", "2025-04-01"),
    "2025Q2": ("2025-04-01", "2025-07-01"),
    "2025Q3": ("2025-07-01", "2025-10-01"),
    "2025Q4": ("2025-10-01", "2026-01-01"),
    "2026Q1": ("2026-01-01", "2026-04-01"),
    "2026Q2_to_Jun02": ("2026-04-01", "2026-06-02"),
}


@dataclass(frozen=True)
class OITransitionGateConfig:
    input_csv: str
    metrics_csv: str
    funding_csv: str
    premium_csv: str
    output: str
    manifest_output: str
    exclude_from: str = "2026-06-02"
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    stress_fee_rate: float = 0.0009
    top_n: int = 10
    top_per_schema_mode: int = 2
    min_fit_state_observations: int = 5_000
    min_transition_trades: int = 8
    min_fit_trades: int = 36
    min_select_trades: int = 12
    min_half_trades: int = 5
    max_abs_spearman: float = 0.30
    metrics_tolerance: str = "5min"
    metrics_delay_bars: int = 1
    funding_tolerance: str = "12h"
    premium_tolerance: str = "65min"
    refresh_manifest: bool = False


def _window_mask(dates: pd.Series, name: str) -> np.ndarray:
    start, end = WINDOWS[name]
    return ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)


def _source_file_hashes(cfg: OITransitionGateConfig) -> dict[str, str]:
    return {
        str(Path(path)): _file_sha256(path)
        for path in (cfg.input_csv, cfg.metrics_csv, cfg.funding_csv, cfg.premium_csv)
    }


def _read_source_prefixes(cfg: OITransitionGateConfig, cutoff: str) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    frames = {
        "market": _read_before(cfg.input_csv, "date", cutoff),
        "metrics": _read_before(cfg.metrics_csv, "create_time", cutoff),
        "funding": _read_before(cfg.funding_csv, "date", cutoff),
        "premium": _read_premium_before(cfg.premium_csv, cutoff),
    }
    return frames, {name: _frame_hash(frame) for name, frame in frames.items()}


def _load_market(cfg: OITransitionGateConfig, *, cutoff: str) -> tuple[pd.DataFrame, pd.Series, dict[str, str]]:
    frames, prefix_hashes = _read_source_prefixes(cfg, cutoff)
    boundary = pd.Timestamp(cutoff)
    market = frames["market"].copy()
    market["date"] = pd.to_datetime(market["date"], utc=True, errors="raise").dt.tz_convert(None)
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    market = attach_binance_um_aux_frames(
        market,
        funding_frame=normalise_funding_history_frame(frames["funding"]),
        premium_frame=normalise_premium_index_frame(frames["premium"]),
        funding_tolerance=cfg.funding_tolerance,
        premium_tolerance=cfg.premium_tolerance,
    )
    market = _attach_delayed_metrics(
        market,
        frames["metrics"],
        tolerance=cfg.metrics_tolerance,
        delay_bars=cfg.metrics_delay_bars,
    )
    market["oi_available"] = (
        pd.to_numeric(market.get("sum_open_interest"), errors="coerce").gt(0.0)
        & pd.to_datetime(market.get("positioning_source_time"), errors="coerce").notna()
    ).astype(float)
    dates = pd.to_datetime(market["date"])
    if len(dates) and dates.max() >= boundary:
        raise RuntimeError("market rows were not physically truncated before cutoff")
    source_time = pd.to_datetime(market["positioning_source_time"], errors="coerce")
    if source_time.notna().any() and source_time.max() >= boundary:
        raise RuntimeError("OI source rows were not physically truncated before cutoff")
    return market, dates, prefix_hashes


def _build_base_components(market: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    close = pd.to_numeric(market["close"], errors="coerce")
    trend_96 = close / close.shift(95).replace(0.0, np.nan) - 1.0
    daily = _completed_timeframe_features(
        market,
        prefix="htf_1d",
        resample_rule="1D",
        min_source_rows=24 * 60 * 4,
    )
    liquidity = build_liquidity_features(market, pd.DataFrame(index=market.index))
    lr_impact = pd.to_numeric(liquidity["lr_impact_72"], errors="coerce")
    funding = pd.to_numeric(market["funding_rate"], errors="coerce")
    premium_change = pd.to_numeric(market["premium_index_change"], errors="coerce")
    funding_available = pd.to_numeric(market.get("funding_available", 0.0), errors="coerce").fillna(0.0)
    premium_available = pd.to_numeric(market.get("premium_available", 0.0), errors="coerce").fillna(0.0)
    funding_component = (
        (funding_available.to_numpy(float) > 0.5)
        & (funding.to_numpy(float) <= FUNDING_THRESHOLD)
        & (trend_96.to_numpy(float) >= TREND_96_THRESHOLD)
        & (lr_impact.to_numpy(float) >= LR_IMPACT_LOWER)
        & (lr_impact.to_numpy(float) <= LR_IMPACT_UPPER)
    )
    premium_component = (
        (premium_available.to_numpy(float) > 0.5)
        & (premium_change.to_numpy(float) <= PREMIUM_CHANGE_THRESHOLD)
        & (daily["htf_1d_return_4"].to_numpy(float) >= DAILY_MOMENTUM_THRESHOLD)
    )
    admission = pd.DataFrame(
        {
            "btc_funding_rate": funding,
            "btc_premium_index_change": premium_change,
            "btc_trend_96": trend_96,
            "btc_daily_mom4": daily["htf_1d_return_4"],
            "btc_lr_impact_72": lr_impact,
        },
        index=market.index,
    ).replace([np.inf, -np.inf], np.nan)
    return funding_component, premium_component, admission


def _completed_hourly_oi(market: pd.DataFrame) -> pd.DataFrame:
    """Return OI features timestamped only when their source hour is complete."""
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(market["date"]),
            "log_oi": np.log(pd.to_numeric(market["sum_open_interest"], errors="coerce").where(lambda x: x > 0.0)),
            "source_time": pd.to_datetime(market["positioning_source_time"], errors="coerce"),
            "available": pd.to_numeric(market["oi_available"], errors="coerce").fillna(0.0),
        }
    )
    raw["source_hour"] = raw["date"].dt.floor("h")
    grouped = raw.groupby("source_hour", sort=True).agg(
        log_oi=("log_oi", "last"),
        source_time=("source_time", "last"),
        source_rows=("date", "size"),
        available_rows=("available", "sum"),
    )
    complete = (grouped["source_rows"] == 12) & (grouped["available_rows"] == 12)
    grouped["log_oi"] = grouped["log_oi"].where(complete)
    grouped["oi_logchg24"] = (grouped["log_oi"] - grouped["log_oi"].shift(24)).where(complete & complete.shift(24, fill_value=False))
    mean = grouped["log_oi"].rolling(168, min_periods=168).mean()
    std = grouped["log_oi"].rolling(168, min_periods=168).std(ddof=0).replace(0.0, np.nan)
    grouped["oi_z168"] = ((grouped["log_oi"] - mean) / std).where(
        grouped["log_oi"].rolling(168, min_periods=168).count() == 168
    )
    grouped["effective_time"] = grouped.index + pd.Timedelta("1h")
    return grouped.reset_index()


def _fit_state_thresholds(hourly: pd.DataFrame, *, min_observations: int) -> dict[str, float]:
    effective = pd.to_datetime(hourly["effective_time"])
    fit = (effective >= pd.Timestamp(FIT_WINDOW[0])) & (effective < pd.Timestamp(FIT_WINDOW[1]))
    change = pd.to_numeric(hourly.loc[fit, "oi_logchg24"], errors="coerce").dropna()
    level = pd.to_numeric(hourly.loc[fit, "oi_z168"], errors="coerce").dropna()
    if len(change) < min_observations or len(level) < min_observations:
        raise ValueError(f"insufficient fit OI states: change={len(change)}, level={len(level)}")
    return {
        "oi_logchg24_q30": float(change.quantile(0.30)),
        "oi_logchg24_q70": float(change.quantile(0.70)),
        "oi_z168_median": float(level.quantile(0.50)),
        "change_observations": int(len(change)),
        "level_observations": int(len(level)),
    }


def _build_transition_features(
    market: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    hourly = _completed_hourly_oi(market)
    change = pd.to_numeric(hourly["oi_logchg24"], errors="coerce")
    level = pd.to_numeric(hourly["oi_z168"], errors="coerce")
    valid = change.notna() & level.notna()
    state3 = pd.Series(np.nan, index=hourly.index, dtype=float)
    state3.loc[valid & (change <= thresholds["oi_logchg24_q30"])] = 0.0
    state3.loc[valid & (change > thresholds["oi_logchg24_q30"]) & (change < thresholds["oi_logchg24_q70"])] = 1.0
    state3.loc[valid & (change >= thresholds["oi_logchg24_q70"])] = 2.0
    level_bucket = (level > thresholds["oi_z168_median"]).astype(float).where(valid)
    state6 = (2.0 * state3 + level_bucket).where(valid)
    hourly["oi_state3"] = state3
    hourly["oi_state6"] = state6
    hourly["oi_transition3"] = (3.0 * state3.shift(1) + state3).where(state3.shift(1).notna() & state3.notna())
    hourly["oi_transition6"] = (6.0 * state6.shift(1) + state6).where(state6.shift(1).notna() & state6.notna())
    right = hourly[
        ["effective_time", "oi_logchg24", "oi_z168", "oi_state3", "oi_state6", "oi_transition3", "oi_transition6"]
    ].sort_values("effective_time")
    left = pd.DataFrame({"date": pd.to_datetime(market["date"]), "_row": np.arange(len(market))})
    merged = pd.merge_asof(
        left.sort_values("date"),
        right,
        left_on="date",
        right_on="effective_time",
        direction="backward",
        tolerance=pd.Timedelta("65min"),
    ).sort_values("_row")
    out = merged.drop(columns=["date", "_row", "effective_time"]).reset_index(drop=True)
    out["oi_transition_available"] = out[["oi_transition3", "oi_transition6"]].notna().all(axis=1).astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


def _correlation_audit(
    features: pd.DataFrame,
    base: pd.DataFrame,
    fit_mask: np.ndarray,
    *,
    max_abs_spearman: float,
) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for feature in ("oi_logchg24", "oi_z168"):
        values = pd.to_numeric(features[feature], errors="coerce")
        correlations: dict[str, float] = {}
        counts: dict[str, int] = {}
        for name in BASE_ADMISSION_FEATURES:
            paired = fit_mask & values.notna().to_numpy(bool) & base[name].notna().to_numpy(bool)
            counts[name] = int(paired.sum())
            correlations[name] = float(values.loc[paired].corr(base.loc[paired, name], method="spearman")) if counts[name] >= 100 else float("nan")
        max_abs = max((abs(value) for value in correlations.values() if np.isfinite(value)), default=float("inf"))
        audit[feature] = {
            "pair_counts": counts,
            "spearman": correlations,
            "max_abs_spearman": float(max_abs),
            "passes_independence": bool(max_abs < max_abs_spearman),
        }
    return audit


def _apply_policy(
    funding: np.ndarray,
    premium: np.ndarray,
    features: pd.DataFrame,
    spec: dict[str, Any],
) -> np.ndarray:
    values = pd.to_numeric(features[f"oi_transition{spec['schema_states']}"], errors="coerce").to_numpy(float)
    available = pd.to_numeric(features["oi_transition_available"], errors="coerce").fillna(0.0).to_numpy(float) > 0.5
    if spec["mode"].startswith("allow_"):
        gate = available & np.isin(values, np.asarray(spec["allowed_states"], dtype=float))
    elif spec["mode"] == "veto_bottom_negative":
        gate = available & ~np.isin(values, np.asarray(spec["vetoed_states"], dtype=float))
    else:
        raise ValueError(f"unknown OI transition mode: {spec['mode']}")
    if spec["target_component"] == "all":
        return (funding | premium) & gate
    if spec["target_component"] == "funding":
        return (funding & gate) | premium
    raise ValueError(f"unknown target component: {spec['target_component']}")


def _trade_returns(
    market: pd.DataFrame,
    dates: pd.Series,
    active: np.ndarray,
    cfg: OITransitionGateConfig,
    *,
    window: str,
) -> np.ndarray:
    start, end = WINDOWS[window]
    period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    opens = pd.to_numeric(market["open"], errors="coerce").to_numpy(float)
    candidates = np.arange(0, len(market) - HOLD_BARS - 2, STRIDE_BARS, dtype=np.int64)
    candidates = candidates[period[candidates] & active[candidates]]
    cost = (cfg.fee_rate + cfg.slippage_rate) * cfg.leverage
    next_position = 0
    returns: list[float] = []
    for position in candidates:
        if position < next_position:
            continue
        entry = position + 1
        exit_position = entry + HOLD_BARS
        if exit_position >= len(market) or not period[exit_position]:
            continue
        raw = opens[exit_position] / opens[entry] - 1.0
        returns.append((1.0 - cost) * (1.0 + cfg.leverage * raw) * (1.0 - cost) - 1.0)
        next_position = exit_position + 1
    return np.asarray(returns, dtype=float)


def _state_score_table(
    market: pd.DataFrame,
    dates: pd.Series,
    funding: np.ndarray,
    premium: np.ndarray,
    features: pd.DataFrame,
    cfg: OITransitionGateConfig,
    *,
    schema_states: int,
    target_component: str,
) -> list[dict[str, Any]]:
    transition = pd.to_numeric(features[f"oi_transition{schema_states}"], errors="coerce").to_numpy(float)
    available = pd.to_numeric(features["oi_transition_available"], errors="coerce").fillna(0.0).to_numpy(float) > 0.5
    target = (funding | premium) if target_component == "all" else funding
    rows: list[dict[str, Any]] = []
    for state in range(schema_states * schema_states):
        returns = _trade_returns(market, dates, target & available & (transition == state), cfg, window="fit")
        rows.append(
            {
                "state": state,
                "previous": state // schema_states,
                "current": state % schema_states,
                "trades": int(len(returns)),
                "mean_trade_return_pct": float(returns.mean() * 100.0) if len(returns) else 0.0,
                "median_trade_return_pct": float(np.median(returns) * 100.0) if len(returns) else 0.0,
            }
        )
    return rows


def _candidate_specs(
    market: pd.DataFrame,
    dates: pd.Series,
    funding: np.ndarray,
    premium: np.ndarray,
    features: pd.DataFrame,
    cfg: OITransitionGateConfig,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for schema_states in (3, 6):
        for target_component in ("funding", "all"):
            scores = _state_score_table(
                market,
                dates,
                funding,
                premium,
                features,
                cfg,
                schema_states=schema_states,
                target_component=target_component,
            )
            positive = sorted(
                (
                    row for row in scores
                    if row["trades"] >= cfg.min_transition_trades
                    and row["mean_trade_return_pct"] >= 0.20
                    and row["median_trade_return_pct"] > 0.0
                ),
                key=lambda row: (row["mean_trade_return_pct"], row["median_trade_return_pct"], row["trades"], -row["state"]),
                reverse=True,
            )
            persistent = [row for row in positive if row["previous"] == row["current"]]
            negative = sorted(
                (
                    row for row in scores
                    if row["trades"] >= cfg.min_transition_trades
                    and row["mean_trade_return_pct"] < 0.0
                    and row["median_trade_return_pct"] < 0.0
                ),
                key=lambda row: (row["mean_trade_return_pct"], row["median_trade_return_pct"], -row["trades"], row["state"]),
            )
            for k in (3, 5, 8):
                if len(positive) >= k:
                    specs.append(
                        {
                            "schema_states": schema_states,
                            "mode": "allow_top_positive",
                            "k": k,
                            "target_component": target_component,
                            "allowed_states": [row["state"] for row in positive[:k]],
                            "vetoed_states": [],
                            "fit_state_scores": scores,
                        }
                    )
                if len(persistent) >= k:
                    specs.append(
                        {
                            "schema_states": schema_states,
                            "mode": "allow_top_persistent_positive",
                            "k": k,
                            "target_component": target_component,
                            "allowed_states": [row["state"] for row in persistent[:k]],
                            "vetoed_states": [],
                            "fit_state_scores": scores,
                        }
                    )
                if len(negative) >= k:
                    specs.append(
                        {
                            "schema_states": schema_states,
                            "mode": "veto_bottom_negative",
                            "k": k,
                            "target_component": target_component,
                            "allowed_states": [],
                            "vetoed_states": [row["state"] for row in negative[:k]],
                            "fit_state_scores": scores,
                        }
                    )
    if len(specs) > 36:
        raise RuntimeError(f"OI transition search exceeded its 36-policy cap: {len(specs)}")
    return specs


def _activation_hash(active: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(np.asarray(active, dtype=bool)).tobytes()).hexdigest()


def _simulate(
    market: pd.DataFrame,
    dates: pd.Series,
    active: np.ndarray,
    cfg: OITransitionGateConfig,
    *,
    window: str,
    extremes: tuple[np.ndarray, np.ndarray],
    windows: dict[str, tuple[str, str]] = WINDOWS,
) -> dict[str, Any]:
    return _simulate_no_stop(
        market,
        dates,
        active,
        np.zeros(len(active), dtype=bool),
        window=window,
        hold_bars=HOLD_BARS,
        stride_bars=STRIDE_BARS,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        extremes=extremes,
        windows=windows,
    )


def _selection_score(stats: dict[str, dict[str, Any]], cfg: OITransitionGateConfig) -> float:
    fit, select = stats["fit"], stats["select_2023"]
    h1, h2 = stats["select_2023_h1"], stats["select_2023_h2"]
    if fit["trades"] < cfg.min_fit_trades or select["trades"] < cfg.min_select_trades:
        return -1e12
    if h1["trades"] < cfg.min_half_trades or h2["trades"] < cfg.min_half_trades:
        return -1e12
    if min(fit["cagr_pct"], select["cagr_pct"], h1["cagr_pct"], h2["cagr_pct"]) <= 0.0:
        return -1e12
    if fit["strict_mdd_pct"] > 25.0 or select["strict_mdd_pct"] > 15.0:
        return -1e12
    ratios = np.asarray([fit["ratio"], select["ratio"], h1["ratio"], h2["ratio"]], dtype=float)
    return float(np.min(ratios) + 0.30 * np.median(ratios) + min(0.25, select["trades"] / 200.0))


def _select_top(rows: list[dict[str, Any]], *, top_n: int, top_per_schema_mode: int) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            row["selection_score"],
            row["selection_stats"]["select_2023"]["ratio"],
            row["selection_stats"]["select_2023"]["return_pct"],
            -row["schema_states"],
            row["mode"],
            row["target_component"],
            -row["k"],
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for row in ordered:
        family = f"{row['schema_states']}:{row['mode']}"
        if counts.get(family, 0) >= top_per_schema_mode:
            continue
        selected.append(row)
        counts[family] = counts.get(family, 0) + 1
        if len(selected) >= top_n:
            break
    return selected


def _select_manifest(cfg: OITransitionGateConfig) -> dict[str, Any]:
    market, dates, source_prefix_hashes = _load_market(cfg, cutoff=SELECTION_END)
    hourly = _completed_hourly_oi(market)
    thresholds = _fit_state_thresholds(hourly, min_observations=cfg.min_fit_state_observations)
    features = _build_transition_features(market, thresholds)
    funding, premium, base_features = _build_base_components(market)
    fit_mask = _window_mask(dates, "fit")
    correlation_audit = _correlation_audit(
        features,
        base_features,
        fit_mask,
        max_abs_spearman=cfg.max_abs_spearman,
    )
    extremes = (
        _future_extreme(market["low"].to_numpy(float), HOLD_BARS, "min"),
        _future_extreme(market["high"].to_numpy(float), HOLD_BARS, "max"),
    )
    baseline_active = funding | premium
    baseline = {
        window: _simulate(market, dates, baseline_active, cfg, window=window, extremes=extremes)
        for window in ("fit", "select_2023", "select_2023_h1", "select_2023_h2")
    }
    specs = _candidate_specs(market, dates, funding, premium, features, cfg)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in specs:
        active = _apply_policy(funding, premium, features, spec)
        activation_hash = _activation_hash(active)
        if activation_hash in seen:
            continue
        seen.add(activation_hash)
        stats = {
            window: _simulate(market, dates, active, cfg, window=window, extremes=extremes)
            for window in ("fit", "select_2023", "select_2023_h1", "select_2023_h2")
        }
        score = _selection_score(stats, cfg)
        if score <= -1e11:
            continue
        rows.append({**spec, "activation_hash": activation_hash, "selection_score": score, "selection_stats": stats})
    selected = _select_top(rows, top_n=cfg.top_n, top_per_schema_mode=cfg.top_per_schema_mode)
    core = {
        "protocol": {
            "base_alpha": "fixed funding/premium squeeze with funding-only central lr_impact_72 gate",
            "state_fit": FIT_WINDOW,
            "selection": {name: WINDOWS[name] for name in ("select_2023", "select_2023_h1", "select_2023_h2")},
            "all_market_and_aux_rows_physically_excluded_before_manifest": True,
            "oi_source": "Binance USD-M BTCUSDT sum_open_interest delayed one complete 5m source bar",
            "hourly_state": "last delayed OI from a complete 12-row hour; exposed from next hour only",
            "state_formula": "3-state 24h OI change bucket; 6-state = 2*change_bucket + 7d OI-z level bucket; observable first-order transition",
            "search_cap": "36 policies; K in {3,5,8}; target in {funding,all}; fixed execution",
            "preflight_revision": "minimum policy trades reduced 60->36 fit and 16->12 select after the first pre-2024-only pass produced zero selectable policies; no OOS policy result existed",
            "entry": "next 5m open",
            "exit": f"fixed {HOLD_BARS} bars; stride {STRIDE_BARS}; no TP/SL",
            "cost": "6bp/side base and 10bp/side stress at 0.5x",
            "mdd": "strict favorable-high-water then adverse OHLC extreme",
            "status_ceiling": "shadow research; no retrospective live promotion",
        },
        "source_prefix_hashes": source_prefix_hashes,
        "state_thresholds": thresholds,
        "transition_feature_hash": _feature_hash(features, dates),
        "base_feature_hash": _feature_hash(base_features, dates),
        "correlation_audit": correlation_audit,
        "search_space": {
            "raw_specs": len(specs),
            "effective_unique_masks": len(seen),
            "eligible_variants": len(rows),
            "top_n": cfg.top_n,
            "top_per_schema_mode": cfg.top_per_schema_mode,
        },
        "baseline_selection_stats": baseline,
        "selected": selected,
    }
    manifest = {"as_of": datetime.now(timezone.utc).isoformat(), "sha256": _manifest_core_hash(core), **core}
    path = Path(cfg.manifest_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=True) + "\n")
    return manifest


def _replay(cfg: OITransitionGateConfig, manifest: dict[str, Any]) -> dict[str, Any]:
    _validate_manifest(manifest)
    _, _, prefix_hashes = _load_market(cfg, cutoff=SELECTION_END)
    if prefix_hashes != manifest["source_prefix_hashes"]:
        raise RuntimeError("pre-2024 source prefixes changed after manifest freeze")
    market, dates, _ = _load_market(cfg, cutoff=cfg.exclude_from)
    features = _build_transition_features(market, manifest["state_thresholds"])
    funding, premium, base_features = _build_base_components(market)
    prefix = dates < pd.Timestamp(SELECTION_END)
    prefix_dates = dates.loc[prefix].reset_index(drop=True)
    if _feature_hash(features.loc[prefix].reset_index(drop=True), prefix_dates) != manifest["transition_feature_hash"]:
        raise RuntimeError("pre-2024 OI transition feature prefix changed during replay")
    if _feature_hash(base_features.loc[prefix].reset_index(drop=True), prefix_dates) != manifest["base_feature_hash"]:
        raise RuntimeError("pre-2024 base feature prefix changed during replay")
    extremes = (
        _future_extreme(market["low"].to_numpy(float), HOLD_BARS, "min"),
        _future_extreme(market["high"].to_numpy(float), HOLD_BARS, "max"),
    )
    base_active = funding | premium
    baseline = {window: _simulate(market, dates, base_active, cfg, window=window, extremes=extremes) for window in WINDOWS}
    stress_cfg = replace(cfg, fee_rate=cfg.stress_fee_rate)
    selected: list[dict[str, Any]] = []
    for rank, frozen in enumerate(manifest["selected"], start=1):
        spec = {
            key: frozen[key]
            for key in ("schema_states", "mode", "k", "target_component", "allowed_states", "vetoed_states")
        }
        active = _apply_policy(funding, premium, features, spec)
        if _activation_hash(active[prefix.to_numpy(bool)]) != frozen["activation_hash"]:
            raise RuntimeError(f"pre-2024 activation drift at rank {rank}")
        stats = {window: _simulate(market, dates, active, cfg, window=window, extremes=extremes) for window in WINDOWS}
        for window in ("fit", "select_2023", "select_2023_h1", "select_2023_h2"):
            if stats[window] != frozen["selection_stats"][window]:
                raise RuntimeError(f"selection replay drift rank={rank} window={window}")
        stress = {
            window: _simulate(market, dates, active, stress_cfg, window=window, extremes=extremes)
            for window in ("test_2024", "eval_2025", "holdout_2026", "oos_2024_2026")
        }
        quarterly = {
            window: _simulate(market, dates, active, cfg, window=window, extremes=extremes, windows=QUARTER_WINDOWS)
            for window in QUARTER_WINDOWS
        }
        summary = {
            "positive_return_quarters": sum(row["return_pct"] > 0.0 for row in quarterly.values()),
            "flat_quarters": sum(row["trades"] == 0 for row in quarterly.values()),
            "negative_return_quarters": sum(row["return_pct"] < 0.0 for row in quarterly.values()),
            "total_quarters": len(quarterly),
        }
        test, evaluation = stats["test_2024"], stats["eval_2025"]
        holdout, combined = stats["holdout_2026"], stats["oos_2024_2026"]
        passes_alpha_pool = (
            test["ratio"] >= 2.5 and test["trades"] >= 20
            and evaluation["ratio"] >= 2.5 and evaluation["trades"] >= 20
            and holdout["return_pct"] > 0.0 and holdout["trades"] >= 12
            and combined["return_pct"] > 0.0
        )
        bonferroni = min(1.0, combined["p_value_mean_return_approx"] * max(1, len(manifest["selected"])))
        strong_shadow = (
            passes_alpha_pool
            and min(test["ratio"], evaluation["ratio"], holdout["ratio"], combined["ratio"]) >= 3.0
            and summary["positive_return_quarters"] >= 7
            and summary["negative_return_quarters"] <= 1
            and bonferroni <= 0.05
            and min(stress[name]["ratio"] for name in ("test_2024", "eval_2025", "holdout_2026")) >= 2.5
        )
        selected.append(
            {
                "manifest_rank": rank,
                **frozen,
                "stats": stats,
                "stress_10bp_each_side": stress,
                "quarterly_stats": quarterly,
                "quarterly_summary": summary,
                "top_n_bonferroni_p_value": float(bonferroni),
                "passes_alpha_pool": bool(passes_alpha_pool),
                "passes_strong_shadow": bool(strong_shadow),
                "passes_live_grade": False,
            }
        )
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "manifest": cfg.manifest_output,
        "manifest_sha256": manifest["sha256"],
        "protocol": manifest["protocol"],
        "source_file_hashes_after_manifest_freeze": _source_file_hashes(cfg),
        "state_thresholds": manifest["state_thresholds"],
        "correlation_audit": manifest["correlation_audit"],
        "baseline": baseline,
        "selected": selected,
        "alpha_pool_qualifiers": [row for row in selected if row["passes_alpha_pool"]],
        "strong_shadow": [row for row in selected if row["passes_strong_shadow"]],
        "live_grade": [],
    }


def run(cfg: OITransitionGateConfig) -> dict[str, Any]:
    manifest_path = Path(cfg.manifest_output)
    if manifest_path.exists() and not cfg.refresh_manifest:
        manifest = json.loads(manifest_path.read_text())
        _validate_manifest(manifest)
    else:
        manifest = _select_manifest(cfg)
    report = _replay(cfg, manifest)
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n")
    return report


def parse_args() -> OITransitionGateConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--funding-csv", required=True)
    parser.add_argument("--premium-csv", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude-from", default=OITransitionGateConfig.exclude_from)
    parser.add_argument("--top-n", type=int, default=OITransitionGateConfig.top_n)
    parser.add_argument("--top-per-schema-mode", type=int, default=OITransitionGateConfig.top_per_schema_mode)
    parser.add_argument("--max-abs-spearman", type=float, default=OITransitionGateConfig.max_abs_spearman)
    parser.add_argument("--refresh-manifest", action="store_true")
    return OITransitionGateConfig(**vars(parser.parse_args()))


def main() -> None:
    report = run(parse_args())
    print(
        json.dumps(
            {
                "manifest_sha256": report["manifest_sha256"],
                "selected": len(report["selected"]),
                "alpha_pool_qualifiers": len(report["alpha_pool_qualifiers"]),
                "strong_shadow": len(report["strong_shadow"]),
                "top": report["selected"][:3],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
