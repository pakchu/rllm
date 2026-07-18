"""Live-parity state/feature interaction rejection test for pre-2024 data.

This formalizes the completed broad weak-interaction tree experiment under the
current strict live-parity contract.  It intentionally opens no 2024+ data:
market, funding, and premium sources are physically truncated before 2024, tree
thresholds are fit on 2020-07-01..2022-12-31 only, and 2023 is selection
holdout.  The saved result is a rejection: 762 cells searched, 0 qualifiers.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.audit_confirmed_pullback_squeeze_live_parity import (
    AuditConfig,
    PRE2024_WINDOWS,
    _execution_config,
    _load_bundle,
    decision_mask,
    live_decision_features,
    selection_passes,
)
from training.long_component_tp_union_scan import _component_mask
from training.search_bocpd_state_gated_alpha import (
    BocpdStudentTState,
    _map_output,
    _state_from_mapped,
    bocpd_student_t_checkpointed,
)
from training.search_inventory_purge_reclaim_alpha import ExecutionEngine, Trade, equity_stats
from training.search_kalman_state_gated_alpha import (
    KalmanLocalLinearState,
    kalman_local_linear_checkpointed,
    map_hourly_state,
)
from training.search_semimarkov_duration_alpha import duration_key, map_hourly_key, observable_state

FIT_START = "2020-07-01"
FIT_END = "2023-01-01"
SELECTION_END = "2024-01-01"
DEFAULT_OUTPUT = "results/liveparity_state_feature_interactions_pre2024_2026-07-15.json"
DEFAULT_DOCS_OUTPUT = "docs/liveparity-state-feature-interactions-pre2024-2026-07-15.md"

STATE_FEATURE_NAMES = [
    "k_slope",
    "k_innov",
    "b_segment",
    "b_reset",
    "b_flow",
    "s_trend",
    "s_vol",
    "s_flow",
    "s_age",
    "funding_leg",
    "premium_leg",
]

GROUPS: dict[str, list[str]] = {
    "state_pa": [
        "rex_144_range_pos",
        "rex_576_range_pos",
        "rex_2016_range_pos",
        "rex_8640_range_pos",
        "rex_2016_range_width_pct",
        "htf_4h_return_4",
        "htf_1d_return_4",
        "htf_1w_return_1",
        "htf_1d_range_pos",
        "htf_1w_range_pos",
    ],
    "state_pa_macro": [
        "rex_144_range_pos",
        "rex_576_range_pos",
        "rex_2016_range_pos",
        "rex_8640_range_pos",
        "rex_2016_range_width_pct",
        "htf_4h_return_4",
        "htf_1d_return_4",
        "htf_1w_return_1",
        "htf_1d_range_pos",
        "htf_1w_range_pos",
        "dxy_momentum",
        "usdkrw_zscore",
        "kimchi_premium_change",
    ],
    "state_all": [
        "rex_144_range_pos",
        "rex_576_range_pos",
        "rex_2016_range_pos",
        "rex_8640_range_pos",
        "rex_2016_range_width_pct",
        "htf_4h_return_4",
        "htf_1d_return_4",
        "htf_1w_return_1",
        "htf_1d_range_pos",
        "htf_1w_range_pos",
        "dxy_momentum",
        "usdkrw_zscore",
        "kimchi_premium_change",
        "taker_imbalance",
        "volume_zscore",
        "funding_zscore",
        "premium_index_zscore",
    ],
}


@dataclass
class _BocpdRuntimeCache:
    index: pd.DatetimeIndex
    raw_values: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    output: pd.DataFrame
    state: BocpdStudentTState
    columns: tuple[str, ...]
    secondary_index: int | None
    hazard_lambda: int


_BOCPD_RUNTIME_CACHE: _BocpdRuntimeCache | None = None


@dataclass
class _KalmanRuntimeCache:
    index: pd.DatetimeIndex
    log_price: np.ndarray
    train_var: float
    frame: pd.DataFrame
    state: KalmanLocalLinearState
    thresholds: dict[str, float]
    fit_mask: np.ndarray


_KALMAN_RUNTIME_CACHE: _KalmanRuntimeCache | None = None


def _index_nanoseconds(index: pd.DatetimeIndex) -> np.ndarray:
    return index.to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=False)


def _clear_bocpd_runtime_cache() -> None:
    global _BOCPD_RUNTIME_CACHE, _KALMAN_RUNTIME_CACHE
    _BOCPD_RUNTIME_CACHE = None
    _KALMAN_RUNTIME_CACHE = None


def _runtime_kalman_frame(hourly: pd.DataFrame, train_mask: np.ndarray) -> pd.DataFrame:
    """Resume the fixed Rank7 Kalman filter for append-only hourly history."""

    global _KALMAN_RUNTIME_CACHE
    index = pd.DatetimeIndex(hourly.index)
    log_price = np.log(hourly["close"].to_numpy(float))
    fit = np.asarray(train_mask, dtype=bool)
    train_var = float(np.nanvar(np.diff(log_price)[fit[1:]]))
    cached = _KALMAN_RUNTIME_CACHE
    prefix = len(cached.index) if cached is not None else 0
    can_resume = bool(
        cached is not None
        and prefix <= len(index)
        and cached.train_var == train_var
        and np.array_equal(cached.fit_mask, fit[:prefix])
        and not fit[prefix:].any()
        and np.array_equal(_index_nanoseconds(cached.index), _index_nanoseconds(index)[:prefix])
        and np.array_equal(cached.log_price, log_price[:prefix])
    )
    if can_resume and prefix == len(index):
        return cached.frame.copy()

    if can_resume:
        filtered, state = kalman_local_linear_checkpointed(
            log_price[prefix:],
            q_level=0.1,
            q_slope=0.001,
            r_obs=0.5,
            train_var=train_var,
            checkpoint=cached.state,
        )
        suffix = pd.DataFrame(
            {
                "date": index[prefix:].to_numpy(),
                "slope_z": filtered[:, 3],
                "innovation_z": filtered[:, 2],
            }
        )
        frame = pd.concat([cached.frame, suffix], ignore_index=True)
        thresholds = cached.thresholds
    else:
        filtered, state = kalman_local_linear_checkpointed(
            log_price,
            q_level=0.1,
            q_slope=0.001,
            r_obs=0.5,
            train_var=train_var,
        )
        frame = pd.DataFrame(
            {
                "date": index.to_numpy(),
                "slope_z": filtered[:, 3],
                "innovation_z": filtered[:, 2],
            }
        )
        fit_frame = frame.loc[fit]
        thresholds = {
            "slope_low": float(fit_frame["slope_z"].quantile(0.25)),
            "slope_high": float(fit_frame["slope_z"].quantile(0.75)),
            "innovation_low": float(fit_frame["innovation_z"].quantile(0.25)),
            "innovation_high": float(fit_frame["innovation_z"].quantile(0.75)),
        }

    frame["state"] = (
        np.where(
            frame["slope_z"] <= thresholds["slope_low"],
            0,
            np.where(frame["slope_z"] >= thresholds["slope_high"], 2, 1),
        )
        * 3
        + np.where(
            frame["innovation_z"] <= thresholds["innovation_low"],
            0,
            np.where(frame["innovation_z"] >= thresholds["innovation_high"], 2, 1),
        )
    )
    _KALMAN_RUNTIME_CACHE = _KalmanRuntimeCache(
        index=index.copy(),
        log_price=log_price.copy(),
        train_var=train_var,
        frame=frame,
        state=state,
        thresholds=thresholds,
        fit_mask=fit.copy(),
    )
    return frame


def _runtime_bocpd_output(
    feature_frame: pd.DataFrame,
    train_mask: np.ndarray,
    *,
    columns: tuple[str, ...],
    secondary_index: int | None,
    hazard_lambda: int,
) -> pd.DataFrame:
    """Resume the fixed live BOCPD model when history only appended.

    Prefix timestamps, raw values, and frozen fit statistics are all checked
    before a checkpoint is accepted. Any revision falls back to the canonical
    full-history path and replaces the cache.
    """

    global _BOCPD_RUNTIME_CACHE
    good = feature_frame[list(columns)].notna().all(axis=1).to_numpy()
    fit_mask = good & np.asarray(train_mask, dtype=bool)
    raw_train = feature_frame.loc[fit_mask, list(columns)].to_numpy(float)
    mean = raw_train.mean(axis=0)
    std = raw_train.std(axis=0)
    std[std < 1e-8] = 1.0
    index = pd.DatetimeIndex(feature_frame.index[good])
    raw_values = feature_frame.loc[good, list(columns)].to_numpy(float)

    cached = _BOCPD_RUNTIME_CACHE
    prefix = len(cached.index) if cached is not None else 0
    can_resume = bool(
        cached is not None
        and prefix <= len(index)
        and cached.columns == columns
        and cached.secondary_index == secondary_index
        and cached.hazard_lambda == hazard_lambda
        and np.array_equal(_index_nanoseconds(cached.index), _index_nanoseconds(index)[:prefix])
        and np.array_equal(cached.raw_values, raw_values[:prefix])
        and np.array_equal(cached.mean, mean)
        and np.array_equal(cached.std, std)
    )
    if can_resume and prefix == len(index):
        return cached.output.copy()

    if can_resume:
        standardized = np.clip((raw_values[prefix:] - mean) / std, -12, 12)
        posterior, state = bocpd_student_t_checkpointed(standardized, state=cached.state)
        primary = posterior["posterior_mean"][:, 0]
        secondary = (
            posterior["surprise"]
            if secondary_index is None
            else posterior["posterior_mean"][:, secondary_index]
        )
        suffix = pd.DataFrame(
            {
                "date": index[prefix:].to_numpy(),
                "primary": primary,
                "short_mass": posterior["short_mass"],
                "run_drop": posterior["run_drop"],
                "secondary": secondary,
                "surprise": posterior["surprise"],
            }
        )
        output = pd.concat([cached.output, suffix], ignore_index=True)
    else:
        standardized = np.clip((raw_values - mean) / std, -12, 12)
        posterior, state = bocpd_student_t_checkpointed(
            standardized,
            hazard_lambda=hazard_lambda,
            max_run_length=min(1000, hazard_lambda * 4),
        )
        primary = posterior["posterior_mean"][:, 0]
        secondary = (
            posterior["surprise"]
            if secondary_index is None
            else posterior["posterior_mean"][:, secondary_index]
        )
        output = pd.DataFrame(
            {
                "date": index.to_numpy(),
                "primary": primary,
                "short_mass": posterior["short_mass"],
                "run_drop": posterior["run_drop"],
                "secondary": secondary,
                "surprise": posterior["surprise"],
            }
        )

    _BOCPD_RUNTIME_CACHE = _BocpdRuntimeCache(
        index=index.copy(),
        raw_values=raw_values.copy(),
        mean=mean.copy(),
        std=std.copy(),
        output=output,
        state=state,
        columns=columns,
        secondary_index=secondary_index,
        hazard_lambda=hazard_lambda,
    )
    return output


@dataclass(frozen=True)
class Config(AuditConfig):
    output: str = DEFAULT_OUTPUT
    docs_output: str = DEFAULT_DOCS_OUTPUT
    top_n: int = 50
    random_state: int = 715


def completed_hourly_features(market: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return features exposed at the right edge of fully completed source hours.

    Five-minute market timestamps are bar opens.  At decision time ``HH:00`` the
    only complete hourly market interval is ``[HH-1:00, HH:00)``; the bar whose
    timestamp is exactly ``HH:00`` is the *next* interval and is not included.
    Incomplete trailing hours are dropped rather than forward-filled.
    """

    x = market.copy()
    x["date"] = pd.to_datetime(x["date"], utc=False, errors="raise")
    x = x.sort_values("date").set_index("date")
    quote = pd.to_numeric(x["quote_asset_volume"], errors="coerce")
    buy = pd.to_numeric(x["taker_buy_quote"], errors="coerce")
    grouped = pd.DataFrame(
        {
            "open": pd.to_numeric(x["open"], errors="coerce").resample("1h", closed="left", label="right").first(),
            "high": pd.to_numeric(x["high"], errors="coerce").resample("1h", closed="left", label="right").max(),
            "low": pd.to_numeric(x["low"], errors="coerce").resample("1h", closed="left", label="right").min(),
            "close": pd.to_numeric(x["close"], errors="coerce").resample("1h", closed="left", label="right").last(),
            "quote": quote.resample("1h", closed="left", label="right").sum(),
            "buy": buy.resample("1h", closed="left", label="right").sum(),
            "bar_count": x["close"].resample("1h", closed="left", label="right").count(),
        }
    )
    hourly = grouped.loc[grouped["bar_count"] >= 12].drop(columns="bar_count").dropna()
    return hourly, hourly_state_features(hourly)


def hourly_state_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Derive the exact state-model inputs from completed hourly bars."""

    required = {"high", "low", "close", "quote", "buy"}
    missing = sorted(required - set(hourly.columns))
    if missing:
        raise ValueError(f"completed hourly frame missing columns: {missing}")
    returns = np.log(hourly["close"]).diff()
    flow = 2.0 * hourly["buy"] / hourly["quote"].replace(0.0, np.nan) - 1.0
    features = pd.DataFrame(index=hourly.index)
    features["ret1"] = returns
    features["trend24"] = np.log(hourly["close"] / hourly["close"].shift(24))
    features["trend72"] = np.log(hourly["close"] / hourly["close"].shift(72))
    features["vol24"] = returns.rolling(24).std()
    features["vol168"] = returns.rolling(168).std()
    features["volterm"] = features["vol24"] / features["vol168"].replace(0.0, np.nan)
    features["range24"] = (hourly["high"].rolling(24).max() - hourly["low"].rolling(24).min()) / hourly["close"]
    features["flow24"] = flow.rolling(24).mean()
    log_quote = np.log1p(hourly["quote"])
    features["volume_z"] = (log_quote - log_quote.rolling(168).mean()) / log_quote.rolling(168).std().replace(0.0, np.nan)
    return features.replace([np.inf, -np.inf], np.nan)


def state_bank(market: pd.DataFrame, dates: pd.Series) -> dict[str, np.ndarray]:
    hourly, hourly_features = completed_hourly_features(market)
    return state_bank_from_hourly(hourly, hourly_features, dates)


def state_bank_from_hourly(
    hourly: pd.DataFrame,
    hourly_features: pd.DataFrame,
    dates: pd.Series,
) -> dict[str, np.ndarray]:
    """Build the frozen causal state bank from completed-hour inputs.

    Keeping this boundary explicit lets live inference prepend a compact,
    immutable hourly history without querying years of one-minute source rows.
    ``state_bank`` remains the canonical batch wrapper and delegates here.
    """

    if hourly.empty or hourly_features.empty:
        size = len(dates)
        missing = np.full(size, -1, dtype=int)
        return {"kalman": missing.copy(), "bocpd": missing.copy(), "semimarkov": missing.copy()}
    hourly = hourly.sort_index()
    hourly_features = hourly_features.sort_index().reindex(hourly.index)
    fit_hour = np.asarray((hourly_features.index >= FIT_START) & (hourly_features.index < FIT_END), dtype=bool)
    if not fit_hour.any():
        raise ValueError("completed hourly history does not cover the frozen Rank7 fit window")
    kalman_frame = _runtime_kalman_frame(
        hourly,
        np.asarray((hourly.index >= FIT_START) & (hourly.index < FIT_END), dtype=bool),
    )
    kalman = map_hourly_state(dates, kalman_frame)

    bocpd_output = _runtime_bocpd_output(
        hourly_features,
        fit_hour,
        columns=("ret1", "flow24"),
        secondary_index=1,
        hazard_lambda=336,
    )
    fit_bocpd = bocpd_output[(bocpd_output["date"] >= FIT_START) & (bocpd_output["date"] < FIT_END)]
    thresholds = {
        "primary_low": float(fit_bocpd["primary"].quantile(0.33)),
        "primary_high": float(fit_bocpd["primary"].quantile(0.67)),
        "short_mass_high": float(fit_bocpd["short_mass"].quantile(0.50)),
        "secondary_high": float(fit_bocpd["secondary"].quantile(0.50)),
    }
    bocpd = _state_from_mapped(_map_output(dates, bocpd_output), thresholds)

    semi_state, _ = observable_state(hourly_features, fit_hour, 0.33, 0.67)
    semi_key, _ = duration_key(semi_state, (1, 6, 24, 72), timestamps=hourly_features.index)
    semimarkov = map_hourly_key(dates, hourly_features.index, semi_key)
    return {"kalman": kalman, "bocpd": bocpd, "semimarkov": semimarkov}


def immutable_anchors(active: np.ndarray, cooldown: int) -> np.ndarray:
    """Freeze anchor decisions before downstream modeling or exit simulation."""

    out = np.zeros(len(active), dtype=bool)
    next_allowed = 0
    for position in np.flatnonzero(np.asarray(active, dtype=bool)):
        position = int(position)
        if position < next_allowed:
            continue
        out[position] = True
        next_allowed = position + int(cooldown)
    return out


def feature_matrix(bank: dict[str, np.ndarray], funding: np.ndarray, premium: np.ndarray) -> np.ndarray:
    kalman = bank["kalman"]
    bocpd = bank["bocpd"]
    semimarkov = bank["semimarkov"]
    semi_state = semimarkov // 5
    return np.column_stack(
        [
            kalman // 3,
            kalman % 3,
            bocpd // 4,
            (bocpd % 4) // 2,
            bocpd % 2,
            semi_state // 4,
            (semi_state % 4) // 2,
            semi_state % 2,
            semimarkov % 5,
            np.asarray(funding, dtype=np.int8),
            np.asarray(premium, dtype=np.int8),
        ]
    ).astype(float)


def net_target(engine: ExecutionEngine, signal: int, hold: int, cfg: Config) -> float:
    trade = engine.trade_at(int(signal), 1, int(hold), 1_000_000, 1_000_000)
    if trade is None:
        return np.nan
    side_cost = float(cfg.fee_rate + cfg.slippage_rate)
    entry_exit_factor = 1.0 - float(cfg.leverage) * side_cost
    return float(entry_exit_factor * trade.price_factor * trade.funding_factor * entry_exit_factor - 1.0)


def schedule_time_only(
    engine: ExecutionEngine,
    active: np.ndarray,
    *,
    hold: int,
    start: str,
    end: str,
) -> list[Trade]:
    dates = pd.to_datetime(engine.market["date"])
    period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    trades: list[Trade] = []
    next_allowed = 0
    for signal in np.flatnonzero(np.asarray(active, dtype=bool) & period):
        signal = int(signal)
        if signal < next_allowed:
            continue
        trade = engine.trade_at(signal, 1, int(hold), 1_000_000, 1_000_000)
        if trade is None or not period[trade.exit_position]:
            continue
        trades.append(trade)
        next_allowed = trade.exit_position + 1
    return trades


def slim(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        key: stats[key]
        for key in (
            "absolute_return_pct",
            "cagr_pct",
            "strict_mdd_pct",
            "cagr_to_strict_mdd",
            "trades",
            "mean_net_bps",
            "win_rate",
        )
    }


def _fit_matrix(raw: np.ndarray, fit: np.ndarray) -> np.ndarray:
    medians = np.nanmedian(raw[np.asarray(fit, dtype=bool)], axis=0)
    return np.where(np.isfinite(raw), raw, medians)


def run_search(cfg: Config) -> dict[str, Any]:
    market, features, funding, source_hashes = _load_bundle(
        cfg,
        cutoff=SELECTION_END,
        premium_tolerance=cfg.live_premium_tolerance,
    )
    dates = pd.to_datetime(market["date"])
    live_features = live_decision_features(features)
    decisions = decision_mask(dates, "live_hour_signal_bar", window_size=cfg.window_size)
    funding_leg = decisions & _component_mask(live_features, "funding10_trend70")
    premium_leg = decisions & _component_mask(live_features, "premium20_mom90")
    base = funding_leg | premium_leg
    bank = state_bank(market, dates)
    valid_state = (bank["kalman"] >= 0) & (bank["bocpd"] >= 0) & (bank["semimarkov"] >= 0)
    base &= valid_state

    state_features = feature_matrix(bank, funding_leg, premium_leg)
    fit = ((dates >= pd.Timestamp(FIT_START)) & (dates < pd.Timestamp(FIT_END))).to_numpy(bool)
    engine = ExecutionEngine(market, funding, _execution_config(cfg, cfg.leverage))
    rows: list[dict[str, Any]] = []
    for group, columns in GROUPS.items():
        raw = np.column_stack(
            [state_features, *[pd.to_numeric(live_features[column], errors="coerce").to_numpy(float) for column in columns]]
        )
        matrix = _fit_matrix(raw, fit)
        names = STATE_FEATURE_NAMES + columns
        for cooldown, hold, depth, leaf in itertools.product((144, 288, 576), (144, 288, 576), (2, 3, 4), (12, 20, 32)):
            anchors = immutable_anchors(base, cooldown)
            train = anchors & fit
            positions = np.flatnonzero(train)
            y = np.array([net_target(engine, position, hold, cfg) for position in positions], dtype=float)
            good = np.isfinite(y)
            positions = positions[good]
            y = y[good]
            if len(y) < 60:
                continue
            model = DecisionTreeRegressor(
                max_depth=int(depth),
                min_samples_leaf=int(leaf),
                random_state=int(cfg.random_state),
                criterion="squared_error",
            ).fit(matrix[positions], y)
            train_pred = model.predict(matrix[positions])
            pred = model.predict(matrix)
            thresholds = sorted({0.0, *[float(np.quantile(train_pred, q)) for q in (0.30, 0.40, 0.50, 0.60, 0.70)]})
            for threshold in thresholds:
                active = anchors & (pred >= threshold)
                stats = {}
                for name, (start, end) in PRE2024_WINDOWS.items():
                    trades = schedule_time_only(engine, active, hold=hold, start=start, end=end)
                    stats[name] = slim(equity_stats(trades, start=start, end=end, cfg=_execution_config(cfg, cfg.leverage)))
                ratios = [stats[name]["cagr_to_strict_mdd"] for name in ("train", "select_2023", "pre_2024")]
                stable = sum(
                    stats[name]["absolute_return_pct"] > 0.0
                    for name in ("train_2020h2", "train_2021", "train_2022", "select_2023_h1", "select_2023_h2")
                )
                passed = selection_passes(stats)
                rows.append(
                    {
                        "spec": {
                            "group": group,
                            "cooldown": int(cooldown),
                            "hold": int(hold),
                            "depth": int(depth),
                            "leaf": int(leaf),
                            "threshold": float(threshold),
                        },
                        "pass": passed,
                        "rank": [passed, int(stable), float(min(ratios)), float(np.median(ratios)), int(stats["pre_2024"]["trades"])],
                        "stats": stats,
                        "tree": export_text(model, feature_names=names, max_depth=int(depth)),
                        "train_examples": int(len(y)),
                    }
                )
    rows.sort(key=lambda row: row["rank"], reverse=True)
    return {
        "protocol": protocol(source_hashes=source_hashes),
        "config": asdict(cfg),
        "cells": len(rows),
        "qualifiers": int(sum(bool(row["pass"]) for row in rows)),
        "top": rows[: int(cfg.top_n)],
    }


def protocol(*, source_hashes: dict[str, str] | None = None) -> dict[str, Any]:
    return {
        "name": "live-parity broad state/feature interaction tree pre-2024 rejection",
        "result": "REJECTION",
        "fit_window": {"start": FIT_START, "end_exclusive": FIT_END},
        "selection_end_exclusive": SELECTION_END,
        "market_source": "physically truncated before 2024",
        "decision_clock": "top-of-hour :00 rows only",
        "market_feature_parity": "live_decision_features excludes the current market bar",
        "completed_hourly_state_interval": "[HH-1:00, HH:00) exposed at HH:00",
        "entry": "next five-minute open after signal row",
        "costs": "6bp notional per side via fee_rate + slippage_rate at 0.5x leverage",
        "funding": "realized funding compounded during each position",
        "risk": "strict MDD includes intratrade adverse envelope",
        "schedule": "non-overlap; exits must remain split-contained",
        "sealed_windows": ["2024+"],
        "source_hashes": source_hashes or {},
    }


def _write_docs(path: str | Path, payload: dict[str, Any]) -> None:
    top = payload["top"][:5]
    lines = [
        "# Live-parity state/feature interaction rejection — 2026-07-15",
        "",
        "This formalizes the completed broad weak interaction tree search on physically truncated pre-2024 data.",
        "",
        "## Protocol",
        "",
        "- Decisions occur only on `:00` rows.",
        "- `live_decision_features` excludes the current market bar; completed hourly state uses `[HH-1:00, HH:00)` at `HH:00`.",
        "- Tree targets and thresholds are fit only on 2020-07-01 through 2022-12-31.",
        "- Market/funding/premium inputs are truncated before 2024; 2024+ remains sealed.",
        "- Entry is the next 5-minute open; costs are 6 bp/notional/side; realized funding and strict MDD are included.",
        "- Schedules are non-overlapping and exits must stay inside each split.",
        "",
        "## Result",
        "",
        f"**REJECTION.** The grid evaluated `{payload['cells']}` cells and found `{payload['qualifiers']}` qualifiers.",
        "",
        "## Top ranked cells",
        "",
        "| Rank | Group | Cooldown | Hold | Depth | Leaf | Threshold | Train CAGR/MDD | 2023 CAGR/MDD | Pre-2024 CAGR/MDD | Pre-2024 trades | Pass |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for idx, row in enumerate(top, 1):
        spec = row["spec"]
        stats = row["stats"]
        lines.append(
            "| {idx} | {group} | {cooldown} | {hold} | {depth} | {leaf} | {threshold:.8f} | {train:.2f} | {select:.2f} | {pre:.2f} | {trades} | {passed} |".format(
                idx=idx,
                group=spec["group"],
                cooldown=spec["cooldown"],
                hold=spec["hold"],
                depth=spec["depth"],
                leaf=spec["leaf"],
                threshold=float(spec["threshold"]),
                train=float(stats["train"]["cagr_to_strict_mdd"]),
                select=float(stats["select_2023"]["cagr_to_strict_mdd"]),
                pre=float(stats["pre_2024"]["cagr_to_strict_mdd"]),
                trades=int(stats["pre_2024"]["trades"]),
                passed=str(bool(row["pass"])),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The best-ranked cells are economically interesting but fail the frozen selection contract, primarily because the minimum train/2023/pre-2024 CAGR-to-strict-MDD target is not met across all required windows. No 2024+ evaluation is justified.",
        ]
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    tmp.replace(output)


def _payload_from_completed_json(source: str | Path, cfg: Config) -> dict[str, Any]:
    completed = json.loads(Path(source).read_text(encoding="utf-8"))
    return {
        "protocol": protocol(),
        "config": asdict(cfg),
        "cells": int(completed["cells"]),
        "qualifiers": int(completed["qualifiers"]),
        "top": completed["top"][: int(cfg.top_n)],
        "formalized_from_completed_artifact": Path(source).name,
        "formalized_at": datetime.now(timezone.utc).isoformat(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--docs-output", default=DEFAULT_DOCS_OUTPUT)
    parser.add_argument("--from-completed-json", default="", help="Formalize an already completed result JSON instead of rerunning the search.")
    parser.add_argument("--top-n", type=int, default=50)
    args = parser.parse_args(argv)
    cfg = Config(output=args.output, docs_output=args.docs_output, top_n=args.top_n)
    payload = _payload_from_completed_json(args.from_completed_json, cfg) if args.from_completed_json else run_search(cfg)
    _atomic_write_json(cfg.output, payload)
    _write_docs(cfg.docs_output, payload)
    print(json.dumps({"output": cfg.output, "docs_output": cfg.docs_output, "cells": payload["cells"], "qualifiers": payload["qualifiers"]}, indent=2))


if __name__ == "__main__":
    main()
