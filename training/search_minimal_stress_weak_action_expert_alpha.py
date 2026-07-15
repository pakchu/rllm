"""Freeze a source-aware weak-signal action expert on minimal stress events.

The entry event is deliberately broader than the confirmed pullback family:
either a train-fitted funding-relief event or a train-fitted premium-discount
event may activate it.  A year-balanced ridge expert then combines prior-bar
weak features, the event source, and completed-hour BOCPD diagnostics to choose
ABSTAIN or a LONG/SHORT TP4/TP8/TP12/time action.

Selection sources are physically truncated before 2024.  Feature scaling,
BOCPD normalization, and counterfactual utility labels use only 2020-07-01 to
2022-12-29 labels whose full 48-hour path ends before 2023.  A one-shot manifest
must be committed before 2024+ is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.audit_causal_bocpd_pullback_overlay import (
    completed_hour_features,
    exact_hour_map,
)
from training.audit_confirmed_pullback_squeeze_live_parity import (
    PRE2024_WINDOWS,
    _activation_hash,
    _execution_config,
    _load_bundle,
    decision_mask,
    live_decision_features,
)
from training.search_bocpd_state_gated_alpha import _model_output
from training.search_causal_bocpd_pullback_exit_router_alpha import (
    Config as RouterConfig,
    _atomic_write_json,
    _frozen_execution_config,
    _slim,
    trade_utility,
)
from training.search_causal_weak_tensor_exit_router_alpha import (
    BOCPD_COLUMNS,
    _array_hash,
    tensor_design,
)
from training.search_funding_premium_external_state_gate_alpha import _frame_hash
from training.search_inventory_purge_reclaim_alpha import (
    ExecutionEngine,
    Trade,
    _schedule_hash,
    equity_stats,
)
from training.search_pullback_premium_overheat_state_machine_alpha import (
    FUTURE_WINDOWS,
    NO_STOP_BPS,
    SELECTION_END,
    _selection_score,
    oos_passes,
    selection_passes,
)
from training.search_specific_pullback_squeeze_alpha import (
    FIT_END,
    FIT_START,
    fit_rule_masks,
)


WEAK_FEATURES = [
    "htf_1w_return_4",
    "rex_2016_range_pos",
    "htf_3d_return_1",
    "dxy_momentum",
    "usdkrw_momentum",
    "kimchi_premium_change",
    "taker_imbalance",
    "dollar_flow_rel_4h_30d",
    "quote_vol_z_1d",
]
EVENT_SOURCE_FEATURE = "event_source_signed"
MODEL_FORMS = ("linear", "tensor", "tensor_pairs")
RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
RISK_GRID = (0.0, 0.25, 0.5)
HAZARD_GRID = (168, 336)
WEIGHT_MODES = ("uniform", "year", "year_source")
TIME_TAKE_BPS = NO_STOP_BPS
HOLD_BARS = 576
ACTION_SETS: dict[str, tuple[tuple[int, int], ...]] = {
    "tp4_time": (
        (1, 400),
        (1, TIME_TAKE_BPS),
        (-1, 400),
        (-1, TIME_TAKE_BPS),
    ),
    "tp4_tp8_time": (
        (1, 400),
        (1, 800),
        (1, TIME_TAKE_BPS),
        (-1, 400),
        (-1, 800),
        (-1, TIME_TAKE_BPS),
    ),
    "tp4_tp8_tp12_time": (
        (1, 400),
        (1, 800),
        (1, 1200),
        (1, TIME_TAKE_BPS),
        (-1, 400),
        (-1, 800),
        (-1, 1200),
        (-1, TIME_TAKE_BPS),
    ),
}

EXPERT_SPEC: dict[str, Any] = {
    "name": "minimal_stress_weak_action_expert",
    "entry": "funding relief OR premium discount after train-only quantiles",
    "actions": {
        name: [list(action) for action in actions]
        for name, actions in ACTION_SETS.items()
    },
    "hold_bars": HOLD_BARS,
    "stop_bps": NO_STOP_BPS,
    "weak_features": WEAK_FEATURES,
    "event_source_feature": EVENT_SOURCE_FEATURE,
    "bocpd_columns": BOCPD_COLUMNS,
    "hazard_hours": list(HAZARD_GRID),
    "forms": list(MODEL_FORMS),
    "ridge": list(RIDGE_GRID),
    "risk_lambda": list(RISK_GRID),
    "weight_modes": list(WEIGHT_MODES),
    "target": "absolute risk-adjusted utility for each executable action; neutral utility is zero",
    "label_purge": "48-hour counterfactual exit must precede 2023-01-01",
    "grid_cells": len(HAZARD_GRID)
    * len(MODEL_FORMS)
    * len(RIDGE_GRID)
    * len(RISK_GRID)
    * len(WEIGHT_MODES)
    * len(ACTION_SETS),
    "selection_rule": "absolute selection gate and lexicographically beat long time-only base",
}

FROZEN_CHAMPION: dict[str, Any] = {
    "features": WEAK_FEATURES + [EVENT_SOURCE_FEATURE],
    "hazard_hours": 168,
    "form": "tensor_pairs",
    "ridge": 10.0,
    "risk_lambda": 0.5,
    "weight_mode": "year",
    "action_set": "tp4_tp8_tp12_time",
    "actions": [list(action) for action in ACTION_SETS["tp4_tp8_tp12_time"]],
}


@dataclass(frozen=True)
class Config(RouterConfig):
    output: str = "results/minimal_stress_weak_action_expert_selection_2026-07-15.json"
    manifest_output: str = (
        "results/minimal_stress_weak_action_expert_manifest_2026-07-15.json"
    )
    docs_output: str = (
        "docs/minimal-stress-weak-action-expert-selection-2026-07-15.md"
    )
    open_oos: bool = False


def _spec_hash() -> str:
    encoded = json.dumps(
        EXPERT_SPEC, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def minimal_stress_events(
    features: pd.DataFrame,
    dates: pd.Series,
    decisions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Return funding, premium, and union events using train-only thresholds."""

    fitted = fit_rule_masks(features, dates, decisions)
    thresholds = fitted["base_thresholds"]

    def values(column: str) -> np.ndarray:
        return pd.to_numeric(features[column], errors="coerce").to_numpy(float)

    funding = (
        decisions
        & (values("funding_available") > 0.5)
        & (values("funding_rate") <= float(thresholds["funding_rate_q10"]))
        & (values("trend_96") >= float(thresholds["trend_96_q70"]))
    )
    premium = (
        decisions
        & (values("premium_available") > 0.5)
        & (
            values("premium_index_change")
            <= float(thresholds["premium_change_q20"])
        )
        & (
            values("htf_1d_return_4")
            >= float(thresholds["daily_momentum_4_q90"])
        )
    )
    return funding, premium, funding | premium, thresholds


def _context(cfg: Config, *, cutoff: str) -> tuple[dict[str, Any], dict[str, str]]:
    market, raw_features, funding, source_hashes = _load_bundle(
        cfg, cutoff=cutoff, premium_tolerance=cfg.live_premium_tolerance
    )
    dates = pd.to_datetime(market["date"])
    decisions = decision_mask(
        dates, "live_hour_signal_bar", window_size=cfg.window_size
    )
    features = live_decision_features(raw_features)
    funding_event, premium_event, active, thresholds = minimal_stress_events(
        features, dates, decisions
    )
    engine = ExecutionEngine(market, funding, _execution_config(cfg, cfg.leverage))
    context: dict[str, Any] = {
        "market": market,
        "funding": funding,
        "dates": dates,
        "features": features,
        "funding_event": funding_event,
        "premium_event": premium_event,
        "active": active,
        "base_thresholds": thresholds,
        "engine": engine,
        "hourly": completed_hour_features(market),
        "bocpd_cache": {},
        "events": None,
        "utility_cache": {},
    }
    base_actions = np.zeros((len(dates), 2), dtype=np.int32)
    base_actions[np.asarray(active, bool)] = (1, TIME_TAKE_BPS)
    base_schedules, base_stats = schedules_and_stats(
        context, cfg, base_actions, windows=PRE2024_WINDOWS
    )
    context["base_schedules"] = base_schedules
    context["base_stats"] = base_stats
    return context, source_hashes


def candidate_events(context: dict[str, Any]) -> dict[str, Any]:
    cached = context.get("events")
    if cached is not None:
        return cached
    engine: ExecutionEngine = context["engine"]
    signals: list[int] = []
    max_exits: list[int] = []
    for signal in np.flatnonzero(np.asarray(context["active"], bool)):
        trade = engine.trade_at(
            int(signal), 1, HOLD_BARS, TIME_TAKE_BPS, NO_STOP_BPS
        )
        if trade is None:
            continue
        signals.append(int(signal))
        max_exits.append(int(trade.exit_position))
    signal_array = np.asarray(signals, dtype=np.int64)
    source_signed = np.where(
        np.asarray(context["funding_event"], bool)[signal_array], 1.0, -1.0
    )
    payload = {
        "signals": signal_array,
        "max_exits": np.asarray(max_exits, dtype=np.int64),
        "source_signed": source_signed,
    }
    context["events"] = payload
    return payload


def fit_event_mask(context: dict[str, Any], events: dict[str, Any]) -> np.ndarray:
    signal_dates = context["dates"].iloc[events["signals"]].reset_index(drop=True)
    exit_dates = context["dates"].iloc[events["max_exits"]].to_numpy(
        dtype="datetime64[ns]"
    )
    return np.asarray(
        (signal_dates >= pd.Timestamp(FIT_START)).to_numpy(bool)
        & (exit_dates < np.datetime64(FIT_END)),
        dtype=bool,
    )


def bocpd_features(
    context: dict[str, Any], hazard_hours: int
) -> tuple[np.ndarray, dict[str, Any]]:
    cache = context["bocpd_cache"]
    if hazard_hours in cache:
        return cache[hazard_hours]
    hourly = context["hourly"]
    fit_hour = np.asarray(
        (hourly.index >= pd.Timestamp(FIT_START))
        & (hourly.index < pd.Timestamp(FIT_END)),
        dtype=bool,
    )
    output, metadata = _model_output(
        hourly,
        fit_hour,
        columns=("ret1", "flow24"),
        secondary_index=1,
        hazard_lambda=int(hazard_hours),
    )
    mapped = exact_hour_map(context["dates"], output)
    values = mapped[BOCPD_COLUMNS].to_numpy(float)
    cache[hazard_hours] = (values, metadata)
    return values, metadata


def event_weights(
    signal_dates: pd.Series,
    source_signed: np.ndarray,
    fit_mask: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Return fit-normalized weights without reading selection-period labels."""

    if mode not in WEIGHT_MODES:
        raise ValueError(f"unsupported weight mode: {mode}")
    weights = np.ones(len(signal_dates), dtype=float)
    years = signal_dates.dt.year.to_numpy(int)
    if mode == "year":
        groups = years.astype(str)
    elif mode == "year_source":
        groups = np.asarray(
            [f"{year}:{source:g}" for year, source in zip(years, source_signed)]
        )
    else:
        groups = np.full(len(signal_dates), "all", dtype=object)
    fit_count = int(fit_mask.sum())
    for group in np.unique(groups[fit_mask]):
        mask = fit_mask & (groups == group)
        weights[mask] = fit_count / int(mask.sum())
    weights /= float(weights[fit_mask].mean())
    return weights


def fit_action_utilities(
    context: dict[str, Any],
    events: dict[str, Any],
    fit_mask: np.ndarray,
    actions: tuple[tuple[int, int], ...],
    risk_lambda: float,
) -> np.ndarray:
    """Build labels only for purged fit events; OOS paths are never label inputs."""

    key = (tuple(actions), float(risk_lambda), _array_hash(fit_mask))
    cache = context["utility_cache"]
    if key in cache:
        return cache[key]
    engine: ExecutionEngine = context["engine"]
    fit_signals = events["signals"][fit_mask]
    columns: list[np.ndarray] = []
    for side, take_bps in actions:
        values = []
        for signal in fit_signals:
            trade = engine.trade_at(
                int(signal), int(side), HOLD_BARS, int(take_bps), NO_STOP_BPS
            )
            if trade is None:
                raise RuntimeError("purged fit action is not executable")
            values.append(trade_utility(trade, float(risk_lambda)))
        columns.append(np.asarray(values, dtype=float))
    target = np.column_stack(columns)
    cache[key] = target
    return target


def fit_weighted_action_ridge(
    design: np.ndarray,
    target_fit: np.ndarray,
    fit_mask: np.ndarray,
    sample_weight: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_design = design[fit_mask]
    if len(train_design) != len(target_fit):
        raise ValueError("target rows must equal the number of purged fit rows")
    weights = np.sqrt(np.asarray(sample_weight[fit_mask], dtype=float))
    augmented = np.column_stack([np.ones(len(train_design)), train_design])
    weighted_design = augmented * weights[:, None]
    weighted_target = target_fit * weights[:, None]
    penalty = np.eye(augmented.shape[1])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        weighted_design.T @ weighted_design + float(ridge) * penalty,
        weighted_design.T @ weighted_target,
    )
    prediction = np.column_stack([np.ones(len(design)), design]) @ coefficients
    return prediction, {
        "intercept": coefficients[0].tolist(),
        "coefficients": coefficients[1:].tolist(),
        "coefficient_l2": float(np.linalg.norm(coefficients[1:])),
    }


def actions_from_prediction(
    positions: int,
    signals: np.ndarray,
    prediction: np.ndarray,
    actions: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Map predicted utility to ABSTAIN or one executable side/exit action."""

    if prediction.shape != (len(signals), len(actions)):
        raise ValueError("prediction shape does not match signals and actions")
    routed = np.zeros((positions, 2), dtype=np.int32)
    chosen = np.argmax(
        np.column_stack([np.zeros(len(prediction), dtype=float), prediction]), axis=1
    )
    for row, action_index in enumerate(chosen):
        if int(action_index) == 0:
            continue
        routed[int(signals[row])] = actions[int(action_index) - 1]
    return routed


def fit_spec(
    context: dict[str, Any], spec: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    events = candidate_events(context)
    signals = events["signals"]
    fit_mask = fit_event_mask(context, events)
    if int(fit_mask.sum()) < 60:
        raise RuntimeError("insufficient purged fit events")
    weak = context["features"][WEAK_FEATURES].iloc[signals].to_numpy(float)
    weak = np.column_stack([weak, events["source_signed"]])
    bocpd, bocpd_metadata = bocpd_features(
        context, int(spec["hazard_hours"])
    )
    design, scaler = tensor_design(
        weak, bocpd[signals], fit_mask, form=str(spec["form"])
    )
    actions = tuple(tuple(int(value) for value in row) for row in spec["actions"])
    target_fit = fit_action_utilities(
        context, events, fit_mask, actions, float(spec["risk_lambda"])
    )
    signal_dates = context["dates"].iloc[signals].reset_index(drop=True)
    weights = event_weights(
        signal_dates,
        events["source_signed"],
        fit_mask,
        str(spec["weight_mode"]),
    )
    prediction, ridge_model = fit_weighted_action_ridge(
        design,
        target_fit,
        fit_mask,
        weights,
        ridge=float(spec["ridge"]),
    )
    routed = actions_from_prediction(
        len(context["dates"]), signals, prediction, actions
    )
    model = {
        "weak_columns": WEAK_FEATURES + [EVENT_SOURCE_FEATURE],
        "bocpd_columns": BOCPD_COLUMNS,
        "bocpd": bocpd_metadata,
        "scaler": scaler,
        "ridge_model": ridge_model,
        "fit_events": int(fit_mask.sum()),
        "fit_last_exit": str(
            context["dates"].iloc[events["max_exits"][fit_mask]].max()
        ),
        "weight_mode": str(spec["weight_mode"]),
    }
    return routed, model


def schedule_window(
    context: dict[str, Any],
    actions: np.ndarray,
    *,
    start: str,
    end: str,
) -> list[Trade]:
    dates = context["dates"]
    period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(
        bool
    )
    trades: list[Trade] = []
    next_allowed = 0
    engine: ExecutionEngine = context["engine"]
    for signal in np.flatnonzero(np.asarray(context["active"], bool) & period):
        signal = int(signal)
        if signal < next_allowed:
            continue
        side, take_bps = (int(value) for value in actions[signal])
        if side == 0:
            continue
        trade = engine.trade_at(
            signal, side, HOLD_BARS, take_bps, NO_STOP_BPS
        )
        if trade is None or not period[trade.exit_position]:
            continue
        trades.append(trade)
        next_allowed = trade.exit_position + 1
    return trades


def schedules_and_stats(
    context: dict[str, Any],
    cfg: Config,
    actions: np.ndarray,
    *,
    windows: dict[str, tuple[str, str]],
) -> tuple[dict[str, list[Trade]], dict[str, dict[str, Any]]]:
    schedules = {
        name: schedule_window(context, actions, start=start, end=end)
        for name, (start, end) in windows.items()
    }
    engine_cfg = _execution_config(cfg, cfg.leverage)
    stats = {
        name: _slim(
            equity_stats(schedules[name], start=start, end=end, cfg=engine_cfg)
        )
        for name, (start, end) in windows.items()
    }
    return schedules, stats


def _action_counts(
    actions: np.ndarray,
    spec_actions: list[list[int]],
    eligible_signals: np.ndarray,
) -> dict[str, int]:
    routed = actions[np.asarray(eligible_signals, dtype=np.int64)]
    counts = {"abstain": int(np.count_nonzero(routed[:, 0] == 0))}
    for side, take_bps in spec_actions:
        name = f"{'long' if side > 0 else 'short'}_"
        name += "time" if int(take_bps) == TIME_TAKE_BPS else f"tp{take_bps // 100}"
        counts[name] = int(
            np.count_nonzero(
                (routed[:, 0] == int(side)) & (routed[:, 1] == int(take_bps))
            )
        )
    return counts


def _grid(context: dict[str, Any], cfg: Config) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_score = _selection_score(context["base_stats"])
    event_signals = candidate_events(context)["signals"]
    for hazard, form, ridge, risk_lambda, weight_mode, action_set in itertools.product(
        HAZARD_GRID,
        MODEL_FORMS,
        RIDGE_GRID,
        RISK_GRID,
        WEIGHT_MODES,
        ACTION_SETS,
    ):
        spec = {
            "features": WEAK_FEATURES + [EVENT_SOURCE_FEATURE],
            "hazard_hours": int(hazard),
            "form": form,
            "ridge": float(ridge),
            "risk_lambda": float(risk_lambda),
            "weight_mode": weight_mode,
            "action_set": action_set,
            "actions": [list(action) for action in ACTION_SETS[action_set]],
        }
        actions, model = fit_spec(context, spec)
        schedules, stats = schedules_and_stats(
            context, cfg, actions, windows=PRE2024_WINDOWS
        )
        score = _selection_score(stats)
        passed = selection_passes(stats)
        beats_base = bool(tuple(score) > tuple(base_score))
        rows.append(
            {
                "spec": spec,
                "dimensions": model["scaler"]["dimensions"],
                "action_counts": _action_counts(
                    actions, spec["actions"], event_signals
                ),
                "stable_action_hash": _array_hash(
                    actions[_stable_action_mask(context["dates"])]
                ),
                "selection_schedule_hashes": {
                    name: _schedule_hash(window_trades)
                    for name, window_trades in schedules.items()
                },
                "selection_passed": passed,
                "beats_base": beats_base,
                "accepted": bool(passed and beats_base),
                "score": score,
                "stats": stats,
            }
        )
    rows.sort(
        key=lambda row: (
            row["accepted"],
            row["selection_passed"],
            *row["score"],
        ),
        reverse=True,
    )
    return rows


def _implementation_hash() -> str:
    functions = (
        completed_hour_features,
        exact_hour_map,
        minimal_stress_events,
        candidate_events,
        fit_event_mask,
        bocpd_features,
        tensor_design,
        event_weights,
        fit_action_utilities,
        fit_weighted_action_ridge,
        actions_from_prediction,
        fit_spec,
        schedule_window,
        schedules_and_stats,
        _grid,
        _freeze_payload,
        _freeze_hash,
        _validate_manifest,
        _write_manifest_once,
        _selection_payload,
        _mark_oos_opened,
        _oos,
        selection_passes,
        oos_passes,
        decision_mask,
        live_decision_features,
        _load_bundle,
        ExecutionEngine.trade_at,
        equity_stats,
    )
    source = "\n\n".join(inspect.getsource(function) for function in functions)
    return hashlib.sha256(source.encode()).hexdigest()


def _stable_action_mask(dates: pd.Series) -> np.ndarray:
    stable_end = pd.Timestamp(SELECTION_END) - pd.Timedelta(minutes=5 * HOLD_BARS)
    return (dates < stable_end).to_numpy(bool)


def _selected_feature_hash(context: dict[str, Any]) -> str:
    events = candidate_events(context)
    source = np.zeros(len(context["dates"]), dtype=float)
    source[events["signals"]] = events["source_signed"]
    frame = context["features"][WEAK_FEATURES].copy()
    frame[EVENT_SOURCE_FEATURE] = source
    frame.insert(0, "date", context["dates"].to_numpy())
    return _frame_hash(frame)


def _freeze_payload(payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "oos_opened",
        "selection_end",
        "expert_spec",
        "spec_hash",
        "implementation_hash",
        "frozen_execution_config",
        "source_prefix_hashes",
        "hourly_feature_prefix_hash",
        "selected_feature_prefix_hash",
        "base_thresholds",
        "base_activation_hash",
        "funding_activation_hash",
        "premium_activation_hash",
        "base_score",
        "base_stats",
        "selected",
        "selection_stats",
        "selection_schedule_hashes",
        "selection_grid",
    )
    return {key: payload[key] for key in keys}


def _freeze_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        _freeze_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _selection_payload(cfg: Config) -> dict[str, Any]:
    context, source_hashes = _context(cfg, cutoff=SELECTION_END)
    if len(context["dates"]) and context["dates"].max() >= pd.Timestamp(SELECTION_END):
        raise RuntimeError("selection source was not physically truncated")
    grid = _grid(context, cfg)
    champion = grid[0]
    if champion["spec"] != FROZEN_CHAMPION:
        raise RuntimeError("frozen action expert is no longer rank one")
    if not champion["accepted"]:
        raise RuntimeError("frozen action expert no longer clears selection")
    actions, model = fit_spec(context, champion["spec"])
    schedules, stats = schedules_and_stats(
        context, cfg, actions, windows=PRE2024_WINDOWS
    )
    payload: dict[str, Any] = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "phase": "pre_2024_freeze",
        "oos_opened": False,
        "selection_end": SELECTION_END,
        "config": asdict(cfg),
        "expert_spec": EXPERT_SPEC,
        "spec_hash": _spec_hash(),
        "implementation_hash": _implementation_hash(),
        "frozen_execution_config": _frozen_execution_config(cfg),
        "source_prefix_hashes": source_hashes,
        "hourly_feature_prefix_hash": _frame_hash(
            context["hourly"].reset_index().rename(columns={"index": "date"})
        ),
        "selected_feature_prefix_hash": _selected_feature_hash(context),
        "base_thresholds": context["base_thresholds"],
        "base_activation_hash": _activation_hash(
            context["active"], context["dates"]
        ),
        "funding_activation_hash": _activation_hash(
            context["funding_event"], context["dates"]
        ),
        "premium_activation_hash": _activation_hash(
            context["premium_event"], context["dates"]
        ),
        "base_score": _selection_score(context["base_stats"]),
        "base_stats": context["base_stats"],
        "selected": {
            "spec": champion["spec"],
            "model": model,
            "stable_action_hash": _array_hash(
                actions[_stable_action_mask(context["dates"])]
            ),
            "action_counts": champion["action_counts"],
            "score": champion["score"],
        },
        "selection_stats": stats,
        "selection_schedule_hashes": {
            name: _schedule_hash(window_trades)
            for name, window_trades in schedules.items()
        },
        "selection_grid": grid,
    }
    payload["freeze_hash"] = _freeze_hash(payload)
    return payload


def _validate_manifest(cfg: Config, manifest: dict[str, Any]) -> None:
    if manifest.get("oos_opened") is not False:
        raise RuntimeError("manifest must be pre-OOS")
    if manifest.get("spec_hash") != _spec_hash():
        raise RuntimeError("action-expert specification changed after freeze")
    if manifest.get("implementation_hash") != _implementation_hash():
        raise RuntimeError("action-expert implementation changed after freeze")
    if manifest.get("frozen_execution_config") != _frozen_execution_config(cfg):
        raise RuntimeError("execution configuration changed after freeze")
    if manifest.get("selected", {}).get("spec") != FROZEN_CHAMPION:
        raise RuntimeError("manifest champion mismatch")
    if manifest.get("freeze_hash") != _freeze_hash(manifest):
        raise RuntimeError("manifest freeze hash mismatch")


def _write_manifest_once(path: Path, payload: dict[str, Any], cfg: Config) -> None:
    _validate_manifest(cfg, payload)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        _validate_manifest(cfg, existing)
        if existing["freeze_hash"] != payload["freeze_hash"]:
            raise RuntimeError("refusing to overwrite a different action-expert freeze")
        return
    _atomic_write_json(path, payload)


def _metric(row: dict[str, Any]) -> str:
    return (
        f"{row['absolute_return_pct']:.2f}% / {row['cagr_pct']:.2f}% / "
        f"{row['strict_mdd_pct']:.2f}% / "
        f"{row['cagr_to_strict_mdd']:.2f} / {row['trades']}"
    )


def _write_docs(path: str, payload: dict[str, Any]) -> None:
    oos = bool(payload.get("oos_opened"))
    lines = [
        "# Minimal-stress weak action expert OOS"
        if oos
        else "# Minimal-stress weak action expert selection",
        "",
        "Metric: absolute return / CAGR / strict MDD / CAGR-MDD / trades.",
        "",
    ]
    if not oos:
        accepted = sum(row["accepted"] for row in payload["selection_grid"])
        base = payload["base_stats"]
        selected = payload["selection_stats"]
        lines += [
            "## Verdict",
            "",
            "**Frozen for OOS.** Funding-relief and premium-discount events own entry. "
            "A year-balanced source-aware weak tensor chooses ABSTAIN or LONG/SHORT "
            "with TP4/TP8/TP12/time; it is not a scalar gate sweep.",
            "",
            f"Multiplicity: {len(payload['selection_grid'])} cells; {accepted} clear the absolute gate and beat the long time-only base.",
            "",
            "| Policy | Train | 2023 selection | Pre-2024 | Score |",
            "|---|---:|---:|---:|---:|",
            f"| Long time-only base | {_metric(base['train'])} | {_metric(base['select_2023'])} | {_metric(base['pre_2024'])} | `{payload['base_score']}` |",
            f"| Weak action expert | {_metric(selected['train'])} | {_metric(selected['select_2023'])} | {_metric(selected['pre_2024'])} | `{payload['selected']['score']}` |",
            "",
            "## Leakage controls",
            "",
            "- Selection market, funding, and premium sources are physically truncated before 2024.",
            "- Market features are prior completed 5-minute bars; BOCPD uses completed hours and exact boundary mapping.",
            "- Scaling and model fitting stop before 2023; every 48-hour utility label exits before 2023.",
            "- Selection schedules enter next-open, pay 6bp/notional/side plus realized funding, force split-contained exits, and use strict favorable-before-adverse MDD.",
        ]
    else:
        lines += [
            "## Verdict",
            "",
            "**OOS gate passed.**" if payload["oos_passed"] else "**OOS gate failed.**",
            "",
            "| Window | Result |",
            "|---|---:|",
        ]
        for name, row in payload["oos_stats"].items():
            lines.append(f"| {name} | {_metric(row)} |")
        lines += [
            "",
            "This exact action expert was manifest-frozen before opening 2024+, but its funding/premium event family has prior research exposure and is therefore shadow OOS rather than pristine market OOS.",
        ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _selection(cfg: Config) -> dict[str, Any]:
    payload = _selection_payload(cfg)
    _write_manifest_once(Path(cfg.manifest_output), payload, cfg)
    _atomic_write_json(Path(cfg.output), payload)
    _write_docs(cfg.docs_output, payload)
    return payload


def _mark_oos_opened(path: Path, manifest: dict[str, Any], output: str) -> None:
    opened = {
        **manifest,
        "phase": "oos_opening",
        "oos_opened": True,
        "oos_opened_at": datetime.now(timezone.utc).isoformat(),
        "oos_output": output,
    }
    _atomic_write_json(path, opened)


def _oos(cfg: Config) -> dict[str, Any]:
    manifest_path = Path(cfg.manifest_output)
    if not manifest_path.exists():
        raise FileNotFoundError("pre-2024 action-expert manifest is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest(cfg, manifest)
    prefix = _selection_payload(cfg)
    if prefix["freeze_hash"] != manifest["freeze_hash"]:
        raise RuntimeError("pre-2024 action-expert replay changed after freeze")
    _mark_oos_opened(manifest_path, manifest, cfg.output)

    context, full_source_hashes = _context(cfg, cutoff=cfg.exclude_from)
    actions, model = fit_spec(context, manifest["selected"]["spec"])
    if model != manifest["selected"]["model"]:
        raise RuntimeError("full-run action-expert model differs from freeze")
    stable = _stable_action_mask(context["dates"])
    if _array_hash(actions[stable]) != manifest["selected"]["stable_action_hash"]:
        raise RuntimeError("full-run action prefix differs from freeze")
    schedules, stats = schedules_and_stats(
        context, cfg, actions, windows=FUTURE_WINDOWS
    )
    payload = {
        **manifest,
        "phase": "oos_result",
        "oos_opened": True,
        "oos_opened_at": datetime.now(timezone.utc).isoformat(),
        "full_source_hashes": full_source_hashes,
        "oos_stats": stats,
        "oos_schedule_hashes": {
            name: _schedule_hash(window_trades)
            for name, window_trades in schedules.items()
        },
        "oos_passed": oos_passes(stats),
    }
    _atomic_write_json(manifest_path, payload)
    _atomic_write_json(Path(cfg.output), payload)
    _write_docs(cfg.docs_output, payload)
    return payload


def run(cfg: Config) -> dict[str, Any]:
    return _oos(cfg) if cfg.open_oos else _selection(cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=Config.input_csv)
    parser.add_argument("--funding-csv", default=Config.funding_csv)
    parser.add_argument("--premium-csv", default=Config.premium_csv)
    parser.add_argument("--exclude-from", default=Config.exclude_from)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--manifest-output", default=Config.manifest_output)
    parser.add_argument("--docs-output", default=Config.docs_output)
    parser.add_argument("--open-oos", action="store_true")
    return parser.parse_args()


def main() -> None:
    payload = run(Config(**vars(parse_args())))
    summary = (
        {
            "phase": payload["phase"],
            "oos_passed": payload["oos_passed"],
            "oos_stats": payload["oos_stats"],
        }
        if payload.get("oos_opened")
        else {
            "phase": payload["phase"],
            "freeze_hash": payload["freeze_hash"],
            "selected": payload["selected"],
            "selection_stats": payload["selection_stats"],
        }
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
