"""Audit and reject the seed-sensitive weak-feature responsibility candidate.

The audit is physically truncated before 2024. It first reproduces the
160-tree/seed-715 selection hit, then tests the same specification across five
seeds with 160 and 2,000 trees plus a 10,000-tree prediction ensemble. A single
seed hit is not promotable. This module intentionally has no path that opens
2024+ data.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

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
from training.search_funding_premium_external_state_gate_alpha import _frame_hash
from training.search_inventory_purge_reclaim_alpha import ExecutionEngine, Trade, _schedule_hash, equity_stats
from training.search_liveparity_state_feature_interactions import (
    STATE_FEATURE_NAMES,
    feature_matrix as state_feature_matrix,
    immutable_anchors,
    net_target,
    slim,
    state_bank,
)
import training.search_market_braid_alpha as market_braid
import training.search_nested_barrier_witness_alpha as nested_barrier
import training.search_orderflow_trophic_succession_alpha as orderflow

FIT_START = "2020-07-01"
FIT_END = "2023-01-01"
SELECTION_END = "2024-01-01"
DEFAULT_OUTPUT = "results/weak_feature_responsibility_stability_rejection_2026-07-15.json"
DEFAULT_DOCS = "docs/weak-feature-responsibility-stability-rejection-2026-07-15.md"
NO_BARRIER_BPS = 1_000_000
PA_COLUMNS = [
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
]
OTHER_COLUMNS = [
    "dxy_momentum",
    "usdkrw_zscore",
    "kimchi_premium_change",
    "taker_imbalance",
    "volume_zscore",
    "funding_zscore",
    "premium_index_zscore",
    "funding_rate",
    "premium_index_change",
]
WEAK_COLUMNS = [
    "nested_high_work_ratio",
    "nested_low_work_ratio",
    "nested_high_coalescence",
    "nested_low_coalescence",
    "nested_recent_24h_side",
    "nested_recent_48h_side",
    "nested_recent_age_capped",
    "braid_recent_24h_side",
    "braid_recent_48h_side",
    "braid_recent_age_capped",
]
FEATURE_COLUMNS = STATE_FEATURE_NAMES + PA_COLUMNS + OTHER_COLUMNS + WEAK_COLUMNS
CANDIDATE_SPEC: dict[str, Any] = {
    "name": "weak_feature_responsibility_candidate",
    "model": "RandomForestRegressor",
    "n_estimators": 160,
    "max_depth": 3,
    "min_samples_leaf": 16,
    "max_features": 0.7,
    "random_state": 715,
    "target": "48h net trade return",
    "anchor_cooldown_bars": 144,
    "fit_window": {"start": FIT_START, "end_exclusive": FIT_END},
    "activation_quantile": 0.50,
    "activation_threshold": 0.00365087256140527,
    "leverage": 0.50,
    "funding_exit": {"hold_bars": 576, "take_bps": 400, "stop_bps": NO_BARRIER_BPS},
    "premium_exit": {"hold_bars": 144, "take_bps": NO_BARRIER_BPS, "stop_bps": 300},
    "decision_clock": ":00 live_hour_signal_bar; current market bar excluded",
    "diagnostic_shift": "nested barrier and market-braid completed-bar diagnostics shifted one 5m row to decision boundary",
    "auxiliary_timing": "OI delayed by builder; premium live tolerance 10m; realized funding",
    "costs": "6bp/notional/side; next-open entry; stop before take; strict MDD; non-overlap; split-contained exits",
    "selection_end_exclusive": SELECTION_END,
    "feature_columns": FEATURE_COLUMNS,
}
STABILITY_SPEC: dict[str, Any] = {
    "seeds": [7, 71, 715, 2026, 71515],
    "tree_counts": [160, 2000],
    "ensemble": "mean prediction from five 2,000-tree forests",
    "minimum_large_forest_seed_passes": 3,
    "require_large_forest_ensemble_pass": True,
    "promotion_rule": "candidate reproducible and stable across model sampling",
}
EXPECTED_CANDIDATE_STATS: dict[str, dict[str, float | int]] = {
    "train": {"absolute_return_pct": 110.3052052715706, "cagr_pct": 34.591111335620624, "strict_mdd_pct": 7.748513455818196, "cagr_to_strict_mdd": 4.464225497298051, "trades": 125},
    "select_2023": {"absolute_return_pct": 9.07791176704642, "cagr_pct": 9.084403756636394, "strict_mdd_pct": 2.938866502924964, "cagr_to_strict_mdd": 3.091125012856135, "trades": 15},
    "select_2023_h1": {"absolute_return_pct": 7.150108234303021, "cagr_pct": 14.953915759004555, "strict_mdd_pct": 2.938866502924964, "cagr_to_strict_mdd": 5.088327674673682, "trades": 10},
    "select_2023_h2": {"absolute_return_pct": 1.7991615356355073, "cagr_pct": 3.603078360374923, "strict_mdd_pct": 2.8831125573808736, "cagr_to_strict_mdd": 1.2497182432752791, "trades": 5},
    "pre_2024": {"absolute_return_pct": 129.39652624762962, "cagr_pct": 26.757733799546823, "strict_mdd_pct": 7.748513455818196, "cagr_to_strict_mdd": 3.4532731926089646, "trades": 140},
}


@dataclass(frozen=True)
class Config(AuditConfig):
    output: str = DEFAULT_OUTPUT
    docs_output: str = DEFAULT_DOCS
    spot_premium_csv: str = "data/cache_spot_premium_5m_2020-01-01_2026-06-01.csv.gz"
    leverage: float = 0.50


def causal_shift(values: np.ndarray, fill: float | int = 0) -> np.ndarray:
    """Expose only the previous completed 5m row at the decision boundary."""
    array = np.asarray(values)
    out = np.full(array.shape, fill, dtype=np.result_type(array.dtype, np.asarray(fill).dtype))
    out[1:] = array[:-1]
    return out


def recent_side(side: np.ndarray, bars: int) -> tuple[np.ndarray, np.ndarray]:
    side = np.asarray(side, dtype=np.int8)
    index = np.arange(len(side))
    last = np.maximum.accumulate(np.where(side != 0, index, -1))
    age = index - last
    out = np.zeros(len(side), dtype=np.int8)
    ok = (last >= 0) & (age <= int(bars))
    out[ok] = side[last[ok]]
    return out, np.where(last >= 0, age, 9999)


def _resolve_existing(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate.resolve()
    fallback = Path("/home/pakchu/rllm") / path
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(path)


def _spec_hash() -> str:
    payload = {"candidate": CANDIDATE_SPEC, "stability": STABILITY_SPEC}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _implementation_hash() -> str:
    functions = (
        causal_shift,
        recent_side,
        build_design,
        fit_candidate_model,
        fit_stability_model,
        ensemble_fit,
        stability_decision,
        routed_schedule,
        schedules_and_stats,
        state_bank,
        state_feature_matrix,
        immutable_anchors,
        net_target,
        nested_barrier.build_barrier_bank,
        nested_barrier.coalesced_barrier_signals,
        market_braid.build_bar_state,
        market_braid.market_braid_events,
        ExecutionEngine.trade_at,
        equity_stats,
    )
    return hashlib.sha256("\n\n".join(inspect.getsource(fn) for fn in functions).encode()).hexdigest()


def _activation_hash(active: np.ndarray, dates: pd.Series) -> str:
    positions = np.flatnonzero(np.asarray(active, dtype=bool))
    records = [[int(p), str(pd.Timestamp(dates.iloc[int(p)]))] for p in positions]
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()


def _array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _action_spec(funding_leg: bool) -> tuple[int, int, int]:
    spec = CANDIDATE_SPEC["funding_exit"] if funding_leg else CANDIDATE_SPEC["premium_exit"]
    return int(spec["hold_bars"]), int(spec["take_bps"]), int(spec["stop_bps"])


def _load_braid_inputs(cfg: Config) -> tuple[pd.DataFrame, pd.Series]:
    old_market, old_spot = market_braid.MARKET, market_braid.SPOT_PREMIUM
    try:
        market_braid.MARKET = str(_resolve_existing("data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz"))
        market_braid.SPOT_PREMIUM = str(_resolve_existing(cfg.spot_premium_csv))
        return market_braid.load_pre2024()
    finally:
        market_braid.MARKET, market_braid.SPOT_PREMIUM = old_market, old_spot


def _load_nested_inputs() -> tuple[pd.DataFrame, pd.Series]:
    old_market = orderflow.MARKET
    try:
        orderflow.MARKET = str(_resolve_existing("data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz"))
        return nested_barrier.load_pre2024()
    finally:
        orderflow.MARKET = old_market


def build_design(cfg: Config) -> dict[str, Any]:
    cfg = replace(
        cfg,
        input_csv=str(_resolve_existing(cfg.input_csv)),
        funding_csv=str(_resolve_existing(cfg.funding_csv)),
        premium_csv=str(_resolve_existing(cfg.premium_csv)),
    )
    market, raw_features, funding, source_hashes = _load_bundle(cfg, cutoff=SELECTION_END, premium_tolerance=cfg.live_premium_tolerance)
    dates = pd.to_datetime(market["date"])
    if len(dates) and dates.max() >= pd.Timestamp(SELECTION_END):
        raise RuntimeError("selection source was not physically truncated")
    features = live_decision_features(raw_features)
    decisions = decision_mask(dates, "live_hour_signal_bar", window_size=cfg.window_size)
    funding_leg = decisions & _component_mask(features, "funding10_trend70")
    premium_leg = decisions & _component_mask(features, "premium20_mom90")
    base = funding_leg | premium_leg
    bank = state_bank(market, dates)
    valid = (bank["kalman"] >= 0) & (bank["bocpd"] >= 0) & (bank["semimarkov"] >= 0)
    base &= valid

    nested_market, nested_dates = _load_nested_inputs()
    if not np.array_equal(dates.to_numpy(), nested_dates.to_numpy()):
        raise RuntimeError("nested-barrier grid mismatch")
    barrier_bank = nested_barrier.build_barrier_bank(nested_market)
    long_sig, short_sig, info = nested_barrier.coalesced_barrier_signals(
        nested_market, barrier_bank, min_coalescence=3, touch_width=0.001, branch="depleted_continuation"
    )
    nested_side = causal_shift(long_sig.astype(np.int8) - short_sig.astype(np.int8))
    nested24, nested_age = recent_side(nested_side, 288)
    nested48, _ = recent_side(nested_side, 576)

    braid_market, braid_dates = _load_braid_inputs(cfg)
    if not np.array_equal(dates.to_numpy(), braid_dates.to_numpy()):
        raise RuntimeError("market-braid grid mismatch")
    braid_state = market_braid.build_bar_state(braid_market)
    events = market_braid.market_braid_events(
        braid_state, shock_z=2.0, passage_z=0.5, max_age=144, topology_mode="relative_order"
    )
    braid_side = causal_shift(events.signal_side.to_numpy(np.int8))
    braid24, braid_age = recent_side(braid_side, 288)
    braid48, _ = recent_side(braid_side, 576)

    state = state_feature_matrix(bank, funding_leg, premium_leg)
    raw = np.column_stack([state, *[pd.to_numeric(features[c], errors="coerce").to_numpy(float) for c in PA_COLUMNS + OTHER_COLUMNS]])
    weak = np.column_stack([
        causal_shift(info["high_work_ratio"], np.nan),
        causal_shift(info["low_work_ratio"], np.nan),
        causal_shift(info["high_coalescence"]),
        causal_shift(info["low_coalescence"]),
        nested24,
        nested48,
        np.minimum(nested_age, 576),
        braid24,
        braid48,
        np.minimum(braid_age, 576),
    ])
    x0 = np.column_stack([raw, weak])
    fit = ((dates >= pd.Timestamp(FIT_START)) & (dates < pd.Timestamp(FIT_END))).to_numpy(bool)
    med = np.nanmedian(x0[fit], axis=0)
    matrix = np.clip(np.where(np.isfinite(x0), x0, med), -20.0, 20.0)
    return {
        "market": market,
        "funding": funding,
        "dates": dates,
        "features": features,
        "matrix": matrix,
        "fit": fit,
        "base": base,
        "funding_leg": funding_leg,
        "premium_leg": premium_leg,
        "source_hashes": source_hashes,
        "feature_prefix_hash": _frame_hash(pd.DataFrame(matrix, columns=FEATURE_COLUMNS).assign(date=dates.to_numpy())),
        "base_activation_hash": _activation_hash(base, dates),
        "nested_side_hash": _array_hash(nested_side),
        "braid_side_hash": _array_hash(braid_side),
    }


def _fit_model(
    context: dict[str, Any],
    cfg: Config,
    *,
    n_estimators: int,
    random_state: int,
) -> dict[str, Any]:
    engine = ExecutionEngine(context["market"], context["funding"], _execution_config(cfg, cfg.leverage))
    anchors = immutable_anchors(context["base"], int(CANDIDATE_SPEC["anchor_cooldown_bars"]))
    positions = np.flatnonzero(anchors & context["fit"])
    y = np.asarray([net_target(engine, int(pos), 576, cfg) for pos in positions], dtype=float)
    good = np.isfinite(y)
    positions, y = positions[good], y[good]
    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(CANDIDATE_SPEC["max_depth"]),
        min_samples_leaf=int(CANDIDATE_SPEC["min_samples_leaf"]),
        max_features=float(CANDIDATE_SPEC["max_features"]),
        random_state=int(random_state),
        n_jobs=-1,
    ).fit(context["matrix"][positions], y)
    train_pred = model.predict(context["matrix"][positions])
    threshold = float(np.quantile(train_pred, float(CANDIDATE_SPEC["activation_quantile"])))
    all_anchor_positions = np.flatnonzero(anchors)
    pred = model.predict(context["matrix"][all_anchor_positions])
    active = np.zeros(len(context["market"]), dtype=bool)
    active[all_anchor_positions] = pred >= threshold
    return {
        "engine": engine,
        "anchors": anchors,
        "active": active,
        "threshold": threshold,
        "train_examples": int(len(y)),
        "train_positions": positions,
        "anchor_positions": all_anchor_positions,
        "train_predictions": train_pred,
        "anchor_predictions": pred,
    }


def fit_candidate_model(context: dict[str, Any], cfg: Config) -> dict[str, Any]:
    fitted = _fit_model(
        context,
        cfg,
        n_estimators=int(CANDIDATE_SPEC["n_estimators"]),
        random_state=int(CANDIDATE_SPEC["random_state"]),
    )
    expected = float(CANDIDATE_SPEC["activation_threshold"])
    if not np.isclose(float(fitted["threshold"]), expected, rtol=0.0, atol=1e-15):
        raise RuntimeError(f"activation threshold drifted: {fitted['threshold']}")
    return fitted


def fit_stability_model(context: dict[str, Any], cfg: Config, *, trees: int, seed: int) -> dict[str, Any]:
    return _fit_model(context, cfg, n_estimators=trees, random_state=seed)


def ensemble_fit(context: dict[str, Any], fitted_models: list[dict[str, Any]]) -> dict[str, Any]:
    if not fitted_models:
        raise ValueError("ensemble requires at least one fitted model")
    first = fitted_models[0]
    for fitted in fitted_models[1:]:
        if not np.array_equal(fitted["train_positions"], first["train_positions"]):
            raise RuntimeError("ensemble train positions differ")
        if not np.array_equal(fitted["anchor_positions"], first["anchor_positions"]):
            raise RuntimeError("ensemble anchor positions differ")
    train_pred = np.mean(np.stack([fitted["train_predictions"] for fitted in fitted_models]), axis=0)
    anchor_pred = np.mean(np.stack([fitted["anchor_predictions"] for fitted in fitted_models]), axis=0)
    threshold = float(np.quantile(train_pred, float(CANDIDATE_SPEC["activation_quantile"])))
    active = np.zeros(len(context["market"]), dtype=bool)
    active[first["anchor_positions"]] = anchor_pred >= threshold
    return {
        "engine": first["engine"],
        "anchors": first["anchors"],
        "active": active,
        "threshold": threshold,
        "train_examples": first["train_examples"],
        "train_positions": first["train_positions"],
        "anchor_positions": first["anchor_positions"],
        "train_predictions": train_pred,
        "anchor_predictions": anchor_pred,
    }


def routed_schedule(context: dict[str, Any], fitted: dict[str, Any], *, start: str, end: str) -> list[Trade]:
    dates = context["dates"]
    period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    trades: list[Trade] = []
    next_allowed = 0
    engine: ExecutionEngine = fitted["engine"]
    active = np.asarray(fitted["active"], dtype=bool)
    funding_leg = np.asarray(context["funding_leg"], dtype=bool)
    for signal in np.flatnonzero(active & period):
        signal = int(signal)
        if signal < next_allowed:
            continue
        hold, take, stop = _action_spec(bool(funding_leg[signal]))
        trade = engine.trade_at(signal, 1, hold, take, stop)
        if trade is None or not period[trade.exit_position]:
            continue
        trades.append(trade)
        next_allowed = trade.exit_position + 1
    return trades


def schedules_and_stats(context: dict[str, Any], fitted: dict[str, Any], cfg: Config) -> tuple[dict[str, list[Trade]], dict[str, dict[str, Any]]]:
    schedules = {name: routed_schedule(context, fitted, start=start, end=end) for name, (start, end) in PRE2024_WINDOWS.items()}
    stats = {
        name: slim(equity_stats(schedules[name], start=start, end=end, cfg=_execution_config(cfg, cfg.leverage)))
        for name, (start, end) in PRE2024_WINDOWS.items()
    }
    return schedules, stats


def assert_expected_candidate_stats(stats: dict[str, dict[str, Any]]) -> None:
    for window, expected in EXPECTED_CANDIDATE_STATS.items():
        actual = stats[window]
        for key, value in expected.items():
            if isinstance(value, int):
                if int(actual[key]) != value:
                    raise RuntimeError(f"{window}.{key} drifted: {actual[key]} != {value}")
            elif not np.isclose(float(actual[key]), float(value), rtol=0.0, atol=1e-10):
                raise RuntimeError(f"{window}.{key} drifted: {actual[key]} != {value}")
    yearly = ["train_2020h2", "train_2021", "train_2022", "select_2023_h1", "select_2023_h2"]
    if not all(stats[name]["absolute_return_pct"] > 0.0 for name in yearly):
        raise RuntimeError("expected all source year/half-year returns to be positive")
    if not selection_passes(stats):
        raise RuntimeError("seed-715 candidate no longer reproduces its selection hit")


def stability_decision(*, candidate_pass: bool, large_seed_passes: int, ensemble_pass: bool) -> bool:
    return bool(
        candidate_pass
        and large_seed_passes >= int(STABILITY_SPEC["minimum_large_forest_seed_passes"])
        and (ensemble_pass or not bool(STABILITY_SPEC["require_large_forest_ensemble_pass"]))
    )


def _audit_payload(payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "oos_opened",
        "sealed_windows",
        "selection_end_exclusive",
        "candidate_spec",
        "stability_spec",
        "spec_hash",
        "implementation_hash",
        "source_prefix_hashes",
        "feature_prefix_hash",
        "base_activation_hash",
        "nested_side_hash",
        "braid_side_hash",
        "candidate",
        "small_forest_seed_runs",
        "large_forest_seed_runs",
        "large_forest_ensemble",
        "stability_summary",
    )
    return {key: payload[key] for key in keys}


def _audit_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(_audit_payload(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _run_record(
    context: dict[str, Any],
    fitted: dict[str, Any],
    cfg: Config,
    *,
    label: str,
    trees: int,
    seed: int | str,
) -> dict[str, Any]:
    schedules, stats = schedules_and_stats(context, fitted, cfg)
    return {
        "label": label,
        "trees": int(trees),
        "seed": seed,
        "threshold": float(fitted["threshold"]),
        "passes_selection": bool(selection_passes(stats)),
        "activation_schedule_hash": _activation_hash(fitted["active"], context["dates"]),
        "schedule_hashes": {name: _schedule_hash(trades) for name, trades in schedules.items()},
        "stats": stats,
    }


def audit_payload(cfg: Config) -> dict[str, Any]:
    context = build_design(cfg)
    candidate_fit = fit_candidate_model(context, cfg)
    candidate = _run_record(
        context,
        candidate_fit,
        cfg,
        label="selection_hit",
        trees=int(CANDIDATE_SPEC["n_estimators"]),
        seed=int(CANDIDATE_SPEC["random_state"]),
    )
    assert_expected_candidate_stats(candidate["stats"])

    small_runs: list[dict[str, Any]] = []
    for seed in STABILITY_SPEC["seeds"]:
        fitted = fit_stability_model(context, cfg, trees=160, seed=int(seed))
        small_runs.append(_run_record(context, fitted, cfg, label=f"seed_{seed}_160", trees=160, seed=int(seed)))

    large_runs: list[dict[str, Any]] = []
    large_fits: list[dict[str, Any]] = []
    for seed in STABILITY_SPEC["seeds"]:
        fitted = fit_stability_model(context, cfg, trees=2000, seed=int(seed))
        large_fits.append(fitted)
        large_runs.append(_run_record(context, fitted, cfg, label=f"seed_{seed}_2000", trees=2000, seed=int(seed)))
    ensemble = _run_record(
        context,
        ensemble_fit(context, large_fits),
        cfg,
        label="mean_5x2000",
        trees=10000,
        seed="ensemble",
    )

    small_passes = sum(int(row["passes_selection"]) for row in small_runs)
    large_passes = sum(int(row["passes_selection"]) for row in large_runs)
    promotable = stability_decision(
        candidate_pass=bool(candidate["passes_selection"]),
        large_seed_passes=large_passes,
        ensemble_pass=bool(ensemble["passes_selection"]),
    )
    payload: dict[str, Any] = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "phase": "pre_2024_stability_audit",
        "oos_opened": False,
        "sealed_windows": ["2024+"],
        "selection_end_exclusive": SELECTION_END,
        "config": asdict(cfg),
        "candidate_spec": CANDIDATE_SPEC,
        "stability_spec": STABILITY_SPEC,
        "spec_hash": _spec_hash(),
        "implementation_hash": _implementation_hash(),
        "source_prefix_hashes": context["source_hashes"],
        "feature_prefix_hash": context["feature_prefix_hash"],
        "base_activation_hash": context["base_activation_hash"],
        "nested_side_hash": context["nested_side_hash"],
        "braid_side_hash": context["braid_side_hash"],
        "candidate": candidate,
        "small_forest_seed_runs": small_runs,
        "large_forest_seed_runs": large_runs,
        "large_forest_ensemble": ensemble,
        "stability_summary": {
            "small_forest_seed_passes": small_passes,
            "small_forest_seed_cells": len(small_runs),
            "large_forest_seed_passes": large_passes,
            "large_forest_seed_cells": len(large_runs),
            "large_forest_ensemble_pass": bool(ensemble["passes_selection"]),
            "promotable": promotable,
            "decision": "promote" if promotable else "reject",
            "reason": "stable across model sampling" if promotable else "selection hit collapses under seed and forest-size convergence",
        },
    }
    if promotable:
        raise RuntimeError("audit expectation changed: candidate is now promotable and requires a new sealed review")
    payload["audit_hash"] = _audit_hash(payload)
    return payload


def validate_audit(payload: dict[str, Any]) -> None:
    if payload.get("oos_opened") is not False or payload.get("sealed_windows") != ["2024+"]:
        raise RuntimeError("audit must keep 2024+ sealed")
    if payload.get("candidate_spec") != CANDIDATE_SPEC or payload.get("stability_spec") != STABILITY_SPEC:
        raise RuntimeError("audit specification changed")
    if payload.get("spec_hash") != _spec_hash() or payload.get("implementation_hash") != _implementation_hash():
        raise RuntimeError("audit implementation changed")
    if payload.get("audit_hash") != _audit_hash(payload):
        raise RuntimeError("audit hash mismatch")
    assert_expected_candidate_stats(payload["candidate"]["stats"])
    if payload["stability_summary"]["decision"] != "reject" or payload["stability_summary"]["promotable"]:
        raise RuntimeError("seed-sensitive candidate must not be promoted")


def _atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    tmp.replace(target)


def _metric(row: dict[str, Any]) -> str:
    return f"{row['absolute_return_pct']:.2f}% / {row['cagr_pct']:.2f}% / {row['strict_mdd_pct']:.2f}% / {row['cagr_to_strict_mdd']:.2f} / {row['trades']}"


def _write_docs(path: str | Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Weak-feature responsibility stability rejection — 2026-07-15",
        "",
        "**Rejected; 2024+ remains sealed.** The apparent alpha was a seed-sensitive random-forest selection hit.",
        "",
        "Metric: absolute return / CAGR / strict MDD / CAGR-to-strict-MDD / trades.",
        "",
        "## Audit protocol",
        "",
        "- Reproduce the original 160-tree, seed-715 candidate, then rerun seeds 7/71/715/2026/71515 at 160 and 2,000 trees.",
        "- Average the five 2,000-tree prediction vectors as a 10,000-tree convergence ensemble.",
        "- Promotion requires at least three large-forest seed passes and an ensemble pass; neither occurred.",
        "- Target is 48h net trade return; anchors use a 144-bar cooldown and are fit only on 2020-07-01..2022-12-31.",
        "- Funding leg: 48h cap, 4% take, no stop. Premium leg: 12h cap, no take, 3% stop.",
        "- Decisions occur on `:00`; current market bar is excluded; nested barrier and market-braid diagnostics are shifted one 5m row.",
        "- Market/funding/premium/OI/spot-premium inputs are physically truncated before 2024; `oos_opened=false`.",
        "",
        "## Original selection hit",
        "",
        "| Window | Result |",
        "|---|---:|",
    ]
    for name in ("train", "select_2023", "select_2023_h1", "select_2023_h2", "pre_2024"):
        lines.append(f"| {name} | {_metric(payload['candidate']['stats'][name])} |")
    lines += [
        "",
        "## Stability result",
        "",
        f"- 160-tree seeds passing: **{payload['stability_summary']['small_forest_seed_passes']}/{payload['stability_summary']['small_forest_seed_cells']}**.",
        f"- 2,000-tree seeds passing: **{payload['stability_summary']['large_forest_seed_passes']}/{payload['stability_summary']['large_forest_seed_cells']}**.",
        f"- 10,000-tree mean ensemble passing: **{payload['stability_summary']['large_forest_ensemble_pass']}**.",
        f"- Decision: **{payload['stability_summary']['decision']}**.",
        "",
        "| Large-forest run | pre-2024 | 2023 | 2023 H2 |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["large_forest_seed_runs"] + [payload["large_forest_ensemble"]]:
        stats = row["stats"]
        lines.append(f"| {row['label']} | {_metric(stats['pre_2024'])} | {_metric(stats['select_2023'])} | {_metric(stats['select_2023_h2'])} |")
    lines += ["", f"Audit hash: `{payload['audit_hash']}`", ""]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")


def run(cfg: Config) -> dict[str, Any]:
    payload = audit_payload(cfg)
    validate_audit(payload)
    _atomic_write_json(cfg.output, payload)
    _write_docs(cfg.docs_output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=Config.input_csv)
    parser.add_argument("--funding-csv", default=Config.funding_csv)
    parser.add_argument("--premium-csv", default=Config.premium_csv)
    parser.add_argument("--spot-premium-csv", default=Config.spot_premium_csv)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--docs-output", default=Config.docs_output)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(Config(**vars(args)))
    print(json.dumps({"phase": payload["phase"], "audit_hash": payload["audit_hash"], "stability_summary": payload["stability_summary"]}, indent=2))


if __name__ == "__main__":
    main()
