"""Freeze a causal weak-feature x BOCPD tensor exit router.

The premium-overheat pullback state machine remains the sole entry owner.
Capitulation still routes to TP4 and overheat still skips.  For every other
entry, a ridge-regularized tensor predicts the counterfactual utility advantage
of TP4 and TP8 relative to TP12.  Market features are prior-bar live features,
BOCPD diagnostics use completed hours, targets and scaling stop before 2023,
and 2024+ cannot be read until a one-shot pre-2024 manifest is committed.
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
    _fit_active,
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
from training.search_funding_premium_external_state_gate_alpha import _frame_hash
from training.search_inventory_purge_reclaim_alpha import (
    ExecutionEngine,
    Trade,
    _schedule_hash,
    equity_stats,
)
from training.search_pullback_premium_overheat_state_machine_alpha import (
    FIT_END,
    FIT_START,
    FROZEN_CHAMPION as PULLBACK_CHAMPION,
    FUTURE_WINDOWS,
    NO_STOP_BPS,
    SELECTION_END,
    SPEC as PULLBACK_SPEC,
    _schedules_and_stats as base_schedules_and_stats,
    _selection_score,
    build_state_masks,
    fit_state_thresholds,
    oos_passes,
    selection_passes,
    state_feature_frame,
)


WEAK_GROUPS: dict[str, list[str]] = {
    "price": ["htf_3d_return_1", "rex_576_range_pos", "range_vol"],
    "flow": ["quote_vol_z_1d", "taker_imbalance", "dollar_flow_rel_4h_30d"],
    "cross": ["premium_index_change", "kimchi_premium_change", "dxy_momentum"],
}
WEAK_GROUPS.update(
    {
        "price_flow": WEAK_GROUPS["price"] + WEAK_GROUPS["flow"],
        "price_cross": WEAK_GROUPS["price"] + WEAK_GROUPS["cross"],
        "flow_cross": WEAK_GROUPS["flow"] + WEAK_GROUPS["cross"],
        "all": WEAK_GROUPS["price"] + WEAK_GROUPS["flow"] + WEAK_GROUPS["cross"],
    }
)
BOCPD_COLUMNS = ["primary", "short_mass", "run_drop", "secondary", "surprise"]
ACTIONS = (4, 8, 12)
MODEL_FORMS = ("weak", "linear", "tensor", "tensor_pairs")
RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
RISK_GRID = (0.0, 0.25, 0.5)
HAZARD_GRID = (168, 336)

TENSOR_SPEC: dict[str, Any] = {
    "name": "causal_weak_tensor_exit_router",
    "base": "pullback_premium_overheat_state_machine",
    "entry_ownership": "base only",
    "normal_actions_tp_pct": list(ACTIONS),
    "target": "risk-adjusted utility advantage of TP4/TP8 relative to TP12",
    "weak_groups": WEAK_GROUPS,
    "bocpd_columns": BOCPD_COLUMNS,
    "hazard_hours": list(HAZARD_GRID),
    "forms": list(MODEL_FORMS),
    "ridge": list(RIDGE_GRID),
    "risk_lambda": list(RISK_GRID),
    "grid_cells": len(WEAK_GROUPS)
    * len(HAZARD_GRID)
    * len(MODEL_FORMS)
    * len(RIDGE_GRID)
    * len(RISK_GRID),
    "selection_rule": "pass absolute gate and lexicographically beat frozen base",
}
FROZEN_CHAMPION = {
    "group": "price",
    "features": WEAK_GROUPS["price"],
    "hazard_hours": 168,
    "form": "tensor",
    "ridge": 10.0,
    "risk_lambda": 0.0,
    "actions": list(ACTIONS),
}


@dataclass(frozen=True)
class Config(RouterConfig):
    output: str = "results/causal_weak_tensor_exit_router_selection_2026-07-15.json"
    manifest_output: str = (
        "results/causal_weak_tensor_exit_router_manifest_2026-07-15.json"
    )
    docs_output: str = (
        "docs/causal-weak-tensor-exit-router-selection-2026-07-15.md"
    )
    open_oos: bool = False


def _spec_hash() -> str:
    encoded = json.dumps(
        TENSOR_SPEC, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _context(cfg: Config, *, cutoff: str) -> tuple[dict[str, Any], dict[str, str]]:
    market, raw_features, funding, source_hashes = _load_bundle(
        cfg, cutoff=cutoff, premium_tolerance=cfg.live_premium_tolerance
    )
    dates = pd.to_datetime(market["date"])
    decisions = decision_mask(
        dates, "live_hour_signal_bar", window_size=cfg.window_size
    )
    features = live_decision_features(raw_features)
    active, base_thresholds = _fit_active(features, dates, decisions)
    execution_features = state_feature_frame(features)
    execution_thresholds = fit_state_thresholds(execution_features, dates, active)
    capitulation, overheat = build_state_masks(
        execution_features,
        execution_thresholds,
        PULLBACK_CHAMPION["overheat"],
    )
    base_schedules, base_stats = base_schedules_and_stats(
        market,
        funding,
        active,
        capitulation,
        overheat,
        cfg,
        overheat_action=PULLBACK_CHAMPION["action"],
        windows=PRE2024_WINDOWS,
    )
    if not selection_passes(base_stats):
        raise RuntimeError("frozen base no longer passes pre-2024 selection")
    engine = ExecutionEngine(market, funding, _execution_config(cfg, cfg.leverage))
    return (
        {
            "market": market,
            "funding": funding,
            "dates": dates,
            "features": features,
            "active": active,
            "capitulation": capitulation,
            "overheat": overheat,
            "base_thresholds": base_thresholds,
            "execution_thresholds": execution_thresholds,
            "base_schedules": base_schedules,
            "base_stats": base_stats,
            "engine": engine,
            "hourly": completed_hour_features(market),
            "bocpd_cache": {},
            "counterfactual": None,
        },
        source_hashes,
    )


def counterfactual_trades(context: dict[str, Any]) -> dict[str, Any]:
    cached = context.get("counterfactual")
    if cached is not None:
        return cached
    normal = (
        np.asarray(context["active"], dtype=bool)
        & ~np.asarray(context["capitulation"], dtype=bool)
        & ~np.asarray(context["overheat"], dtype=bool)
    )
    signals: list[int] = []
    trades: dict[int, list[Trade]] = {action: [] for action in ACTIONS}
    engine: ExecutionEngine = context["engine"]
    for signal in np.flatnonzero(normal):
        candidates = {
            action: engine.trade_at(
                int(signal),
                1,
                int(PULLBACK_SPEC["hold_bars"]),
                int(action * 100),
                NO_STOP_BPS,
            )
            for action in ACTIONS
        }
        if any(trade is None for trade in candidates.values()):
            continue
        signals.append(int(signal))
        for action in ACTIONS:
            trades[action].append(candidates[action])
    payload = {"signals": np.asarray(signals, dtype=np.int64), "trades": trades}
    context["counterfactual"] = payload
    return payload


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


def tensor_design(
    weak_values: np.ndarray,
    bocpd_values: np.ndarray,
    fit_mask: np.ndarray,
    *,
    form: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if form not in MODEL_FORMS:
        raise ValueError(f"unsupported form: {form}")
    base = weak_values if form == "weak" else np.column_stack([weak_values, bocpd_values])
    mean = np.nanmean(base[fit_mask], axis=0)
    std = np.nanstd(base[fit_mask], axis=0)
    std[std < 1e-8] = 1.0
    standardized = np.nan_to_num(np.clip((base - mean) / std, -8.0, 8.0))
    if form in {"weak", "linear"}:
        design = standardized
    else:
        weak_count = weak_values.shape[1]
        weak = standardized[:, :weak_count]
        bocpd = standardized[:, weak_count:]
        cross = np.einsum("ij,ik->ijk", weak, bocpd).reshape(len(weak), -1)
        design = np.column_stack([standardized, cross])
        if form == "tensor_pairs":
            pairs = [
                weak[:, left] * weak[:, right]
                for left, right in itertools.combinations(range(weak_count), 2)
            ]
            if pairs:
                design = np.column_stack([design, *pairs])
    metadata = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "dimensions": int(design.shape[1]),
    }
    return design, metadata


def utility_advantages(
    counterfactual: dict[str, Any], risk_lambda: float
) -> np.ndarray:
    utilities = {
        action: np.asarray(
            [trade_utility(trade, risk_lambda) for trade in counterfactual["trades"][action]],
            dtype=float,
        )
        for action in ACTIONS
    }
    return np.column_stack(
        [utilities[4] - utilities[12], utilities[8] - utilities[12]]
    )


def fit_ridge_router(
    design: np.ndarray,
    target: np.ndarray,
    fit_mask: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_design = design[fit_mask]
    train_target = target[fit_mask]
    augmented = np.column_stack([np.ones(len(train_design)), train_design])
    penalty = np.eye(augmented.shape[1])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        augmented.T @ augmented + float(ridge) * penalty,
        augmented.T @ train_target,
    )
    prediction = np.column_stack([np.ones(len(design)), design]) @ coefficients
    return prediction, {
        "intercept": coefficients[0].tolist(),
        "coefficients": coefficients[1:].tolist(),
        "coefficient_l2": float(np.linalg.norm(coefficients[1:])),
    }


def actions_from_prediction(
    positions: int, signals: np.ndarray, prediction: np.ndarray
) -> np.ndarray:
    actions = np.full(positions, -1, dtype=np.int16)
    # The neutral comparator is TP12. Exact ties therefore preserve the base.
    score = np.column_stack([np.zeros(len(prediction)), prediction[:, 1], prediction[:, 0]])
    order = np.asarray([12, 8, 4], dtype=np.int16)
    actions[signals] = order[np.argmax(score, axis=1)]
    return actions


def fit_spec(
    context: dict[str, Any], spec: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    counterfactual = counterfactual_trades(context)
    signals = counterfactual["signals"]
    dates = context["dates"].iloc[signals]
    fit_mask = np.asarray(
        (dates >= pd.Timestamp(FIT_START)) & (dates < pd.Timestamp(FIT_END)),
        dtype=bool,
    )
    if int(fit_mask.sum()) < 60:
        raise RuntimeError("insufficient normal fit events")
    weak_columns = list(spec["features"])
    weak = context["features"][weak_columns].iloc[signals].to_numpy(float)
    bocpd, bocpd_metadata = bocpd_features(context, int(spec["hazard_hours"]))
    design, scaler = tensor_design(
        weak, bocpd[signals], fit_mask, form=str(spec["form"])
    )
    target = utility_advantages(counterfactual, float(spec["risk_lambda"]))
    prediction, ridge_model = fit_ridge_router(
        design, target, fit_mask, ridge=float(spec["ridge"])
    )
    actions = actions_from_prediction(len(context["dates"]), signals, prediction)
    model = {
        "weak_columns": weak_columns,
        "bocpd_columns": BOCPD_COLUMNS,
        "bocpd": bocpd_metadata,
        "scaler": scaler,
        "ridge_model": ridge_model,
        "fit_events": int(fit_mask.sum()),
    }
    return actions, model


def schedule_window(
    context: dict[str, Any],
    actions: np.ndarray,
    *,
    start: str,
    end: str,
) -> list[Trade]:
    dates = context["dates"]
    period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    trades: list[Trade] = []
    next_allowed = 0
    engine: ExecutionEngine = context["engine"]
    for signal in np.flatnonzero(np.asarray(context["active"], bool) & period):
        signal = int(signal)
        if signal < next_allowed:
            continue
        if bool(context["capitulation"][signal]):
            take_pct = 4
        elif bool(context["overheat"][signal]):
            continue
        else:
            take_pct = int(actions[signal]) if int(actions[signal]) > 0 else 12
        trade = engine.trade_at(
            signal,
            1,
            int(PULLBACK_SPEC["hold_bars"]),
            int(take_pct * 100),
            NO_STOP_BPS,
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
            equity_stats(
                schedules[name], start=start, end=end, cfg=engine_cfg
            )
        )
        for name, (start, end) in windows.items()
    }
    return schedules, stats


def _grid(context: dict[str, Any], cfg: Config) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_score = _selection_score(context["base_stats"])
    for group, hazard, form, ridge, risk_lambda in itertools.product(
        WEAK_GROUPS,
        HAZARD_GRID,
        MODEL_FORMS,
        RIDGE_GRID,
        RISK_GRID,
    ):
        spec = {
            "group": group,
            "features": WEAK_GROUPS[group],
            "hazard_hours": int(hazard),
            "form": form,
            "ridge": float(ridge),
            "risk_lambda": float(risk_lambda),
            "actions": list(ACTIONS),
        }
        actions, model = fit_spec(context, spec)
        schedules, stats = schedules_and_stats(
            context, cfg, actions, windows=PRE2024_WINDOWS
        )
        score = _selection_score(stats)
        passed = selection_passes(stats)
        beats_base = bool(tuple(score) > tuple(base_score))
        action_counts = {
            f"tp{action}": int(np.count_nonzero(actions == action)) for action in ACTIONS
        }
        rows.append(
            {
                "spec": spec,
                "dimensions": model["scaler"]["dimensions"],
                "action_counts": action_counts,
                "action_hash": _array_hash(actions),
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
        counterfactual_trades,
        bocpd_features,
        tensor_design,
        utility_advantages,
        fit_ridge_router,
        actions_from_prediction,
        fit_spec,
        schedule_window,
        schedules_and_stats,
        decision_mask,
        live_decision_features,
        _fit_active,
        _load_bundle,
        ExecutionEngine.trade_at,
        equity_stats,
    )
    source = "\n\n".join(inspect.getsource(function) for function in functions)
    return hashlib.sha256(source.encode()).hexdigest()


def _stable_action_mask(dates: pd.Series) -> np.ndarray:
    stable_end = pd.Timestamp(SELECTION_END) - pd.Timedelta(
        minutes=5 * int(PULLBACK_SPEC["hold_bars"])
    )
    return (dates < stable_end).to_numpy(bool)


def _freeze_payload(payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "oos_opened",
        "selection_end",
        "tensor_spec",
        "spec_hash",
        "implementation_hash",
        "frozen_execution_config",
        "source_prefix_hashes",
        "hourly_feature_prefix_hash",
        "selected_feature_prefix_hash",
        "base_thresholds",
        "execution_thresholds",
        "base_activation_hash",
        "capitulation_hash",
        "overheat_hash",
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


def _selected_feature_hash(context: dict[str, Any]) -> str:
    columns = list(FROZEN_CHAMPION["features"])
    frame = context["features"][columns].copy()
    frame.insert(0, "date", context["dates"].to_numpy())
    return _frame_hash(frame)


def _selection_payload(cfg: Config) -> dict[str, Any]:
    context, source_hashes = _context(cfg, cutoff=SELECTION_END)
    if len(context["dates"]) and context["dates"].max() >= pd.Timestamp(SELECTION_END):
        raise RuntimeError("selection source was not physically truncated")
    grid = _grid(context, cfg)
    champion = grid[0]
    if champion["spec"] != FROZEN_CHAMPION:
        raise RuntimeError("frozen tensor champion is no longer rank one")
    if not champion["accepted"]:
        raise RuntimeError("frozen tensor champion no longer beats the base")
    selected_actions, selected_model = fit_spec(context, champion["spec"])
    schedules, stats = schedules_and_stats(
        context, cfg, selected_actions, windows=PRE2024_WINDOWS
    )
    payload: dict[str, Any] = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "phase": "pre_2024_freeze",
        "oos_opened": False,
        "selection_end": SELECTION_END,
        "config": asdict(cfg),
        "tensor_spec": TENSOR_SPEC,
        "spec_hash": _spec_hash(),
        "implementation_hash": _implementation_hash(),
        "frozen_execution_config": _frozen_execution_config(cfg),
        "source_prefix_hashes": source_hashes,
        "hourly_feature_prefix_hash": _frame_hash(
            context["hourly"].reset_index().rename(columns={"index": "date"})
        ),
        "selected_feature_prefix_hash": _selected_feature_hash(context),
        "base_thresholds": context["base_thresholds"],
        "execution_thresholds": context["execution_thresholds"],
        "base_activation_hash": _activation_hash(
            context["active"], context["dates"]
        ),
        "capitulation_hash": _activation_hash(
            context["capitulation"], context["dates"]
        ),
        "overheat_hash": _activation_hash(context["overheat"], context["dates"]),
        "base_score": _selection_score(context["base_stats"]),
        "base_stats": context["base_stats"],
        "selected": {
            "spec": champion["spec"],
            "model": selected_model,
            "stable_action_hash": _array_hash(
                selected_actions[_stable_action_mask(context["dates"])]
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
        raise RuntimeError("tensor specification changed after freeze")
    if manifest.get("implementation_hash") != _implementation_hash():
        raise RuntimeError("tensor implementation changed after freeze")
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
            raise RuntimeError("refusing to overwrite a different tensor freeze")
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
        "# Causal weak tensor exit router OOS"
        if oos
        else "# Causal weak tensor exit router selection",
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
            "**Frozen for OOS.** The base remains the sole entry trigger. The tensor "
            "combines three price-structure weak features with five completed-hour "
            "BOCPD diagnostics to route TP4/TP8/TP12; earlier exits may admit a later "
            "base trigger under the unchanged non-overlap rule.",
            "",
            f"Multiplicity: {len(payload['selection_grid'])} cells; {accepted} beat the frozen base after the absolute gate.",
            "",
            "| Policy | Train | 2023 selection | Pre-2024 | Score |",
            "|---|---:|---:|---:|---:|",
            f"| Frozen base | {_metric(base['train'])} | {_metric(base['select_2023'])} | {_metric(base['pre_2024'])} | `{payload['base_score']}` |",
            f"| Tensor router | {_metric(selected['train'])} | {_metric(selected['select_2023'])} | {_metric(selected['pre_2024'])} | `{payload['selected']['score']}` |",
            "",
            "## Leakage controls",
            "",
            "- Every source is physically truncated before 2024 for selection.",
            "- Market features are prior-bar live features; BOCPD uses `[H-1h,H)` and exact H mapping.",
            "- Scaling, BOCPD standardization, and counterfactual action labels stop before 2023.",
            "- Entry is next-open; costs, realized funding, split-contained exits, and strict MDD are unchanged.",
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
            "The pullback family has prior research exposure; this exact manifest-frozen tensor is a shadow result, not pristine market OOS.",
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
        raise FileNotFoundError("pre-2024 tensor manifest is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest(cfg, manifest)
    prefix = _selection_payload(cfg)
    if prefix["freeze_hash"] != manifest["freeze_hash"]:
        raise RuntimeError("pre-2024 tensor replay changed after freeze")
    _mark_oos_opened(manifest_path, manifest, cfg.output)

    context, full_source_hashes = _context(cfg, cutoff=cfg.exclude_from)
    actions, model = fit_spec(context, manifest["selected"]["spec"])
    if model != manifest["selected"]["model"]:
        raise RuntimeError("full-run tensor model differs from freeze")
    stable = _stable_action_mask(context["dates"])
    if _array_hash(actions[stable]) != manifest["selected"]["stable_action_hash"]:
        raise RuntimeError("full-run tensor action prefix differs from freeze")
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
