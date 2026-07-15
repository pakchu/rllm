"""Freeze a structured pair-context formula expert on minimal stress events.

The causal entry universe is the already-audited union of funding-relief and
premium-discount stress events.  Fifteen prior-bar Alpha101/VPIN formula
features form the local evidence.  Exactly two slow context variables route
that evidence through low/mid/high responsibilities; this is a structured
mixture of experts rather than a scalar gate or an unrestricted feature
tensor.  A ridge action critic is fit only on fully resolved 2020-2022 paths.

Selection sources are physically truncated before 2024.  The exact champion
must be committed in a one-shot manifest before ``--open-oos`` can read 2024+
rows.
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

import training.search_minimal_stress_weak_action_expert_alpha as base
from training.audit_confirmed_pullback_squeeze_live_parity import (
    _activation_hash,
    _load_bundle,
    decision_mask,
    live_decision_features,
)
from training.search_alpha101_derivative_alphas import add_features as add_a101
from training.search_funding_premium_external_state_gate_alpha import _frame_hash
from training.search_inventory_purge_reclaim_alpha import (
    ExecutionEngine,
    _schedule_hash,
    equity_stats,
)
from training.search_pullback_premium_overheat_state_machine_alpha import (
    FROZEN_CONFIG_KEYS,
    FUTURE_WINDOWS,
    PRE2024_WINDOWS,
    SELECTION_END,
    _selection_score,
    oos_passes,
    selection_passes,
)
from training.search_vpin_formulaic_alpha import add_vpin_formulaic_features


CORE_FEATURES = [
    "a_ret_z_12",
    "a_ret_vol_corr_288",
    "a_absret_vol_rank",
    "a_vwap_gap_z",
    "a_clv_rank_288",
    "vp_ret_rank_72",
    "vp_imb_z_144",
    "vp_vpin_z_144",
    "fq_ret_vol_corr_72",
    "fq_signed_vol_pressure_rank",
    "x101_volcorr_reversal_up",
    "x101_gap_revert_up",
    "x101_vwap_exhaust_down",
    "vx_toxic_selloff",
    "vx_lowtox_momo_long",
]
PAIR_CONTEXTS = (
    "dxy_momentum",
    "kimchi_premium_change",
    "usdkrw_momentum",
    "dollar_flow_rel_4h_30d",
    "quote_vol_z_1d",
)
TAIL_GRID = (0.20, 0.33)
RIDGE_GRID = (0.1, 1.0, 10.0, 100.0)
RISK_GRID = (0.25, 0.50)
WEIGHT_MODES = ("uniform", "year")
ACTION_SETS: dict[str, tuple[tuple[int, int], ...]] = {
    "sides_time": (
        (1, base.TIME_TAKE_BPS),
        (-1, base.TIME_TAKE_BPS),
    ),
    "tp4_time": base.ACTION_SETS["tp4_time"],
}

EXPERT_SPEC: dict[str, Any] = {
    "name": "minimal_stress_pair_context_formula_expert",
    "entry": "funding relief OR premium discount after train-only quantiles",
    "core_features": CORE_FEATURES,
    "pair_contexts": list(PAIR_CONTEXTS),
    "context_pairs": [list(pair) for pair in itertools.combinations(PAIR_CONTEXTS, 2)],
    "tail_grid": list(TAIL_GRID),
    "actions": {name: [list(action) for action in actions] for name, actions in ACTION_SETS.items()},
    "ridge_grid": list(RIDGE_GRID),
    "risk_grid": list(RISK_GRID),
    "weight_modes": list(WEIGHT_MODES),
    "hold_bars": base.HOLD_BARS,
    "stop_bps": base.NO_STOP_BPS,
    "leverage": 0.45,
    "design": "core + source + core*source + each context low/high responsibility and core interactions",
    "target": "absolute risk-adjusted utility for each executable action; neutral utility is zero",
    "label_purge": "48-hour counterfactual exit must precede 2023-01-01",
    "grid_cells": len(tuple(itertools.combinations(PAIR_CONTEXTS, 2)))
    * len(TAIL_GRID)
    * len(ACTION_SETS)
    * len(RISK_GRID)
    * len(RIDGE_GRID)
    * len(WEIGHT_MODES),
    "selection_rule": "absolute selection gate and lexicographically beat long time-only base",
}

FROZEN_CHAMPION: dict[str, Any] = {
    "contexts": ["usdkrw_momentum", "dollar_flow_rel_4h_30d"],
    "tail": 0.20,
    "action_set": "tp4_time",
    "actions": [list(action) for action in ACTION_SETS["tp4_time"]],
    "risk_lambda": 0.25,
    "ridge": 100.0,
    "weight_mode": "uniform",
}


@dataclass(frozen=True)
class Config(base.Config):
    leverage: float = 0.45
    output: str = "results/minimal_stress_pair_context_formula_expert_selection_2026-07-15.json"
    manifest_output: str = "results/minimal_stress_pair_context_formula_expert_manifest_2026-07-15.json"
    docs_output: str = "docs/minimal-stress-pair-context-formula-expert-selection-2026-07-15.md"
    open_oos: bool = False


def _spec_hash() -> str:
    encoded = json.dumps(EXPERT_SPEC, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _frozen_execution_config(cfg: Config) -> dict[str, Any]:
    values = asdict(cfg)
    frozen = {key: values[key] for key in FROZEN_CONFIG_KEYS}
    for key in ("input_csv", "funding_csv", "premium_csv"):
        parts = Path(str(frozen[key])).parts
        if "data" in parts:
            parts = parts[parts.index("data") :]
        frozen[key] = Path(*parts).as_posix()
    return frozen


def formula_feature_frame(context: dict[str, Any]) -> pd.DataFrame:
    """Return prefix-causal formula features available at each decision."""

    cached = context.get("pair_formula_features")
    if cached is not None:
        return cached
    frame = add_vpin_formulaic_features(
        context["market"], add_a101(context["market"])
    )
    frame = (
        frame[CORE_FEATURES]
        .shift(1)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    context["pair_formula_features"] = frame
    return frame


def _standardize(
    raw: np.ndarray, fit_mask: np.ndarray
) -> tuple[np.ndarray, list[float], list[float]]:
    values = np.asarray(raw, dtype=float)
    fit = np.asarray(fit_mask, dtype=bool)
    mean = np.mean(values[fit], axis=0)
    scale = np.std(values[fit], axis=0)
    scale[scale < 1e-8] = 1.0
    output = np.clip((values - mean) / scale, -8.0, 8.0)
    return output, mean.tolist(), scale.tolist()


def pair_context_design(
    core: np.ndarray,
    source_signed: np.ndarray,
    contexts: dict[str, np.ndarray],
    fit_mask: np.ndarray,
    *,
    tail: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build two independent low/mid/high responsibility blocks."""

    if len(contexts) != 2:
        raise ValueError("exactly two context responsibilities are required")
    zcore, core_mean, core_scale = _standardize(core, fit_mask)
    source = np.asarray(source_signed, dtype=float)[:, None]
    blocks = [zcore, source, zcore * source]
    metadata: dict[str, Any] = {
        "core_mean": core_mean,
        "core_scale": core_scale,
        "contexts": {},
    }
    for name, raw_values in contexts.items():
        raw = np.asarray(raw_values, dtype=float)
        zcontext, context_mean, context_scale = _standardize(
            raw[:, None], fit_mask
        )
        low_threshold = float(np.quantile(raw[fit_mask], float(tail)))
        high_threshold = float(np.quantile(raw[fit_mask], 1.0 - float(tail)))
        low = (raw <= low_threshold).astype(float)[:, None]
        high = (raw >= high_threshold).astype(float)[:, None]
        blocks.extend(
            [
                zcontext,
                low,
                high,
                zcontext * source,
                low * source,
                high * source,
                zcore * low,
                zcore * high,
            ]
        )
        metadata["contexts"][name] = {
            "mean": context_mean[0],
            "scale": context_scale[0],
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
        }
    design = np.column_stack(blocks)
    metadata["dimensions"] = int(design.shape[1])
    return design, metadata


def _design_for_spec(
    context: dict[str, Any], spec: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    events = base.candidate_events(context)
    signals = events["signals"]
    fit_mask = base.fit_event_mask(context, events)
    key = (tuple(spec["contexts"]), float(spec["tail"]))
    cache = context.setdefault("pair_design_cache", {})
    if key not in cache:
        formula = formula_feature_frame(context)
        core = formula[CORE_FEATURES].iloc[signals].to_numpy(float)
        contexts = {
            name: context["features"][name].iloc[signals].to_numpy(float)
            for name in spec["contexts"]
        }
        cache[key] = pair_context_design(
            core,
            events["source_signed"],
            contexts,
            fit_mask,
            tail=float(spec["tail"]),
        )
    design, metadata = cache[key]
    return design, metadata, events


def fit_spec(
    context: dict[str, Any], spec: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    design, design_metadata, events = _design_for_spec(context, spec)
    fit_mask = base.fit_event_mask(context, events)
    if int(fit_mask.sum()) < 60:
        raise RuntimeError("insufficient purged fit events")
    actions = tuple(tuple(int(value) for value in row) for row in spec["actions"])
    target = base.fit_action_utilities(
        context, events, fit_mask, actions, float(spec["risk_lambda"])
    )
    signal_dates = context["dates"].iloc[events["signals"]].reset_index(drop=True)
    weights = base.event_weights(
        signal_dates,
        events["source_signed"],
        fit_mask,
        str(spec["weight_mode"]),
    )
    prediction, ridge_model = base.fit_weighted_action_ridge(
        design,
        target,
        fit_mask,
        weights,
        ridge=float(spec["ridge"]),
    )
    routed = base.actions_from_prediction(
        len(context["dates"]), events["signals"], prediction, actions
    )
    model = {
        "core_features": CORE_FEATURES,
        "context_features": list(spec["contexts"]),
        "design": design_metadata,
        "ridge_model": ridge_model,
        "fit_events": int(fit_mask.sum()),
        "fit_last_exit": str(
            context["dates"].iloc[events["max_exits"][fit_mask]].max()
        ),
        "weight_mode": str(spec["weight_mode"]),
    }
    return routed, model


def _grid(context: dict[str, Any], cfg: Config) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_score = _selection_score(context["base_stats"])
    event_signals = base.candidate_events(context)["signals"]
    for pair, tail, action_set, risk, ridge, weight_mode in itertools.product(
        itertools.combinations(PAIR_CONTEXTS, 2),
        TAIL_GRID,
        ACTION_SETS,
        RISK_GRID,
        RIDGE_GRID,
        WEIGHT_MODES,
    ):
        spec = {
            "contexts": list(pair),
            "tail": float(tail),
            "action_set": action_set,
            "actions": [list(action) for action in ACTION_SETS[action_set]],
            "risk_lambda": float(risk),
            "ridge": float(ridge),
            "weight_mode": weight_mode,
        }
        actions, model = fit_spec(context, spec)
        schedules, stats = base.schedules_and_stats(
            context, cfg, actions, windows=PRE2024_WINDOWS
        )
        score = _selection_score(stats)
        passed = selection_passes(stats)
        beats_base = bool(tuple(score) > tuple(base_score))
        rows.append(
            {
                "spec": spec,
                "dimensions": model["design"]["dimensions"],
                "context_thresholds": model["design"]["contexts"],
                "action_counts": base._action_counts(
                    actions, spec["actions"], event_signals
                ),
                "stable_action_hash": base._array_hash(
                    actions[base._stable_action_mask(context["dates"])]
                ),
                "selection_schedule_hashes": {
                    name: _schedule_hash(trades)
                    for name, trades in schedules.items()
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


def _selected_feature_hash(context: dict[str, Any]) -> str:
    events = base.candidate_events(context)
    source = np.zeros(len(context["dates"]), dtype=float)
    source[events["signals"]] = events["source_signed"]
    frame = formula_feature_frame(context).copy()
    for name in PAIR_CONTEXTS:
        frame[name] = context["features"][name].to_numpy(float)
    frame[base.EVENT_SOURCE_FEATURE] = source
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


def _implementation_hash() -> str:
    functions = (
        add_a101,
        add_vpin_formulaic_features,
        base._context,
        base.fit_rule_masks,
        base.minimal_stress_events,
        base.candidate_events,
        base.fit_event_mask,
        base.event_weights,
        base.trade_utility,
        base.fit_action_utilities,
        base.fit_weighted_action_ridge,
        base.actions_from_prediction,
        base.schedule_window,
        base.schedules_and_stats,
        base._execution_config,
        base._stable_action_mask,
        base._array_hash,
        formula_feature_frame,
        _standardize,
        pair_context_design,
        _design_for_spec,
        fit_spec,
        _grid,
        _selected_feature_hash,
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
        _frame_hash,
        _activation_hash,
        ExecutionEngine.trade_at,
        equity_stats,
    )
    source = "\n\n".join(inspect.getsource(function) for function in functions)
    constants = {
        "expert_spec": EXPERT_SPEC,
        "frozen_champion": FROZEN_CHAMPION,
        "frozen_config_keys": list(FROZEN_CONFIG_KEYS),
        "selection_end": SELECTION_END,
        "pre2024_windows": PRE2024_WINDOWS,
        "future_windows": FUTURE_WINDOWS,
    }
    encoded_constants = json.dumps(
        constants, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(f"{source}\n\n{encoded_constants}".encode()).hexdigest()


def _selection_payload(cfg: Config) -> dict[str, Any]:
    context, source_hashes = base._context(cfg, cutoff=SELECTION_END)
    if len(context["dates"]) and context["dates"].max() >= pd.Timestamp(SELECTION_END):
        raise RuntimeError("selection source was not physically truncated")
    grid = _grid(context, cfg)
    champion = grid[0]
    if champion["spec"] != FROZEN_CHAMPION:
        raise RuntimeError("frozen pair-context expert is no longer rank one")
    if not champion["accepted"]:
        raise RuntimeError("frozen pair-context expert no longer clears selection")
    actions, model = fit_spec(context, champion["spec"])
    schedules, stats = base.schedules_and_stats(
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
            "stable_action_hash": base._array_hash(
                actions[base._stable_action_mask(context["dates"])]
            ),
            "action_counts": champion["action_counts"],
            "score": champion["score"],
        },
        "selection_stats": stats,
        "selection_schedule_hashes": {
            name: _schedule_hash(trades) for name, trades in schedules.items()
        },
        "selection_grid": grid,
    }
    payload["freeze_hash"] = _freeze_hash(payload)
    return payload


def _validate_manifest(cfg: Config, manifest: dict[str, Any]) -> None:
    if manifest.get("oos_opened") is not False:
        raise RuntimeError("manifest must be pre-OOS")
    if manifest.get("spec_hash") != _spec_hash():
        raise RuntimeError("pair-context expert specification changed after freeze")
    if manifest.get("implementation_hash") != _implementation_hash():
        raise RuntimeError("pair-context expert implementation changed after freeze")
    if manifest.get("frozen_execution_config") != _frozen_execution_config(cfg):
        raise RuntimeError("execution configuration changed after freeze")
    if manifest.get("selected", {}).get("spec") != FROZEN_CHAMPION:
        raise RuntimeError("manifest champion mismatch")
    if manifest.get("freeze_hash") != _freeze_hash(manifest):
        raise RuntimeError("manifest freeze hash mismatch")


def _write_manifest_once(
    path: Path, payload: dict[str, Any], cfg: Config
) -> None:
    _validate_manifest(cfg, payload)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        _validate_manifest(cfg, existing)
        if existing["freeze_hash"] != payload["freeze_hash"]:
            raise RuntimeError("refusing to overwrite a different pair-context freeze")
        return
    base._atomic_write_json(path, payload)


def _metric(row: dict[str, Any]) -> str:
    return (
        f"{row['absolute_return_pct']:.2f}% / {row['cagr_pct']:.2f}% / "
        f"{row['strict_mdd_pct']:.2f}% / "
        f"{row['cagr_to_strict_mdd']:.2f} / {row['trades']}"
    )


def _write_docs(path: str, payload: dict[str, Any]) -> None:
    oos = bool(payload.get("oos_opened"))
    lines = [
        "# Minimal-stress pair-context formula expert OOS"
        if oos
        else "# Minimal-stress pair-context formula expert selection",
        "",
        "Metric: absolute return / CAGR / strict MDD / CAGR-MDD / trades.",
        "",
    ]
    if not oos:
        accepted = sum(row["accepted"] for row in payload["selection_grid"])
        base_stats = payload["base_stats"]
        selected = payload["selection_stats"]
        lines += [
            "## Verdict",
            "",
            "**Frozen for OOS.** Minimal funding/premium stress owns entry. "
            "Prior-bar Alpha101/VPIN formulas are routed by independent "
            "USD/KRW and dollar-flow context responsibilities; the critic chooses "
            "ABSTAIN or LONG/SHORT TP4/time.",
            "",
            f"Multiplicity: {len(payload['selection_grid'])} cells; {accepted} clear the absolute gate and beat the long time-only base.",
            "",
            "| Policy | Train | 2023 selection | Pre-2024 | Score |",
            "|---|---:|---:|---:|---:|",
            f"| Long time-only base | {_metric(base_stats['train'])} | {_metric(base_stats['select_2023'])} | {_metric(base_stats['pre_2024'])} | `{payload['base_score']}` |",
            f"| Pair-context expert | {_metric(selected['train'])} | {_metric(selected['select_2023'])} | {_metric(selected['pre_2024'])} | `{payload['selected']['score']}` |",
            "",
            "## Leakage controls",
            "",
            "- Selection market, funding, and premium sources are physically truncated before 2024.",
            "- Formula features are shifted one complete 5-minute bar; context fields use the audited live feature contract.",
            "- Context thresholds, scaling, and model fitting stop before 2023; every 48-hour utility label exits before 2023.",
            "- Trades enter next-open, pay 6bp/notional/side plus realized funding, remain split-contained, and use strict favorable-before-adverse MDD.",
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
            "The exact 640-cell search and champion were manifest-frozen before opening 2024+. The underlying stress event family has prior research exposure, so this is shadow OOS rather than pristine market OOS.",
        ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _selection(cfg: Config) -> dict[str, Any]:
    payload = _selection_payload(cfg)
    _write_manifest_once(Path(cfg.manifest_output), payload, cfg)
    base._atomic_write_json(Path(cfg.output), payload)
    _write_docs(cfg.docs_output, payload)
    return payload


def _oos_lock_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(manifest_path.name + ".oos-opening.json")


def _mark_oos_opened(
    path: Path, manifest: dict[str, Any], output: str
) -> dict[str, Any]:
    """Write or validate a resumable one-shot OOS lock."""

    expected = {
        "phase": "oos_opening",
        "freeze_hash": manifest["freeze_hash"],
        "spec_hash": manifest["spec_hash"],
        "implementation_hash": manifest["implementation_hash"],
        "oos_output": output,
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if existing.get(key) != value:
                raise RuntimeError("existing OOS lock belongs to a different freeze")
        return existing
    lock = {
        **expected,
        "oos_opened_at": datetime.now(timezone.utc).isoformat(),
    }
    base._atomic_write_json(path, lock)
    return lock


def _oos(cfg: Config) -> dict[str, Any]:
    manifest_path = Path(cfg.manifest_output)
    if not manifest_path.exists():
        raise FileNotFoundError("pre-2024 pair-context manifest is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest(cfg, manifest)
    prefix = _selection_payload(cfg)
    if prefix["freeze_hash"] != manifest["freeze_hash"]:
        raise RuntimeError("pre-2024 pair-context replay changed after freeze")
    lock_path = _oos_lock_path(manifest_path)
    opening = _mark_oos_opened(lock_path, manifest, cfg.output)

    context, full_source_hashes = base._context(cfg, cutoff=cfg.exclude_from)
    actions, model = fit_spec(context, manifest["selected"]["spec"])
    if model != manifest["selected"]["model"]:
        raise RuntimeError("full-run pair-context model differs from freeze")
    stable = base._stable_action_mask(context["dates"])
    if base._array_hash(actions[stable]) != manifest["selected"]["stable_action_hash"]:
        raise RuntimeError("full-run pair-context action prefix differs from freeze")
    schedules, stats = base.schedules_and_stats(
        context, cfg, actions, windows=FUTURE_WINDOWS
    )
    payload = {
        **manifest,
        "phase": "oos_result",
        "oos_opened": True,
        "oos_opened_at": opening["oos_opened_at"],
        "full_source_hashes": full_source_hashes,
        "oos_stats": stats,
        "oos_schedule_hashes": {
            name: _schedule_hash(trades) for name, trades in schedules.items()
        },
        "oos_passed": oos_passes(stats),
    }
    base._atomic_write_json(manifest_path, payload)
    base._atomic_write_json(Path(cfg.output), payload)
    _write_docs(cfg.docs_output, payload)
    lock_path.unlink(missing_ok=True)
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
