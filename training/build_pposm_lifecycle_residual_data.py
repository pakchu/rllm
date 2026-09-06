"""Build train-only lifecycle-aware residual PPOSM action data.

This dataset is stricter than the per-signal counterfactual residual set: every
candidate replacement is valued by replaying the full pre-2024 ALWAYS-TP4 route
stream with exactly one signal replaced by SKIP or TP12.  The prompt remains
signal-time only; full-route economics are offline label/reward metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import build_pposm_counterfactual_action_data as counterfactual
from training import build_pposm_sft_rlvr_data as frozen
from training import build_pposm_state_router_data as numeric
from training import build_pposm_symbolic_router_data as symbolic
from training import search_pullback_premium_overheat_state_machine_alpha as pposm
from training.audit_confirmed_pullback_squeeze_live_parity import (
    _activation_hash,
    _execution_config,
    _load_bundle,
)
from training.search_inventory_purge_reclaim_alpha import (
    ExecutionEngine,
    Trade,
    _schedule_hash,
    equity_stats,
)

DEFAULT_TRAIN_OUTPUT = Path(
    "data/pposm_lifecycle_residual_train_pre2024_2026-09-02.jsonl"
)
DEFAULT_SUMMARY_OUTPUT = Path(
    "results/pposm_lifecycle_residual_data_summary_2026-09-02.json"
)
TRAIN_WINDOW = ("pre_2024", "2020-07-01", "2024-01-01")
DEFAULT_ACTION = "TP4"
CANDIDATE_ACTIONS = ("SKIP", "TP12")
LABELS = ("KEEP", "SWITCH")
COSTS = {"base_6bp": 0.0006, "stress_10bp": 0.0010}
MDD_TOLERANCE_PP = 0.01
EPSILON = 1e-12


@dataclass(frozen=True)
class Config:
    manifest: Path = counterfactual.DEFAULT_MANIFEST
    train_output: Path = DEFAULT_TRAIN_OUTPUT
    summary_output: Path = DEFAULT_SUMMARY_OUTPUT


def canonical_json(value: Any) -> str:
    return counterfactual.canonical_json(value)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def lifecycle_identity(window: str, signal_position: int, candidate: str) -> str:
    if candidate not in CANDIDATE_ACTIONS:
        raise ValueError(f"candidate must be one of {CANDIDATE_ACTIONS}")
    return f"pposm-lifecycle-residual|{candidate}|{window}|{int(signal_position)}"


def lifecycle_prompt(
    features: dict[str, float], predicates: dict[str, bool], *, candidate: str
) -> str:
    if candidate not in CANDIDATE_ACTIONS:
        raise ValueError(f"candidate must be one of {CANDIDATE_ACTIONS}")
    return "\n".join(
        (
            "Frozen PPOSM lifecycle residual router.",
            "Default action is TP4. Decide whether to keep TP4 or switch only this signal to the candidate.",
            "Return exactly one token: KEEP or SWITCH.",
            f"candidate_action: {candidate}",
            f"default_action: {DEFAULT_ACTION}",
            f"causal_predicates: {canonical_json(predicates)}",
            f"signal_time_state: {canonical_json(features)}",
            "predicate_priority: capitulation predicates before premium-overheat predicates.",
        )
    )


def _apply_routes(
    engine: ExecutionEngine,
    signals: Sequence[int],
    routes: Sequence[str],
    *,
    start: str,
    end: str,
    spec: dict[str, Any],
) -> tuple[Trade, ...]:
    if len(signals) != len(routes):
        raise ValueError("signals and routes must have equal length")
    dates = pd.to_datetime(engine.market["date"])
    period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(
        bool
    )
    take_bps = {
        "TP4": int(spec["capitulation_take_bps"]),
        "TP12": int(spec["normal_take_bps"]),
    }
    trades: list[Trade] = []
    next_allowed = 0
    for signal, route in zip(signals, routes, strict=True):
        signal = int(signal)
        if route not in (*CANDIDATE_ACTIONS, DEFAULT_ACTION):
            raise ValueError(f"unsupported route: {route}")
        if signal < next_allowed or route == "SKIP":
            continue
        trade = engine.trade_at(
            signal,
            int(spec["side"]),
            int(spec["hold_bars"]),
            take_bps[route],
            int(spec["stop_bps"]),
        )
        if trade is None or not period[trade.exit_position]:
            continue
        trades.append(trade)
        next_allowed = trade.exit_position + 1
    if any(
        right.entry_position <= left.exit_position
        for left, right in zip(trades, trades[1:])
    ):
        raise RuntimeError("route simulation produced overlapping trades")
    return tuple(trades)


def _stats_by_cost(
    trades: Sequence[Trade], *, start: str, end: str, cfg: Any
) -> dict[str, dict[str, Any]]:
    return {
        name: equity_stats(trades, start=start, end=end, cfg=cfg, cost_rate=cost)
        for name, cost in COSTS.items()
    }


def _metric_delta(
    replacement: dict[str, Any], baseline: dict[str, Any], metric: str
) -> float:
    return float(replacement[metric]) - float(baseline[metric])


def lifecycle_deltas(
    replacement: dict[str, dict[str, Any]], baseline: dict[str, dict[str, Any]]
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for cost_name in COSTS:
        output[cost_name] = {
            "absolute_return_pct": _metric_delta(
                replacement[cost_name], baseline[cost_name], "absolute_return_pct"
            ),
            "cagr_pct": _metric_delta(
                replacement[cost_name], baseline[cost_name], "cagr_pct"
            ),
            "strict_mdd_pct": _metric_delta(
                replacement[cost_name], baseline[cost_name], "strict_mdd_pct"
            ),
            "cagr_to_strict_mdd": _metric_delta(
                replacement[cost_name], baseline[cost_name], "cagr_to_strict_mdd"
            ),
            "trades": _metric_delta(replacement[cost_name], baseline[cost_name], "trades"),
        }
    return output


def switch_utility_from_deltas(
    deltas: dict[str, dict[str, float]],
    replacement: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
) -> tuple[float, dict[str, bool], dict[str, float]]:
    base = deltas["base_6bp"]
    stress = deltas["stress_10bp"]
    delta_log_equity_base = float(
        np.log1p(float(replacement["base_6bp"]["absolute_return_pct"]) / 100.0)
        - np.log1p(float(baseline["base_6bp"]["absolute_return_pct"]) / 100.0)
    )
    delta_log_equity_stress = float(
        np.log1p(float(replacement["stress_10bp"]["absolute_return_pct"]) / 100.0)
        - np.log1p(float(baseline["stress_10bp"]["absolute_return_pct"]) / 100.0)
    )
    stress_mdd_delta_fraction = stress["strict_mdd_pct"] / 100.0
    risk_adjusted_delta = 0.5 * (
        base["cagr_to_strict_mdd"] + stress["cagr_to_strict_mdd"]
    )
    gates = {
        "base_log_equity_delta_nonnegative": delta_log_equity_base >= -EPSILON,
        "stress_log_equity_delta_positive": delta_log_equity_stress > EPSILON,
        "stress_mdd_delta_le_0_01pp": (
            stress_mdd_delta_fraction <= MDD_TOLERANCE_PP / 100.0 + EPSILON
        ),
    }
    components = {
        "delta_log_equity_base": delta_log_equity_base,
        "delta_log_equity_stress": delta_log_equity_stress,
        "stress_mdd_delta_fraction": stress_mdd_delta_fraction,
        "mean_base_stress_cagr_mdd_delta": risk_adjusted_delta,
    }
    if all(gates.values()):
        return float(risk_adjusted_delta), gates, components
    violation = (
        max(0.0, -delta_log_equity_base)
        + max(0.0, -delta_log_equity_stress)
        + max(0.0, stress_mdd_delta_fraction - MDD_TOLERANCE_PP / 100.0)
    )
    return -float(abs(risk_adjusted_delta) + violation + EPSILON), gates, components


def _replay_digest(signals: Sequence[int], routes: Sequence[str], trades: Sequence[Trade]) -> str:
    payload = {
        "signals": [int(value) for value in signals],
        "routes": list(routes),
        "trades": [
            {
                "signal": int(trade.signal_position),
                "entry": int(trade.entry_position),
                "exit": int(trade.exit_position),
            }
            for trade in trades
        ],
    }
    return _sha256(canonical_json(payload).encode("utf-8"))


def rows_from_train_context(
    market: pd.DataFrame,
    state: pd.DataFrame,
    active: Sequence[bool],
    manifest: dict[str, Any],
    strategy_cfg: Any,
    engine: ExecutionEngine,
) -> list[dict[str, Any]]:
    window, start, end = TRAIN_WINDOW
    dates = pd.to_datetime(market["date"])
    if len(dates) and dates.max() >= pd.Timestamp(end):
        raise RuntimeError("train-only lifecycle builder received OOS market rows")
    signals = tuple(
        int(value)
        for value in np.flatnonzero(
            np.asarray(active, dtype=bool)
            & ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
        )
    )
    baseline_routes = tuple(DEFAULT_ACTION for _ in signals)
    baseline_trades = _apply_routes(
        engine, signals, baseline_routes, start=start, end=end, spec=manifest["spec"]
    )
    baseline_stats = _stats_by_cost(
        baseline_trades, start=start, end=end, cfg=strategy_cfg
    )
    baseline_digest = _replay_digest(signals, baseline_routes, baseline_trades)
    baseline_schedule_hash = _schedule_hash(baseline_trades)
    admitted = {int(trade.signal_position) for trade in baseline_trades}

    rows: list[dict[str, Any]] = []
    for signal_index, signal in enumerate(signals):
        if signal not in admitted:
            continue
        features = numeric._signal_features(state, signal)
        predicates = symbolic.predicates(features, manifest["state_thresholds"])
        for candidate in CANDIDATE_ACTIONS:
            routes = list(baseline_routes)
            routes[signal_index] = candidate
            replacement_routes = tuple(routes)
            replacement_trades = _apply_routes(
                engine,
                signals,
                replacement_routes,
                start=start,
                end=end,
                spec=manifest["spec"],
            )
            replacement_stats = _stats_by_cost(
                replacement_trades, start=start, end=end, cfg=strategy_cfg
            )
            deltas = lifecycle_deltas(replacement_stats, baseline_stats)
            switch_utility, gates, utility_components = switch_utility_from_deltas(
                deltas, replacement_stats, baseline_stats
            )
            target = "SWITCH" if switch_utility > 0.0 else "KEEP"
            replacement_digest = _replay_digest(
                signals, replacement_routes, replacement_trades
            )
            signal_time = str(pd.Timestamp(market.iloc[signal]["date"]))
            rows.append(
                {
                    "task": "pposm_lifecycle_residual_action",
                    "split": "train",
                    "prompt": lifecycle_prompt(
                        features, predicates, candidate=candidate
                    ),
                    "target": target,
                    "metadata": {
                        "identity": lifecycle_identity(window, signal, candidate),
                        "base_identity": counterfactual.signal_identity(window, signal),
                        "window": window,
                        "signal_position": int(signal),
                        "signal_time": signal_time,
                        "default_action": DEFAULT_ACTION,
                        "candidate_action": candidate,
                        "residual_utilities": {
                            "KEEP": 0.0,
                            "SWITCH": switch_utility,
                        },
                        "switch_gates": gates,
                        "utility_components": utility_components,
                        "economic_deltas": deltas,
                        "portfolio_metrics": {
                            "baseline": baseline_stats,
                            "replacement": replacement_stats,
                        },
                        "reference_anchor": True,
                        "replacement_changed_lifecycle": _schedule_hash(replacement_trades)
                        != baseline_schedule_hash,
                        "baseline_schedule_hash": baseline_schedule_hash,
                        "replacement_schedule_hash": _schedule_hash(replacement_trades),
                        "baseline_replay_digest": baseline_digest,
                        "replacement_replay_digest": replacement_digest,
                        "entry_rule": "next_5m_open_if_lifecycle_admitted",
                        "lifecycle": "non_overlapping_TP_or_48h_cap_full_stream_replay",
                        "costs": COSTS,
                        "utility_source": "train_only_anchor_single_replacement_full_lifecycle_log_equity_mdd_ratio",
                        "offline_label_only": True,
                        "future_outcome_present_in_prompt": False,
                    },
                }
            )
    _validate_rows(rows)
    return rows


def _validate_rows(rows: Sequence[dict[str, Any]]) -> None:
    identities = [str(row["metadata"]["identity"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate lifecycle residual identity")
    by_signal: dict[int, list[str]] = {}
    for row in rows:
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        by_signal.setdefault(int(meta["signal_position"]), []).append(
            str(meta["candidate_action"])
        )
        if row.get("split") != "train":
            raise ValueError("lifecycle residual builder is train-only")
        if row.get("target") not in LABELS:
            raise ValueError("target must be KEEP or SWITCH")
        prompt = str(row.get("prompt", ""))
        forbidden = ("economic_delta", "residual_util", "return_pct", "mdd")
        if any(token in prompt for token in forbidden):
            raise ValueError("prompt contains offline outcome metadata")
    for signal, candidates in by_signal.items():
        if tuple(sorted(candidates)) != tuple(sorted(CANDIDATE_ACTIONS)):
            raise ValueError(f"signal {signal} lacks exactly one SKIP and TP12 pair")


def _jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join((canonical_json(row) + "\n").encode("utf-8") for row in rows)


def load_train_context(
    manifest: dict[str, Any], strategy_cfg: pposm.Config
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, ExecutionEngine]:
    selection_dates = frozen._validate_pre2024(manifest, strategy_cfg)
    market, raw_features, funding, source_hashes = _load_bundle(
        strategy_cfg,
        cutoff=pposm.SELECTION_END,
        premium_tolerance=strategy_cfg.live_premium_tolerance,
    )
    if source_hashes != manifest["source_prefix_hashes"]:
        raise RuntimeError("train-only source prefix hashes changed")
    dates, features, active, base_thresholds = frozen._state_inputs(
        market, raw_features, strategy_cfg
    )
    if not dates.reset_index(drop=True).equals(selection_dates.reset_index(drop=True)):
        raise RuntimeError("train-only replay changed the pre-2024 clock")
    if base_thresholds != manifest["base_thresholds"]:
        raise RuntimeError("train-only base thresholds changed")
    if pposm.feature_hash(features) != manifest["feature_prefix_hash"]:
        raise RuntimeError("train-only state feature hash changed")
    capitulation, overheat = pposm.build_state_masks(
        features, manifest["state_thresholds"], pposm.FROZEN_CHAMPION["overheat"]
    )
    observed = (
        _activation_hash(active, dates),
        _activation_hash(capitulation, dates),
        _activation_hash(overheat, dates),
    )
    expected = (
        manifest["activation_hash"],
        manifest["capitulation_hash"],
        manifest["overheat_hash"],
    )
    if observed != expected:
        raise RuntimeError("train-only state activation hashes changed")
    state = features.copy()
    state["capitulation"] = capitulation
    state["overheat"] = overheat
    engine = ExecutionEngine(
        market, funding, _execution_config(strategy_cfg, strategy_cfg.leverage)
    )
    return market, funding, state, np.asarray(active, dtype=bool), engine


def build(cfg: Config) -> dict[str, Any]:
    manifest, strategy_cfg = frozen.load_frozen_manifest(cfg.manifest)
    market, _, state, active, engine = load_train_context(manifest, strategy_cfg)
    rows = rows_from_train_context(market, state, active, manifest, strategy_cfg, engine)
    train_bytes = _jsonl_bytes(rows)
    cfg.train_output.parent.mkdir(parents=True, exist_ok=True)
    cfg.train_output.write_bytes(train_bytes)
    train_pre2024 = all(
        pd.Timestamp(row["metadata"]["signal_time"]).tz_localize(None)
        < pd.Timestamp("2024-01-01")
        for row in rows
    )
    summary = {
        "protocol": "pposm_train_only_lifecycle_residual_action_v1",
        "config": {key: str(value) for key, value in asdict(cfg).items()},
        "manifest_freeze_hash": manifest["freeze_hash"],
        "window": {"name": TRAIN_WINDOW[0], "start": TRAIN_WINDOW[1], "end_exclusive": TRAIN_WINDOW[2]},
        "default_action": DEFAULT_ACTION,
        "candidate_actions": list(CANDIDATE_ACTIONS),
        "labels": list(LABELS),
        "rows": {"train": len(rows), "oos": 0, "total": len(rows)},
        "signals": {"train": len(rows) // len(CANDIDATE_ACTIONS), "oos": 0},
        "targets": dict(sorted(Counter(row["target"] for row in rows).items())),
        "candidate_targets": dict(
            sorted(
                Counter(
                    f"{row['metadata']['candidate_action']}={row['target']}"
                    for row in rows
                ).items()
            )
        ),
        "reference_anchor_pairs": int(
            sum(bool(row["metadata"]["reference_anchor"]) for row in rows)
        ),
        "reference_anchors": len(rows) // len(CANDIDATE_ACTIONS),
        "lifecycle_changed_pairs": int(
            sum(bool(row["metadata"]["replacement_changed_lifecycle"]) for row in rows)
        ),
        "output_sha256": {"train": _sha256(train_bytes)},
        "identity_sha256": _sha256(
            "\n".join(row["metadata"]["identity"] for row in rows).encode("utf-8")
        ),
        "median_nonzero_absolute_switch_utility": float(
            np.median(
                [
                    abs(float(row["metadata"]["residual_utilities"]["SWITCH"]))
                    for row in rows
                    if float(row["metadata"]["residual_utilities"]["SWITCH"]) != 0.0
                ]
            )
        ),
        "utility_contract": {
            "risk_adjusted_delta": "0.5 * (base CAGR/MDD delta + stress CAGR/MDD delta)",
            "positive_gate": [
                "base log-equity delta >= -1e-12",
                "stress log-equity delta > 1e-12",
                "stress strict-MDD delta <= 0.0001 fraction plus 1e-12",
            ],
            "failed_gate_utility": "-(abs(risk_adjusted_delta) + log-equity/MDD violation + 1e-12)",
            "target": "SWITCH iff lifecycle residual utility > 0; otherwise KEEP",
        },
        "causality": {
            "train_signal_time_pre2024_only": train_pre2024,
            "default_mode_train_only": True,
            "oos_rows_written": False,
            "future_outcome_in_prompt": False,
            "utilities_are_offline_train_labels_only": True,
            "single_replacement_full_lifecycle_replay": True,
            "base_cost_bps_per_side": 6,
            "stress_cost_bps_per_side": 10,
        },
    }
    cfg.summary_output.parent.mkdir(parents=True, exist_ok=True)
    cfg.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=counterfactual.DEFAULT_MANIFEST)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(Config(**vars(parse_args()))), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
