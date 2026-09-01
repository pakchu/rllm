"""Select train-only two-specialist lifecycle thresholds for PPOSM.

This module consumes separately scored SKIP and TP12 lifecycle specialist rows.
It never consumes OOS rows: thresholds are searched on the frozen pre-2024
lifecycle train stream only, then written as a frozen artifact if the same
materiality/economic gates used by the lifecycle residual selector pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import build_pposm_counterfactual_action_data as counterfactual
from training import build_pposm_lifecycle_residual_data as lifecycle
from training.audit_confirmed_pullback_squeeze_live_parity import _execution_config
from training.search_inventory_purge_reclaim_alpha import ExecutionEngine, Trade, equity_stats

DEFAULT_SKIP_TRAIN_SCORES = Path("results/pposm_lifecycle_residual_skip_train_scores_2026-09-02.jsonl")
DEFAULT_TP12_TRAIN_SCORES = Path("results/pposm_lifecycle_residual_tp12_train_scores_2026-09-02.jsonl")
DEFAULT_TRAIN_DATA = lifecycle.DEFAULT_TRAIN_OUTPUT
DEFAULT_DATA_SUMMARY = lifecycle.DEFAULT_SUMMARY_OUTPUT
DEFAULT_THRESHOLD_OUTPUT = Path("results/pposm_lifecycle_specialist_train_pair_threshold_2026-09-02.json")
DEFAULT_FAILURE_OUTPUT = Path("results/pposm_lifecycle_specialist_train_pair_threshold_failure_2026-09-02.json")
DEFAULT_PREREGISTRATION = Path(
    "results/pposm_lifecycle_specialist_sft_rlvr_preregistration_2026-09-02.json"
)
DEFAULT_EXPECTED_IDENTITY_SHA256 = "d0d2578ee463b2282915933afdcbc168a4178efe584a98886dbabc7099cdf8c2"
TRAIN_WINDOW = lifecycle.TRAIN_WINDOW
DEFAULT_ACTION = lifecycle.DEFAULT_ACTION
CANDIDATE_ACTIONS = lifecycle.CANDIDATE_ACTIONS
ALLOWED_ROUTES = (*CANDIDATE_ACTIONS, DEFAULT_ACTION)
EPSILON = 1e-12


@dataclass(frozen=True)
class Config:
    manifest: Path = counterfactual.DEFAULT_MANIFEST
    train_data: Path = DEFAULT_TRAIN_DATA
    data_summary: Path = DEFAULT_DATA_SUMMARY
    skip_train_scores: Path = DEFAULT_SKIP_TRAIN_SCORES
    tp12_train_scores: Path = DEFAULT_TP12_TRAIN_SCORES
    threshold_output: Path = DEFAULT_THRESHOLD_OUTPUT
    failure_output: Path = DEFAULT_FAILURE_OUTPUT
    preregistration: Path = DEFAULT_PREREGISTRATION
    expected_identity_sha256: str = DEFAULT_EXPECTED_IDENTITY_SHA256


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"line {number} in {path} is not an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def _identity_sha256(rows: Sequence[dict[str, Any]]) -> str:
    return _sha256_bytes("\n".join(str(row["identity"]) for row in rows).encode("utf-8"))


def _time_pre2024(value: Any) -> bool:
    ts = pd.Timestamp(str(value)).tz_localize(None)
    return ts < pd.Timestamp(TRAIN_WINDOW[2])


def _score_margin(row: dict[str, Any]) -> float:
    if "switch_margin" in row:
        margin = float(row["switch_margin"])
    else:
        scores = row.get("scores")
        if not isinstance(scores, dict) or "SWITCH" not in scores or "KEEP" not in scores:
            raise ValueError("score row must contain switch_margin or KEEP/SWITCH scores")
        margin = float(scores["SWITCH"]) - float(scores["KEEP"])
    if not math.isfinite(margin):
        raise ValueError("switch margin must be finite")
    return margin


def _signal_position(row: dict[str, Any]) -> int:
    raw = row.get("signal_position", row.get("signal_pos"))
    if raw is None:
        raise ValueError("score row missing signal_position/signal_pos")
    return int(raw)


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def validate_preregistration(preregistration: dict[str, Any]) -> dict[str, Any]:
    if (
        preregistration.get("protocol_version")
        != "pposm_lifecycle_two_specialist_sft_rlvr_v1"
        or preregistration.get("status")
        != "preregistered_before_specialist_training_and_oos"
    ):
        raise ValueError("specialist preregistration protocol/status changed")
    source = preregistration.get("source")
    if not isinstance(source, dict):
        raise ValueError("specialist preregistration lacks source bindings")
    split_binding = source.get("split_summary")
    specialist_bindings = source.get("specialists")
    if not isinstance(split_binding, dict) or not isinstance(specialist_bindings, dict):
        raise ValueError("specialist preregistration lacks split/source bindings")

    split_path = Path(str(split_binding.get("path", "")))
    if not split_path.is_file() or _sha256_file(split_path) != split_binding.get("sha256"):
        raise ValueError("specialist split summary hash/path changed")
    split_summary = _load_json(split_path)
    expected_model = preregistration.get("base_model", {}).get("name")
    if expected_model != "Qwen/Qwen2.5-1.5B-Instruct":
        raise ValueError("specialist base model changed")

    validated_specialists: dict[str, Any] = {}
    for candidate in CANDIDATE_ACTIONS:
        binding = specialist_bindings.get(candidate)
        output = split_summary.get("outputs", {}).get(candidate)
        if not isinstance(binding, dict) or not isinstance(output, dict):
            raise ValueError(f"missing {candidate} preregistration/split binding")
        data_path = Path(str(binding.get("path", "")))
        if not data_path.is_file() or _sha256_file(data_path) != binding.get("sha256"):
            raise ValueError(f"{candidate} specialist source hash/path changed")
        expected = {
            "sha256": binding.get("sha256"),
            "identity_sha256": binding.get("identity_sha256"),
            "rows": binding.get("rows"),
            "target_counts": binding.get("targets"),
        }
        observed = {
            "sha256": output.get("sha256"),
            "identity_sha256": output.get("identity_sha256"),
            "rows": output.get("rows"),
            "target_counts": output.get("target_counts"),
        }
        if observed != expected:
            raise ValueError(f"{candidate} split summary differs from preregistration")
        validated_specialists[candidate] = {
            "path": str(data_path),
            **expected,
        }
    if (
        source.get("lifecycle_train_identity_sha256")
        != DEFAULT_EXPECTED_IDENTITY_SHA256
        or split_summary.get("identity_sha256") != DEFAULT_EXPECTED_IDENTITY_SHA256
    ):
        raise ValueError("specialist lifecycle train identity binding changed")
    return {
        "manifest_freeze_hash": source.get("manifest_freeze_hash"),
        "base_model": expected_model,
        "split_summary": {
            "path": str(split_path),
            "sha256": split_binding.get("sha256"),
        },
        "specialists": validated_specialists,
    }


def _validate_candidate_score_rows(
    rows: Sequence[dict[str, Any]],
    *,
    candidate: str,
    source_binding: dict[str, Any] | None = None,
    expected_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    expected_count: int = 102,
) -> dict[str, Any]:
    if candidate not in CANDIDATE_ACTIONS:
        raise ValueError("candidate must be SKIP or TP12")
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} {candidate} specialist score rows, observed {len(rows)}")
    identities = [row.get("identity") for row in rows]
    if not all(isinstance(identity, str) and identity for identity in identities):
        raise ValueError("every score row must include non-empty identity")
    if len(set(identities)) != len(identities):
        raise ValueError(f"duplicate {candidate} score row identity")
    seen_bases: set[str] = set()
    adapter_hashes: set[str] = set()
    model_names: set[str] = set()
    source_hashes: set[str] = set()
    source_identity_hashes: set[str] = set()
    target_counts: Counter[str] = Counter()
    for row in rows:
        if row.get("split") != "train" or row.get("window") != TRAIN_WINDOW[0]:
            raise ValueError(f"{candidate} specialist rows must be train/pre_2024 only")
        if str(row.get("candidate_action")) != candidate:
            raise ValueError(f"{candidate} specialist file contains another candidate_action")
        if not _time_pre2024(row.get("signal_time", row.get("date"))):
            raise ValueError(f"{candidate} specialist signal_time/date must be pre-2024")
        signal = _signal_position(row)
        expected_identity = lifecycle.lifecycle_identity(TRAIN_WINDOW[0], signal, candidate)
        if str(row["identity"]) != expected_identity:
            raise ValueError(f"{candidate} identity does not match candidate/window/signal_position")
        expected_base = counterfactual.signal_identity(TRAIN_WINDOW[0], signal)
        if str(row.get("base_identity")) != expected_base:
            raise ValueError(f"{candidate} base_identity does not match pre_2024 signal_position")
        if expected_base in seen_bases:
            raise ValueError(f"duplicate {candidate} base_identity")
        seen_bases.add(expected_base)
        adapter_sha256 = str(row.get("adapter_sha256", ""))
        if len(adapter_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in adapter_sha256.lower()
        ):
            raise ValueError(f"{candidate} score row lacks a valid adapter_sha256")
        adapter_hashes.add(adapter_sha256.lower())
        model_name = str(row.get("model_name", ""))
        if not model_name:
            raise ValueError(f"{candidate} score row lacks model_name")
        model_names.add(model_name)
        if row.get("score_normalization") != "mean":
            raise ValueError(f"{candidate} scores must use mean label-logprob normalization")
        source_hashes.add(str(row.get("source_jsonl_sha256", "")))
        source_identity_hashes.add(str(row.get("source_identity_sha256", "")))
        target_counts[str(row.get("target"))] += 1
        _score_margin(row)
    if (
        len(adapter_hashes) != 1
        or model_names != {expected_model}
        or len(source_hashes) != 1
        or len(source_identity_hashes) != 1
    ):
        raise ValueError(f"{candidate} score rows mix adapter, model, or source provenance")
    if source_binding is not None:
        if source_hashes != {str(source_binding.get("sha256"))}:
            raise ValueError(f"{candidate} score source hash differs from preregistration")
        if source_identity_hashes != {str(source_binding.get("identity_sha256"))}:
            raise ValueError(f"{candidate} score source identity differs from preregistration")
        if dict(sorted(target_counts.items())) != source_binding.get("target_counts"):
            raise ValueError(f"{candidate} score targets differ from preregistration")
    return {
        "rows": len(rows),
        "candidate": candidate,
        "identity_sha256": _identity_sha256(rows),
        "adapter_sha256": next(iter(adapter_hashes)),
        "model_name": next(iter(model_names)),
        "score_normalization": "mean",
        "source_jsonl_sha256": next(iter(source_hashes)),
        "source_identity_sha256": next(iter(source_identity_hashes)),
        "target_counts": dict(sorted(target_counts.items())),
    }


def _train_data_pairs(train_data_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for row in train_data_rows:
        meta = row.get("metadata")
        if not isinstance(meta, dict):
            raise ValueError("train data row missing metadata")
        pairs.append(
            {
                "identity": meta.get("identity"),
                "base_identity": meta.get("base_identity"),
                "candidate_action": meta.get("candidate_action"),
                "signal_position": meta.get("signal_position"),
                "signal_time": meta.get("signal_time"),
                "split": row.get("split"),
                "window": meta.get("window"),
            }
        )
    return pairs


def bind_specialist_scores_to_train_data(
    train_data_rows: Sequence[dict[str, Any]],
    skip_rows: Sequence[dict[str, Any]],
    tp12_rows: Sequence[dict[str, Any]],
    *,
    expected_identity_sha256: str = DEFAULT_EXPECTED_IDENTITY_SHA256,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return combined specialist rows in frozen train-data identity order."""

    _validate_candidate_score_rows(skip_rows, candidate="SKIP")
    _validate_candidate_score_rows(tp12_rows, candidate="TP12")
    score_by_identity = {str(row["identity"]): dict(row) for row in (*skip_rows, *tp12_rows)}
    if len(score_by_identity) != len(skip_rows) + len(tp12_rows):
        raise ValueError("duplicate identity across specialist score files")
    data_pairs = _train_data_pairs(train_data_rows)
    if len(data_pairs) != 204:
        raise ValueError(f"expected 204 frozen lifecycle train rows, observed {len(data_pairs)}")
    data_hash = _identity_sha256(data_pairs)
    if expected_identity_sha256 and data_hash != expected_identity_sha256:
        raise ValueError(f"train data identity hash mismatch: expected {expected_identity_sha256}, observed {data_hash}")
    combined: list[dict[str, Any]] = []
    for pair in data_pairs:
        identity = str(pair["identity"])
        score = score_by_identity.get(identity)
        if score is None:
            raise ValueError(f"missing specialist score for frozen train identity {identity}")
        for key in ("base_identity", "candidate_action"):
            if str(score.get(key)) != str(pair.get(key)):
                raise ValueError(f"specialist score {identity} does not match train-data {key}")
        if _signal_position(score) != int(pair["signal_position"]):
            raise ValueError(f"specialist score {identity} does not match train-data signal_position")
        if str(score.get("signal_time", score.get("date"))) != str(pair["signal_time"]):
            # Timestamp stringification may differ only by a zero time component; compare normalized time too.
            if pd.Timestamp(str(score.get("signal_time", score.get("date")))).tz_localize(None) != pd.Timestamp(str(pair["signal_time"])).tz_localize(None):
                raise ValueError(f"specialist score {identity} does not match train-data signal_time")
        combined.append(score)
    combined_hash = _identity_sha256(combined)
    if combined_hash != data_hash:
        raise ValueError("combined specialist score order does not match frozen train-data identity order")
    return combined, {
        "train_data_rows": len(train_data_rows),
        "combined_score_rows": len(combined),
        "identity_sha256": combined_hash,
    }


def validate_data_summary(summary: dict[str, Any], *, expected_identity_sha256: str, train_data_sha256: str) -> dict[str, Any]:
    expected = {
        "identity_sha256": expected_identity_sha256,
        "train_data_sha256": train_data_sha256,
        "train_rows": 204,
        "oos_rows": 0,
        "reference_anchors": 102,
        "reference_anchor_pairs": 204,
    }
    observed = {
        "identity_sha256": summary.get("identity_sha256"),
        "train_data_sha256": summary.get("output_sha256", {}).get("train"),
        "train_rows": summary.get("rows", {}).get("train"),
        "oos_rows": summary.get("rows", {}).get("oos"),
        "reference_anchors": summary.get("reference_anchors"),
        "reference_anchor_pairs": summary.get("reference_anchor_pairs"),
    }
    if observed != expected:
        raise ValueError(f"lifecycle data summary does not bind the frozen train data: {observed}")
    return observed


def _full_pre2024_signals(market: pd.DataFrame, active: Sequence[bool]) -> tuple[int, ...]:
    _window, start, end = TRAIN_WINDOW
    dates = pd.to_datetime(market["date"])
    mask = np.asarray(active, dtype=bool)
    period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    return tuple(int(value) for value in np.flatnonzero(mask & period))


def _apply_routes(engine: ExecutionEngine, signals: Sequence[int], routes: Sequence[str], *, start: str, end: str, spec: dict[str, Any]) -> tuple[Trade, ...]:
    if len(signals) != len(routes):
        raise ValueError("signals and routes length mismatch")
    dates = pd.to_datetime(engine.market["date"])
    period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    take_bps = {"TP4": int(spec["capitulation_take_bps"]), "TP12": int(spec["normal_take_bps"])}
    trades: list[Trade] = []
    next_allowed = 0
    for signal, route in zip(signals, routes, strict=True):
        if route not in ALLOWED_ROUTES:
            raise ValueError(f"unsupported route: {route}")
        signal = int(signal)
        if signal < next_allowed or route == "SKIP":
            continue
        trade = engine.trade_at(signal, int(spec["side"]), int(spec["hold_bars"]), take_bps[route], int(spec["stop_bps"]))
        if trade is None or not period[trade.exit_position]:
            continue
        trades.append(trade)
        next_allowed = trade.exit_position + 1
    if any(right.entry_position <= left.exit_position for left, right in zip(trades, trades[1:])):
        raise RuntimeError("route simulation produced overlapping trades")
    return tuple(trades)


def _economics(trades: Sequence[Trade], *, start: str, end: str, cfg: Any) -> dict[str, dict[str, Any]]:
    return {name: equity_stats(trades, start=start, end=end, cfg=cfg, cost_rate=cost) for name, cost in lifecycle.COSTS.items()}


def _log_equity_delta(predicted: dict[str, Any], control: dict[str, Any]) -> float:
    return float(math.log1p(float(predicted["absolute_return_pct"]) / 100.0) - math.log1p(float(control["absolute_return_pct"]) / 100.0))


def _metric_deltas(predicted: dict[str, dict[str, Any]], control: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for cost in lifecycle.COSTS:
        output[cost] = {
            "absolute_return_pct": float(predicted[cost]["absolute_return_pct"]) - float(control[cost]["absolute_return_pct"]),
            "cagr_to_strict_mdd": float(predicted[cost]["cagr_to_strict_mdd"]) - float(control[cost]["cagr_to_strict_mdd"]),
            "strict_mdd_pct": float(predicted[cost]["strict_mdd_pct"]) - float(control[cost]["strict_mdd_pct"]),
            "log_equity": _log_equity_delta(predicted[cost], control[cost]),
            "trades": float(predicted[cost]["trades"]) - float(control[cost]["trades"]),
        }
    return output


def _threshold_candidates(margins: Sequence[float]) -> list[float]:
    finite = sorted({float(value) for value in margins if math.isfinite(float(value))})
    if not finite:
        raise ValueError("no finite score margins")
    return [math.nextafter(finite[0], -math.inf), *finite, math.nextafter(finite[-1], math.inf)]


def _pair_threshold_candidates(score_rows: Sequence[dict[str, Any]]) -> tuple[list[float], list[float]]:
    by_candidate: dict[str, list[float]] = {action: [] for action in CANDIDATE_ACTIONS}
    for row in score_rows:
        by_candidate[str(row["candidate_action"])].append(_score_margin(row))
    return (_threshold_candidates(by_candidate["SKIP"]), _threshold_candidates(by_candidate["TP12"]))


def _anchor_routes_for_thresholds(score_rows: Sequence[dict[str, Any]], thresholds: dict[str, float]) -> dict[int, dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        grouped[_signal_position(row)].append(row)
    selected: dict[int, dict[str, Any]] = {}
    for signal, rows in grouped.items():
        if tuple(sorted(str(row["candidate_action"]) for row in rows)) != tuple(sorted(CANDIDATE_ACTIONS)):
            raise ValueError(f"signal {signal} lacks exactly one SKIP and TP12 specialist row")
        margins = {str(row["candidate_action"]): _score_margin(row) for row in rows}
        excess = {action: margins[action] - float(thresholds[action]) for action in CANDIDATE_ACTIONS}
        eligible = [action for action in CANDIDATE_ACTIONS if excess[action] > 0.0]
        if eligible:
            route = max(eligible, key=lambda action: (excess[action], 0 if action == "SKIP" else -1))
            decision = "SWITCH"
            selected_margin = margins[route]
            selected_excess = excess[route]
        else:
            route = DEFAULT_ACTION
            decision = "KEEP"
            selected_margin = max(margins.values())
            selected_excess = max(excess.values())
        selected[signal] = {
            "route": route,
            "decision": decision,
            "selected_margin": float(selected_margin),
            "selected_normalized_excess_margin": float(selected_excess),
            "candidate_margins": dict(sorted(margins.items())),
            "candidate_normalized_excess_margins": dict(sorted((k, float(v)) for k, v in excess.items())),
        }
    return selected


def route_stream_for_thresholds(score_rows: Sequence[dict[str, Any]], full_signals: Sequence[int], thresholds: dict[str, float]) -> tuple[str, ...]:
    anchor_routes = _anchor_routes_for_thresholds(score_rows, thresholds)
    return tuple(anchor_routes.get(int(signal), {"route": DEFAULT_ACTION})["route"] for signal in full_signals)


def _materiality(routes: Sequence[str]) -> dict[str, Any]:
    counts = Counter(routes)
    total = len(routes)
    non_default_counts = {action: int(counts.get(action, 0)) for action in CANDIDATE_ACTIONS}
    non_default_total = sum(non_default_counts.values())
    used_nondefault_counts = [count for count in non_default_counts.values() if count > 0]
    return {
        "signals": total,
        "route_counts": dict(sorted(counts.items())),
        "non_default_count": non_default_total,
        "difference_rate_vs_always_tp4": non_default_total / total if total else 0.0,
        "non_default_counts": non_default_counts,
        "max_action_share": max(counts.values(), default=0) / total if total else 0.0,
        "passes": bool(total > 0 and non_default_total / total >= 0.10 and all(count >= 10 for count in used_nondefault_counts) and max(counts.values(), default=0) / total <= 0.90),
    }


def feasibility(economics: dict[str, dict[str, Any]], control: dict[str, dict[str, Any]], routes: Sequence[str]) -> dict[str, Any]:
    deltas = _metric_deltas(economics, control)
    materiality = _materiality(routes)
    gates = {
        "materiality": bool(materiality["passes"]),
        "base_return_ge_control": float(deltas["base_6bp"]["absolute_return_pct"]) >= -EPSILON,
        "stress_return_gt_control": float(deltas["stress_10bp"]["absolute_return_pct"]) > EPSILON,
        "base_ratio_lift_ge_0_05": float(deltas["base_6bp"]["cagr_to_strict_mdd"]) >= 0.05 - EPSILON,
        "stress_mdd_worsening_le_0_01pp": float(deltas["stress_10bp"]["strict_mdd_pct"]) <= 0.01 + EPSILON,
        "base_mdd_le_15": float(economics["base_6bp"]["strict_mdd_pct"]) <= 15.0 + EPSILON,
        "base_ratio_ge_3": float(economics["base_6bp"]["cagr_to_strict_mdd"]) >= 3.0 - EPSILON,
        "trades_ge_40": int(economics["base_6bp"]["trades"]) >= 40,
    }
    return {
        "passes": all(gates.values()),
        "gates": gates,
        "materiality": materiality,
        "deltas": deltas,
        "ranking": {
            "min_ratio_delta": min(deltas["base_6bp"]["cagr_to_strict_mdd"], deltas["stress_10bp"]["cagr_to_strict_mdd"]),
            "min_log_equity_delta": min(deltas["base_6bp"]["log_equity"], deltas["stress_10bp"]["log_equity"]),
        },
    }


def _stream_hash(signals: Sequence[int], routes: Sequence[str]) -> str:
    return _sha256_bytes(canonical_json({"signals": list(map(int, signals)), "routes": list(routes)}).encode("utf-8"))


def _trade_schedule_hash(trades: Sequence[Trade]) -> str:
    payload = [{"signal": int(t.signal_position), "entry": int(t.entry_position), "exit": int(t.exit_position)} for t in trades]
    return _sha256_bytes(canonical_json(payload).encode("utf-8"))


def evaluate_thresholds(*, score_rows: Sequence[dict[str, Any]], full_signals: Sequence[int], engine: ExecutionEngine, strategy_cfg: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    _window, start, end = TRAIN_WINDOW
    baseline_routes = tuple(DEFAULT_ACTION for _ in full_signals)
    baseline_trades = _apply_routes(engine, full_signals, baseline_routes, start=start, end=end, spec=manifest["spec"])
    baseline_economics = _economics(baseline_trades, start=start, end=end, cfg=strategy_cfg)
    skip_thresholds, tp12_thresholds = _pair_threshold_candidates(score_rows)
    evaluations: list[dict[str, Any]] = []
    feasible: list[dict[str, Any]] = []
    for skip_threshold in skip_thresholds:
        for tp12_threshold in tp12_thresholds:
            thresholds = {"SKIP": float(skip_threshold), "TP12": float(tp12_threshold)}
            routes = route_stream_for_thresholds(score_rows, full_signals, thresholds)
            trades = _apply_routes(engine, full_signals, routes, start=start, end=end, spec=manifest["spec"])
            economics = _economics(trades, start=start, end=end, cfg=strategy_cfg)
            feas = feasibility(economics, baseline_economics, routes)
            item = {
                "thresholds": thresholds,
                "route_stream_sha256": _stream_hash(full_signals, routes),
                "route_stream": list(routes),
                "trade_schedule_sha256": _trade_schedule_hash(trades),
                "trades": len(trades),
                "economics": economics,
                "control_economics": baseline_economics,
                "feasibility": feas,
            }
            evaluations.append(item)
            if feas["passes"]:
                feasible.append(item)
    if not feasible:
        return {
            "status": "no_feasible_train_pair_threshold",
            "candidate_thresholds": len(evaluations),
            "candidate_thresholds_by_specialist": {"SKIP": len(skip_thresholds), "TP12": len(tp12_thresholds)},
            "control_economics": baseline_economics,
            "evaluations": evaluations,
        }
    selected = max(
        feasible,
        key=lambda item: (
            float(item["feasibility"]["ranking"]["min_ratio_delta"]),
            float(item["feasibility"]["ranking"]["min_log_equity_delta"]),
            float(item["thresholds"]["SKIP"]),
            float(item["thresholds"]["TP12"]),
            -int(item["route_stream_sha256"], 16),
        ),
    )
    return {
        "status": "feasible_train_pair_threshold",
        "candidate_thresholds": len(evaluations),
        "candidate_thresholds_by_specialist": {"SKIP": len(skip_thresholds), "TP12": len(tp12_thresholds)},
        "control_economics": baseline_economics,
        "selected": selected,
        "evaluations": evaluations,
    }


def _failure_diagnostics(evaluations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not evaluations:
        return {}
    best = max(
        evaluations,
        key=lambda item: (
            float(item["feasibility"]["ranking"]["min_ratio_delta"]),
            float(item["feasibility"]["ranking"]["min_log_equity_delta"]),
            float(item["thresholds"]["SKIP"]),
            float(item["thresholds"]["TP12"]),
        ),
    )
    return {
        "best_thresholds_by_ranking_even_if_infeasible": dict(best["thresholds"]),
        "route_stream_sha256": best["route_stream_sha256"],
        "failed_gates": [key for key, ok in best["feasibility"]["gates"].items() if not ok],
        "materiality": best["feasibility"]["materiality"],
        "deltas": best["feasibility"]["deltas"],
    }


def run(cfg: Config) -> dict[str, Any]:
    preregistration = _load_json(cfg.preregistration)
    preregistration_binding = validate_preregistration(preregistration)
    skip_rows = _load_jsonl(cfg.skip_train_scores)
    tp12_rows = _load_jsonl(cfg.tp12_train_scores)
    skip_validation = _validate_candidate_score_rows(
        skip_rows,
        candidate="SKIP",
        source_binding=preregistration_binding["specialists"]["SKIP"],
        expected_model=preregistration_binding["base_model"],
    )
    tp12_validation = _validate_candidate_score_rows(
        tp12_rows,
        candidate="TP12",
        source_binding=preregistration_binding["specialists"]["TP12"],
        expected_model=preregistration_binding["base_model"],
    )
    if skip_validation["adapter_sha256"] == tp12_validation["adapter_sha256"]:
        raise ValueError("SKIP and TP12 specialist scores must use independent adapters")
    if skip_validation["model_name"] != tp12_validation["model_name"]:
        raise ValueError("SKIP and TP12 specialists must share the frozen base model")
    train_rows = _load_jsonl(cfg.train_data)
    score_rows, binding_validation = bind_specialist_scores_to_train_data(train_rows, skip_rows, tp12_rows, expected_identity_sha256=cfg.expected_identity_sha256)
    train_data_sha = _sha256_bytes(cfg.train_data.read_bytes())
    summary = _load_json(cfg.data_summary)
    summary_validation = validate_data_summary(summary, expected_identity_sha256=cfg.expected_identity_sha256, train_data_sha256=train_data_sha)
    manifest, strategy_cfg = lifecycle.frozen.load_frozen_manifest(cfg.manifest)
    if (
        summary.get("manifest_freeze_hash") != manifest.get("freeze_hash")
        or preregistration_binding["manifest_freeze_hash"] != manifest.get("freeze_hash")
    ):
        raise ValueError("lifecycle data summary manifest hash does not match")
    market, _, _state, active, engine = lifecycle.load_train_context(manifest, strategy_cfg)
    engine_cfg = _execution_config(strategy_cfg, strategy_cfg.leverage)
    full_signals = _full_pre2024_signals(market, active)
    if len(full_signals) != 462:
        raise ValueError(f"expected 462 pre-2024 frozen active signals, observed {len(full_signals)}")
    result = evaluate_thresholds(score_rows=score_rows, full_signals=full_signals, engine=engine, strategy_cfg=engine_cfg, manifest=manifest)
    common = {
        "protocol": "pposm_lifecycle_two_specialist_train_only_pair_threshold_v1",
        "status": result["status"],
        "default_action": DEFAULT_ACTION,
        "candidate_actions": list(CANDIDATE_ACTIONS),
        "selection_boundary": "pre_2024_train_only_no_oos_inputs",
        "manifest_freeze_hash": manifest.get("freeze_hash"),
        "inputs": {
            "preregistration": {
                "path": str(cfg.preregistration),
                "sha256": _sha256_bytes(cfg.preregistration.read_bytes()),
                **preregistration_binding,
            },
            "skip_train_scores": {"path": str(cfg.skip_train_scores), "sha256": _sha256_bytes(cfg.skip_train_scores.read_bytes()), **skip_validation},
            "tp12_train_scores": {"path": str(cfg.tp12_train_scores), "sha256": _sha256_bytes(cfg.tp12_train_scores.read_bytes()), **tp12_validation},
            "train_data": {"path": str(cfg.train_data), "sha256": train_data_sha, **binding_validation},
            "data_summary": {"path": str(cfg.data_summary), "sha256": _sha256_bytes(cfg.data_summary.read_bytes()), **summary_validation},
        },
        "full_pre2024_signals": len(full_signals),
        "anchor_signals": binding_validation["combined_score_rows"] // len(CANDIDATE_ACTIONS),
        "non_anchor_policy": "TP4",
        "threshold_grid": "independent SKIP and TP12 distinct scored train margins plus nextafter below-min/above-max sentinels",
        "routing_rule": "strict specialist margin > specialist threshold; if both switch choose higher normalized excess margin=(margin-threshold); exact excess tie SKIP; nonanchors TP4",
        "ranking_rule": [
            "maximize min(base/stress CAGR-to-MDD delta)",
            "maximize min(base/stress log-equity delta)",
            "higher SKIP threshold",
            "higher TP12 threshold",
            "stable route stream sha256",
        ],
        "future_can_rank_repair_or_reselect": False,
    }
    if result["status"] != "feasible_train_pair_threshold":
        failure = {
            **common,
            "candidate_thresholds": result["candidate_thresholds"],
            "candidate_thresholds_by_specialist": result["candidate_thresholds_by_specialist"],
            "control_economics": result["control_economics"],
            "best_diagnostics": _failure_diagnostics(result["evaluations"]),
        }
        cfg.failure_output.parent.mkdir(parents=True, exist_ok=True)
        cfg.failure_output.write_text(json.dumps(failure, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        raise SystemExit(2)
    selected = dict(result["selected"])
    route_stream = selected.pop("route_stream")
    artifact = {
        **common,
        "candidate_thresholds": result["candidate_thresholds"],
        "candidate_thresholds_by_specialist": result["candidate_thresholds_by_specialist"],
        "thresholds": dict(selected["thresholds"]),
        "selected_train_evaluation": selected,
        "full_train_routes": [{"signal_position": int(signal), "route": route} for signal, route in zip(full_signals, route_stream, strict=True)],
    }
    cfg.threshold_output.parent.mkdir(parents=True, exist_ok=True)
    cfg.threshold_output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=counterfactual.DEFAULT_MANIFEST)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--data-summary", type=Path, default=DEFAULT_DATA_SUMMARY)
    parser.add_argument("--skip-train-scores", type=Path, default=DEFAULT_SKIP_TRAIN_SCORES)
    parser.add_argument("--tp12-train-scores", type=Path, default=DEFAULT_TP12_TRAIN_SCORES)
    parser.add_argument("--threshold-output", type=Path, default=DEFAULT_THRESHOLD_OUTPUT)
    parser.add_argument("--failure-output", type=Path, default=DEFAULT_FAILURE_OUTPUT)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--expected-identity-sha256", default=DEFAULT_EXPECTED_IDENTITY_SHA256)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(Config(**vars(parse_args()))), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
