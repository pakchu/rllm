"""Select a train-only lifecycle residual threshold for PPOSM.

This module intentionally has no OOS inputs.  It consumes pre-2024 scored
lifecycle-anchor pair rows, replays the full pre-2024 frozen active stream with
non-anchor signals defaulted to TP4, and freezes a threshold only if all
pre-registered train materiality/economic gates pass.
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

DEFAULT_TRAIN_DATA = lifecycle.DEFAULT_TRAIN_OUTPUT
DEFAULT_DATA_SUMMARY = lifecycle.DEFAULT_SUMMARY_OUTPUT
DEFAULT_TRAIN_SCORES = Path("results/pposm_lifecycle_residual_train_scores_2026-09-02.jsonl")
DEFAULT_THRESHOLD_OUTPUT = Path("results/pposm_lifecycle_residual_train_threshold_2026-09-02.json")
DEFAULT_FAILURE_OUTPUT = Path("results/pposm_lifecycle_residual_train_threshold_failure_2026-09-02.json")
DEFAULT_PREREGISTRATION = Path(
    "results/pposm_lifecycle_residual_sft_rlvr_preregistration_2026-09-02.json"
)
DEFAULT_RLVR_CONFIG = Path(
    "checkpoints/pposm_lifecycle_residual_sft_rlvr_2026-09-02/config_diagnostics.json"
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
    train_scores: Path = DEFAULT_TRAIN_SCORES
    threshold_output: Path = DEFAULT_THRESHOLD_OUTPUT
    failure_output: Path = DEFAULT_FAILURE_OUTPUT
    preregistration: Path = DEFAULT_PREREGISTRATION
    rlvr_config: Path = DEFAULT_RLVR_CONFIG
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


def validate_preregistration(
    preregistration: dict[str, Any],
    *,
    train_data: Path,
    data_summary: Path,
    expected_identity_sha256: str = DEFAULT_EXPECTED_IDENTITY_SHA256,
) -> dict[str, Any]:
    allowed = {
        "pposm_lifecycle_anchor_residual_sft_rlvr_v1": "preregistered_before_lifecycle_sft_and_oos",
        "pposm_lifecycle_dual_verifier_sft_rlvr_v1": "preregistered_before_dual_verifier_rlvr_and_oos",
    }
    protocol = str(preregistration.get("protocol_version", ""))
    if protocol not in allowed or preregistration.get("status") != allowed[protocol]:
        raise ValueError("lifecycle preregistration protocol/status changed")
    source = preregistration.get("source")
    if not isinstance(source, dict):
        raise ValueError("lifecycle preregistration lacks source bindings")
    train_binding = source.get("train_data")
    if not isinstance(train_binding, dict):
        raise ValueError("lifecycle preregistration lacks train_data binding")
    if (
        Path(str(train_binding.get("path"))) != train_data
        or train_binding.get("sha256") != _sha256_bytes(train_data.read_bytes())
        or train_binding.get("identity_sha256") != expected_identity_sha256
        or train_binding.get("rows") != 204
        or train_binding.get("anchors") != 102
    ):
        raise ValueError("lifecycle preregistration train_data binding changed")
    summary_binding = source.get("data_summary") or source.get("train_summary")
    if not isinstance(summary_binding, dict):
        raise ValueError("lifecycle preregistration lacks data summary binding")
    if (
        Path(str(summary_binding.get("path"))) != data_summary
        or summary_binding.get("sha256") != _sha256_bytes(data_summary.read_bytes())
    ):
        raise ValueError("lifecycle preregistration data summary binding changed")
    base_model = preregistration.get("base_model", {}).get("name")
    frozen_sft_sha256 = None
    if protocol == "pposm_lifecycle_dual_verifier_sft_rlvr_v1":
        frozen_sft = source.get("frozen_sft_adapter")
        if not isinstance(frozen_sft, dict):
            raise ValueError("dual-verifier preregistration lacks frozen SFT binding")
        sft_path = Path(str(frozen_sft.get("path", "")))
        summary_path = Path(str(frozen_sft.get("summary_path", "")))
        if (
            not sft_path.is_file()
            or _sha256_bytes(sft_path.read_bytes()) != frozen_sft.get("sha256")
            or not summary_path.is_file()
            or _sha256_bytes(summary_path.read_bytes())
            != frozen_sft.get("summary_sha256")
        ):
            raise ValueError("dual-verifier frozen SFT artifact binding changed")
        base_model = _load_json(summary_path).get("model_name")
        frozen_sft_sha256 = frozen_sft.get("sha256")
    if base_model != "Qwen/Qwen2.5-1.5B-Instruct":
        raise ValueError("lifecycle preregistration base model changed")
    return {
        "protocol_version": protocol,
        "status": allowed[protocol],
        "manifest_freeze_hash": source.get("manifest_freeze_hash"),
        "base_model": base_model,
        "architecture": preregistration.get("architecture", {}).get("name"),
        "expected_rlvr_schema": (
            "pposm_residual_target_utility"
            if protocol == "pposm_lifecycle_dual_verifier_sft_rlvr_v1"
            else "pposm_residual_utility"
        ),
        "frozen_sft_sha256": frozen_sft_sha256,
    }


def validate_rlvr_provenance(
    rlvr_config_path: Path,
    *,
    preregistration_validation: dict[str, Any],
    score_validation: dict[str, Any],
    train_data: Path,
) -> dict[str, Any]:
    diagnostics = _load_json(rlvr_config_path)
    config = diagnostics.get("config")
    if not isinstance(config, dict) or diagnostics.get("dry_run") is not False:
        raise ValueError("RLVR config diagnostics are not from a completed run")
    expected = {
        "schema": preregistration_validation["expected_rlvr_schema"],
        "base_model": preregistration_validation["base_model"],
        "train_jsonl": str(train_data),
    }
    observed = {
        "schema": config.get("label_schema"),
        "base_model": config.get("base_model"),
        "train_jsonl": config.get("train_jsonl"),
    }
    if observed != expected:
        raise ValueError(
            f"RLVR config does not match selected preregistration family: {observed}"
        )
    adapter_path = Path(str(config.get("output_dir", ""))) / "adapter_model.safetensors"
    if (
        not adapter_path.is_file()
        or _sha256_bytes(adapter_path.read_bytes())
        != score_validation["adapter_sha256"]
    ):
        raise ValueError("scored adapter hash differs from completed RLVR run")
    frozen_sft_sha256 = preregistration_validation.get("frozen_sft_sha256")
    if frozen_sft_sha256:
        sft_path = Path(str(config.get("sft_adapter_dir", ""))) / "adapter_model.safetensors"
        if (
            not sft_path.is_file()
            or _sha256_bytes(sft_path.read_bytes()) != frozen_sft_sha256
        ):
            raise ValueError("dual-verifier RLVR did not start from frozen SFT adapter")
    return {
        "path": str(rlvr_config_path),
        "sha256": _sha256_bytes(rlvr_config_path.read_bytes()),
        "schema": observed["schema"],
        "base_model": observed["base_model"],
        "adapter_path": str(adapter_path),
        "adapter_sha256": score_validation["adapter_sha256"],
    }


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


def validate_score_rows(
    rows: Sequence[dict[str, Any]],
    *,
    expected_identity_sha256: str = DEFAULT_EXPECTED_IDENTITY_SHA256,
) -> dict[str, Any]:
    """Validate the exact 204 train anchor pair score rows."""

    if len(rows) != 204:
        raise ValueError(f"expected 204 pre-2024 anchor pair score rows, observed {len(rows)}")
    identities = [row.get("identity") for row in rows]
    if not all(isinstance(identity, str) and identity for identity in identities):
        raise ValueError("every score row must include non-empty identity")
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate score row identity")
    identity_hash = _identity_sha256(rows)
    if expected_identity_sha256 and identity_hash != expected_identity_sha256:
        raise ValueError(
            f"anchor pair identity hash mismatch: expected {expected_identity_sha256}, observed {identity_hash}"
        )
    by_base: dict[str, list[str]] = defaultdict(list)
    adapter_hashes: set[str] = set()
    source_hashes: set[str] = set()
    source_identity_hashes: set[str] = set()
    model_names: set[str] = set()
    for row in rows:
        if row.get("split") != "train" or row.get("window") != TRAIN_WINDOW[0]:
            raise ValueError("score rows must be train/pre_2024 only")
        if not _time_pre2024(row.get("signal_time", row.get("date"))):
            raise ValueError("score row signal_time/date must be pre-2024")
        identity = str(row["identity"])
        candidate = str(row.get("candidate_action"))
        signal_position = row.get("signal_position", row.get("signal_pos"))
        expected_identity = lifecycle.lifecycle_identity(TRAIN_WINDOW[0], int(signal_position), candidate)
        if identity != expected_identity:
            raise ValueError("score row identity does not match candidate/window/signal_position")
        base_identity = str(row.get("base_identity"))
        expected_base = counterfactual.signal_identity(TRAIN_WINDOW[0], int(signal_position))
        if base_identity != expected_base:
            raise ValueError("score row base_identity does not match pre_2024 signal_position")
        if candidate not in CANDIDATE_ACTIONS:
            raise ValueError("candidate_action must be SKIP or TP12")
        _score_margin(row)
        if row.get("score_normalization") != "mean":
            raise ValueError("lifecycle score rows must use mean label-logprob normalization")
        adapter_sha256 = str(row.get("adapter_sha256", ""))
        if len(adapter_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in adapter_sha256.lower()
        ):
            raise ValueError("lifecycle score row lacks adapter_sha256")
        adapter_hashes.add(adapter_sha256)
        source_hashes.add(str(row.get("source_jsonl_sha256", "")))
        source_identity_hashes.add(str(row.get("source_identity_sha256", "")))
        model_names.add(str(row.get("model_name", "")))
        by_base[base_identity].append(candidate)
    for base, candidates in by_base.items():
        if tuple(sorted(candidates)) != tuple(sorted(CANDIDATE_ACTIONS)):
            raise ValueError(f"base identity {base} lacks exactly one SKIP and TP12 row")
    if (
        len(adapter_hashes) != 1
        or len(source_hashes) != 1
        or len(source_identity_hashes) != 1
        or model_names != {"Qwen/Qwen2.5-1.5B-Instruct"}
    ):
        raise ValueError("lifecycle score rows mix adapter/model/source provenance")
    return {
        "rows": len(rows),
        "anchors": len(by_base),
        "identity_sha256": identity_hash,
        "adapter_sha256": next(iter(adapter_hashes)),
        "source_jsonl_sha256": next(iter(source_hashes)),
        "source_identity_sha256": next(iter(source_identity_hashes)),
        "model_name": next(iter(model_names)),
        "score_normalization": "mean",
    }


def validate_train_data_identity(
    train_data_rows: Sequence[dict[str, Any]], score_rows: Sequence[dict[str, Any]], expected_identity_sha256: str
) -> dict[str, Any]:
    data_pairs: list[dict[str, Any]] = []
    for row in train_data_rows:
        meta = row.get("metadata")
        if not isinstance(meta, dict):
            raise ValueError("train data row missing metadata")
        data_pairs.append(
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
    data_identity_hash = _identity_sha256(data_pairs)
    score_identity_hash = _identity_sha256(score_rows)
    if data_identity_hash != expected_identity_sha256 or score_identity_hash != data_identity_hash:
        raise ValueError("train data and scored row identity hashes do not match the frozen anchor hash")
    data_identity_order = [str(row["identity"]) for row in data_pairs]
    score_identity_order = [str(row["identity"]) for row in score_rows]
    if data_identity_order != score_identity_order:
        raise ValueError("score rows are not in the same frozen anchor identity order as train data")
    return {"train_data_rows": len(train_data_rows), "identity_sha256": data_identity_hash}


def validate_data_summary(
    summary: dict[str, Any],
    *,
    expected_identity_sha256: str,
    train_data_sha256: str,
) -> dict[str, Any]:
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
        raise ValueError(
            f"lifecycle data summary does not bind the frozen train data: {observed}"
        )
    return observed


def _full_pre2024_signals(market: pd.DataFrame, active: Sequence[bool]) -> tuple[int, ...]:
    window, start, end = TRAIN_WINDOW
    del window
    dates = pd.to_datetime(market["date"])
    mask = np.asarray(active, dtype=bool)
    period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    return tuple(int(value) for value in np.flatnonzero(mask & period))


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
    if any(right.entry_position <= left.exit_position for left, right in zip(trades, trades[1:])):
        raise RuntimeError("route simulation produced overlapping trades")
    return tuple(trades)


def _economics(trades: Sequence[Trade], *, start: str, end: str, cfg: Any) -> dict[str, dict[str, Any]]:
    return {
        name: equity_stats(trades, start=start, end=end, cfg=cfg, cost_rate=cost)
        for name, cost in lifecycle.COSTS.items()
    }


def _log_equity_delta(predicted: dict[str, Any], control: dict[str, Any]) -> float:
    return float(
        math.log1p(float(predicted["absolute_return_pct"]) / 100.0)
        - math.log1p(float(control["absolute_return_pct"]) / 100.0)
    )


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


def _anchor_routes_for_threshold(score_rows: Sequence[dict[str, Any]], threshold: float) -> dict[int, dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        grouped[int(row.get("signal_position", row.get("signal_pos")))].append(row)
    tie_rank = {"SKIP": 0, "TP12": 1}
    selected: dict[int, dict[str, Any]] = {}
    for signal, rows in grouped.items():
        eligible = [row for row in rows if _score_margin(row) > threshold]
        margins = {str(row["candidate_action"]): _score_margin(row) for row in rows}
        if eligible:
            chosen = max(eligible, key=lambda row: (_score_margin(row), -tie_rank[str(row["candidate_action"])]))
            route = str(chosen["candidate_action"])
            decision = "SWITCH"
            selected_margin = _score_margin(chosen)
        else:
            route = DEFAULT_ACTION
            decision = "KEEP"
            selected_margin = max(margins.values())
        selected[signal] = {
            "route": route,
            "decision": decision,
            "selected_margin": float(selected_margin),
            "candidate_margins": dict(sorted(margins.items())),
        }
    return selected


def route_stream_for_threshold(
    score_rows: Sequence[dict[str, Any]], full_signals: Sequence[int], threshold: float
) -> tuple[str, ...]:
    anchor_routes = _anchor_routes_for_threshold(score_rows, threshold)
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
        "passes": bool(
            total > 0
            and non_default_total / total >= 0.10
            and all(count >= 10 for count in used_nondefault_counts)
            and max(counts.values(), default=0) / total <= 0.90
        ),
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
    payload = [
        {"signal": int(t.signal_position), "entry": int(t.entry_position), "exit": int(t.exit_position)}
        for t in trades
    ]
    return _sha256_bytes(canonical_json(payload).encode("utf-8"))


def evaluate_thresholds(
    *,
    score_rows: Sequence[dict[str, Any]],
    full_signals: Sequence[int],
    engine: ExecutionEngine,
    strategy_cfg: Any,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    _, start, end = TRAIN_WINDOW
    baseline_routes = tuple(DEFAULT_ACTION for _ in full_signals)
    baseline_trades = _apply_routes(engine, full_signals, baseline_routes, start=start, end=end, spec=manifest["spec"])
    baseline_economics = _economics(baseline_trades, start=start, end=end, cfg=strategy_cfg)
    candidates = _threshold_candidates([_score_margin(row) for row in score_rows])
    evaluations: list[dict[str, Any]] = []
    feasible: list[dict[str, Any]] = []
    for threshold in candidates:
        routes = route_stream_for_threshold(score_rows, full_signals, threshold)
        trades = _apply_routes(engine, full_signals, routes, start=start, end=end, spec=manifest["spec"])
        economics = _economics(trades, start=start, end=end, cfg=strategy_cfg)
        feas = feasibility(economics, baseline_economics, routes)
        item = {
            "threshold": float(threshold),
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
            "status": "no_feasible_train_threshold",
            "candidate_thresholds": len(candidates),
            "control_economics": baseline_economics,
            "evaluations": evaluations,
        }
    selected = max(
        feasible,
        key=lambda item: (
            float(item["feasibility"]["ranking"]["min_ratio_delta"]),
            float(item["feasibility"]["ranking"]["min_log_equity_delta"]),
            float(item["threshold"]),
            -int(item["route_stream_sha256"], 16),
        ),
    )
    return {
        "status": "feasible_train_threshold",
        "candidate_thresholds": len(candidates),
        "control_economics": baseline_economics,
        "selected": selected,
        "evaluations": evaluations,
    }


def run(cfg: Config) -> dict[str, Any]:
    preregistration = _load_json(cfg.preregistration)
    preregistration_validation = validate_preregistration(
        preregistration,
        train_data=cfg.train_data,
        data_summary=cfg.data_summary,
        expected_identity_sha256=cfg.expected_identity_sha256,
    )
    score_rows = _load_jsonl(cfg.train_scores)
    score_validation = validate_score_rows(score_rows, expected_identity_sha256=cfg.expected_identity_sha256)
    rlvr_validation = validate_rlvr_provenance(
        cfg.rlvr_config,
        preregistration_validation=preregistration_validation,
        score_validation=score_validation,
        train_data=cfg.train_data,
    )
    train_rows = _load_jsonl(cfg.train_data)
    data_validation = validate_train_data_identity(train_rows, score_rows, cfg.expected_identity_sha256)
    data_validation["train_data_sha256"] = _sha256_bytes(cfg.train_data.read_bytes())
    if (
        score_validation["source_jsonl_sha256"]
        != data_validation["train_data_sha256"]
        or score_validation["source_identity_sha256"]
        != cfg.expected_identity_sha256
    ):
        raise ValueError("lifecycle score source binding differs from frozen train data")
    score_sha = _sha256_bytes(cfg.train_scores.read_bytes())
    summary = _load_json(cfg.data_summary)
    summary_validation = validate_data_summary(
        summary,
        expected_identity_sha256=cfg.expected_identity_sha256,
        train_data_sha256=data_validation["train_data_sha256"],
    )
    manifest, strategy_cfg = lifecycle.frozen.load_frozen_manifest(cfg.manifest)
    if (
        summary.get("manifest_freeze_hash") != manifest.get("freeze_hash")
        or preregistration_validation["manifest_freeze_hash"]
        != manifest.get("freeze_hash")
    ):
        raise ValueError("lifecycle data summary manifest hash does not match")
    market, _, _state, active, engine = lifecycle.load_train_context(manifest, strategy_cfg)
    engine_cfg = _execution_config(strategy_cfg, strategy_cfg.leverage)
    full_signals = _full_pre2024_signals(market, active)
    if len(full_signals) != 462:
        raise ValueError(f"expected 462 pre-2024 frozen active signals, observed {len(full_signals)}")
    result = evaluate_thresholds(
        score_rows=score_rows,
        full_signals=full_signals,
        engine=engine,
        strategy_cfg=engine_cfg,
        manifest=manifest,
    )
    common = {
        "protocol": "pposm_lifecycle_train_only_threshold_v1",
        "status": result["status"],
        "default_action": DEFAULT_ACTION,
        "candidate_actions": list(CANDIDATE_ACTIONS),
        "selection_boundary": "pre_2024_train_only_no_oos_inputs",
        "manifest_freeze_hash": manifest.get("freeze_hash"),
        "inputs": {
            "preregistration": {
                "path": str(cfg.preregistration),
                "sha256": _sha256_bytes(cfg.preregistration.read_bytes()),
                **preregistration_validation,
            },
            "train_scores": {"path": str(cfg.train_scores), "sha256": score_sha, **score_validation},
            "rlvr_run": rlvr_validation,
            "train_data": {"path": str(cfg.train_data), **data_validation},
            "data_summary": {
                "path": str(cfg.data_summary),
                "sha256": _sha256_bytes(cfg.data_summary.read_bytes()),
                **summary_validation,
            },
        },
        "full_pre2024_signals": len(full_signals),
        "anchor_signals": score_validation["anchors"],
        "non_anchor_policy": "TP4",
        "threshold_grid": "distinct scored train margins plus nextafter below-min/above-max sentinels",
        "routing_rule": "strict margin > threshold; dual candidate larger margin; exact tie SKIP; nonanchors TP4",
        "ranking_rule": [
            "maximize min(base/stress CAGR-to-MDD delta)",
            "maximize min(base/stress log-equity delta)",
            "higher threshold",
            "stable route stream sha256",
        ],
        "future_can_rank_repair_or_reselect": False,
    }
    if result["status"] != "feasible_train_threshold":
        failure = {
            **common,
            "candidate_thresholds": result["candidate_thresholds"],
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
        "threshold": selected["threshold"],
        "selected_train_evaluation": selected,
        "full_train_routes": [
            {"signal_position": int(signal), "route": route}
            for signal, route in zip(full_signals, route_stream, strict=True)
        ],
    }
    cfg.threshold_output.parent.mkdir(parents=True, exist_ok=True)
    cfg.threshold_output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return artifact


def _failure_diagnostics(evaluations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not evaluations:
        return {}
    best = max(
        evaluations,
        key=lambda item: (
            float(item["feasibility"]["ranking"]["min_ratio_delta"]),
            float(item["feasibility"]["ranking"]["min_log_equity_delta"]),
            float(item["threshold"]),
        ),
    )
    return {
        "best_threshold_by_ranking_even_if_infeasible": float(best["threshold"]),
        "route_stream_sha256": best["route_stream_sha256"],
        "failed_gates": [key for key, ok in best["feasibility"]["gates"].items() if not ok],
        "materiality": best["feasibility"]["materiality"],
        "deltas": best["feasibility"]["deltas"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=counterfactual.DEFAULT_MANIFEST)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--data-summary", type=Path, default=DEFAULT_DATA_SUMMARY)
    parser.add_argument("--train-scores", type=Path, default=DEFAULT_TRAIN_SCORES)
    parser.add_argument("--threshold-output", type=Path, default=DEFAULT_THRESHOLD_OUTPUT)
    parser.add_argument("--failure-output", type=Path, default=DEFAULT_FAILURE_OUTPUT)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--rlvr-config", type=Path, default=DEFAULT_RLVR_CONFIG)
    parser.add_argument("--expected-identity-sha256", default=DEFAULT_EXPECTED_IDENTITY_SHA256)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(Config(**vars(parse_args()))), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
