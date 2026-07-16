#!/usr/bin/env python3
"""Audit frozen historical candidate signals against no-order live scorers.

This audit deliberately stops before order placement, fills, exits, and
portfolio netting.  It proves only that each currently scoreable shadow sleeve
reconstructs the same scheduled candidate side on the historical interval for
which its source policy is defined.  A passing report is therefore necessary,
but not sufficient, for live promotion.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from execution.portfolio_live import (
    _build_portfolio_feature_frame,
    _required_availability_flags,
)
from execution.portfolio_shadow_policies import (
    build_fresh_kimchi_feature_frame,
    build_markov_feature_frame,
    observable_markov_transition_keys,
)
from execution.rex_llm_live import _rex_policy_features
from preprocessing.live_db_features import LiveDbFeatureConfig
from preprocessing.market_features import build_market_feature_frame
from training.audit_fresh_kimchi_orthogonal_alpha import fresh_candidate_masks
from training.audit_rex8640_usdkrw_gate import gate_match as rex_gate_match
from training.build_rex_event_reasoning_policy_data import _rex_pullback_reclaim_arrays
from training.evaluate_portfolio_llm_selector import _prep
from training.event_candidate_pool_probe import _feature_candidates
from training.long_regime_interest_gate_validation import build_interest_features
from training.portfolio_opt_added_alpha_update import (
    REX_GATES,
    SPLIT_BOUNDS,
    feature_frame as historical_markov_feature_frame,
    markov_active as historical_markov_active,
)
import training.portfolio_opt_all_discovered_alpha_gross10 as legacy_all
import training.portfolio_opt_new_alpha_pool as new_alpha
from training.search_bidirectional_state_alpha import extra as bidirectional_features
from training.search_gaussian_hmm_regime_alpha import hourly_features
from training.search_kimchi_leadlag_bidirectional_alpha import features as kimchi_features


DEFAULT_OUTPUT = "results/portfolio_added_alpha_shadow_signal_parity_2026-07-16.json"
SOURCE_MANIFEST = "configs/shadow/portfolio_added_alpha_signal_parity_sources_2026-07-16.json"
PORTFOLIO_START = pd.Timestamp(SPLIT_BOUNDS["train"][0])
MARKET_INTERVAL_MINUTES = 5
HISTORICAL_GRID_START = 143
REQUIRED_HISTORY_BARS = 17_280
REQUIRED_LOOKBACK_MINUTES = REQUIRED_HISTORY_BARS * MARKET_INTERVAL_MINUTES


def _resolve_existing(path: str | Path, *, artifact_root: str | Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate.resolve()
    if artifact_root is not None and not candidate.is_absolute():
        external = Path(artifact_root) / candidate
        if external.exists():
            return external.resolve()
    raise FileNotFoundError(
        f"required audit artifact is missing: {path}; pass --artifact-root explicitly if it "
        "is stored outside this checkout"
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve_existing(path).read_text())


def _load_jsonl(paths: Iterable[str | Path], *, deduplicate: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(
            json.loads(line)
            for line in Path(path).read_text().splitlines()
            if line.strip()
        )
    if deduplicate:
        unique = {(int(row["signal_pos"]), str(row["date"])): row for row in rows}
        rows = list(unique.values())
    return sorted(rows, key=lambda row: int(row["signal_pos"]))


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify_source_manifest(
    *,
    artifact_root: str | Path | None,
) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    manifest = _load_json(SOURCE_MANIFEST)
    resolved: dict[str, Path] = {}
    evidence: list[dict[str, Any]] = []
    for source in manifest["sources"]:
        name = str(source["name"])
        logical_path = str(source["path"])
        path = _resolve_existing(logical_path, artifact_root=artifact_root)
        actual_hash = _sha256_file(path)
        expected_hash = str(source["sha256"])
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"audit source hash mismatch for {name}: {actual_hash} != {expected_hash}"
            )
        expected_rows = source.get("rows")
        actual_rows = None
        if expected_rows is not None:
            actual_rows = sum(bool(line.strip()) for line in path.read_text().splitlines())
            if actual_rows != int(expected_rows):
                raise RuntimeError(
                    f"audit source row-count mismatch for {name}: {actual_rows} != {expected_rows}"
                )
        resolved[name] = path
        evidence.append(
            {
                "name": name,
                "logical_path": logical_path,
                "sha256": actual_hash,
                "rows": actual_rows,
                "resolved_outside_checkout": not str(path).startswith(
                    str(Path.cwd().resolve()) + "/"
                ),
            }
        )
    return resolved, evidence


def _date_ns(dates: pd.Series | pd.Index | np.ndarray) -> np.ndarray:
    return pd.to_datetime(dates, utc=True).astype("int64").to_numpy(dtype="<i8", copy=False)


def signal_digest(
    dates: pd.Series | pd.Index | np.ndarray,
    signal: np.ndarray,
    domain: np.ndarray,
) -> str:
    """Hash timestamp + {-1,0,+1} signal records on an explicit domain."""

    values = np.asarray(signal, dtype=np.int8)
    selected = np.asarray(domain, dtype=bool)
    if len(values) != len(selected):
        raise ValueError("signal/domain length mismatch")
    timestamps = _date_ns(dates)
    if len(timestamps) != len(values):
        raise ValueError("date/signal length mismatch")
    payload = np.empty(int(selected.sum()), dtype=[("timestamp_ns", "<i8"), ("signal", "i1")])
    payload["timestamp_ns"] = timestamps[selected]
    payload["signal"] = values[selected]
    return hashlib.sha256(payload.tobytes()).hexdigest()


def compare_signals(
    dates: pd.Series | pd.Index | np.ndarray,
    expected: np.ndarray,
    actual: np.ndarray,
    domain: np.ndarray,
    *,
    max_examples: int = 10,
) -> dict[str, Any]:
    expected_values = np.asarray(expected, dtype=np.int8)
    actual_values = np.asarray(actual, dtype=np.int8)
    selected = np.asarray(domain, dtype=bool)
    if len(expected_values) != len(actual_values) or len(expected_values) != len(selected):
        raise ValueError("comparison vector length mismatch")
    mismatch = selected & (expected_values != actual_values)
    positions = np.flatnonzero(mismatch)
    date_values = pd.to_datetime(dates)

    def summary(values: np.ndarray) -> dict[str, Any]:
        scoped = values[selected]
        return {
            "active_count": int(np.count_nonzero(scoped)),
            "long_count": int(np.count_nonzero(scoped > 0)),
            "short_count": int(np.count_nonzero(scoped < 0)),
            "sha256": signal_digest(dates, values, selected),
        }

    return {
        "passed": not len(positions),
        "domain_rows": int(selected.sum()),
        "mismatch_count": int(len(positions)),
        "expected": summary(expected_values),
        "actual": summary(actual_values),
        "first_mismatches": [
            {
                "position": int(position),
                "date": str(date_values.iloc[position] if hasattr(date_values, "iloc") else date_values[position]),
                "expected": int(expected_values[position]),
                "actual": int(actual_values[position]),
            }
            for position in positions[:max_examples]
        ],
    }


def compare_integer_vectors(
    dates: pd.Series | pd.Index | np.ndarray,
    expected: np.ndarray,
    actual: np.ndarray,
    domain: np.ndarray,
    *,
    max_examples: int = 10,
) -> dict[str, Any]:
    """Compare non-signal integer state vectors without int8 truncation."""

    expected_values = np.asarray(expected, dtype="<i2")
    actual_values = np.asarray(actual, dtype="<i2")
    selected = np.asarray(domain, dtype=bool)
    if len(expected_values) != len(actual_values) or len(expected_values) != len(selected):
        raise ValueError("comparison vector length mismatch")
    positions = np.flatnonzero(selected & (expected_values != actual_values))
    timestamps = _date_ns(dates)

    def digest(values: np.ndarray) -> str:
        payload = np.empty(
            int(selected.sum()),
            dtype=[("timestamp_ns", "<i8"), ("value", "<i2")],
        )
        payload["timestamp_ns"] = timestamps[selected]
        payload["value"] = values[selected]
        return hashlib.sha256(payload.tobytes()).hexdigest()

    date_values = pd.to_datetime(dates)
    return {
        "passed": not len(positions),
        "domain_rows": int(selected.sum()),
        "mismatch_count": int(len(positions)),
        "expected_sha256": digest(expected_values),
        "actual_sha256": digest(actual_values),
        "first_mismatches": [
            {
                "position": int(position),
                "date": str(date_values.iloc[position] if hasattr(date_values, "iloc") else date_values[position]),
                "expected": int(expected_values[position]),
                "actual": int(actual_values[position]),
            }
            for position in positions[:max_examples]
        ],
    }


def interval_slots(
    dates: pd.Series | pd.Index | np.ndarray,
    stride_bars: int,
    stride_offset_bars: int,
    *,
    interval_minutes: int = MARKET_INTERVAL_MINUTES,
) -> np.ndarray:
    timestamps = _date_ns(dates)
    minutes = timestamps // 60_000_000_000
    return ((minutes // int(interval_minutes)) % int(stride_bars)) == (
        int(stride_offset_bars) % int(stride_bars)
    )


def historical_schedule(length: int, hold_bars: int, stride_bars: int) -> tuple[np.ndarray, np.ndarray]:
    schedule = np.zeros(int(length), dtype=bool)
    stop = max(HISTORICAL_GRID_START, int(length) - int(hold_bars) - 2)
    schedule[np.arange(HISTORICAL_GRID_START, stop, int(stride_bars), dtype=np.int64)] = True
    domain = np.zeros(int(length), dtype=bool)
    domain[HISTORICAL_GRID_START:stop] = True
    return schedule, domain


def vector_gate_pass(frame: pd.DataFrame, gates: list[dict[str, Any]]) -> np.ndarray:
    active = np.ones(len(frame), dtype=bool)
    for gate in gates:
        feature = str(gate["feature"])
        for flag in _required_availability_flags(feature):
            values = pd.to_numeric(frame.get(flag, np.nan), errors="coerce")
            if not isinstance(values, pd.Series):
                values = pd.Series(values, index=frame.index)
            array = values.to_numpy(dtype=float)
            active &= np.isfinite(array) & (array > 0.5)
        if feature not in frame:
            active &= False
            continue
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=float)
        threshold = float(gate["threshold"])
        op = str(gate["op"])
        if op in {">=", "ge"}:
            passed = values >= threshold
        elif op in {"<=", "le"}:
            passed = values <= threshold
        else:
            raise ValueError(f"unsupported gate op: {op}")
        active &= np.isfinite(values) & passed
    return active


def vector_gate_clauses(frame: pd.DataFrame, clauses: list[list[dict[str, Any]]]) -> np.ndarray:
    if not clauses:
        return np.zeros(len(frame), dtype=bool)
    return np.logical_or.reduce([vector_gate_pass(frame, clause) for clause in clauses])


def _feature_parity(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    columns: Iterable[str],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for column in sorted(set(columns)):
        left = pd.to_numeric(expected[column], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(actual[column], errors="coerce").to_numpy(dtype=float)
        left_finite = np.isfinite(left)
        right_finite = np.isfinite(right)
        finite_mismatch = left_finite != right_finite
        value_mismatch = left_finite & right_finite & (left != right)
        mismatch = finite_mismatch | value_mismatch
        max_abs_diff = float(np.max(np.abs(left[left_finite & right_finite] - right[left_finite & right_finite]))) if np.any(left_finite & right_finite) else 0.0
        report[column] = {
            "passed": not bool(np.any(mismatch)),
            "mismatch_count": int(np.count_nonzero(mismatch)),
            "max_abs_diff": max_abs_diff,
        }
    return {
        "passed": all(item["passed"] for item in report.values()),
        "columns": report,
    }


def _schedule_parity(
    dates: pd.Series,
    *,
    hold_bars: int,
    stride_bars: int,
    stride_offset_bars: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    expected, domain = historical_schedule(len(dates), hold_bars, stride_bars)
    actual = interval_slots(dates, stride_bars, stride_offset_bars)
    report = compare_signals(dates, expected.astype(np.int8), actual.astype(np.int8), domain)
    report["runtime_slots_outside_historical_eligibility"] = int(np.count_nonzero(actual & ~domain))
    return expected, actual, report


def _fresh_audit(market: pd.DataFrame, runtime_features: pd.DataFrame) -> dict[str, Any]:
    cfg = _load_json("configs/shadow/fresh_kimchi_fx_2026-07-16.json")
    base = build_market_feature_frame(market, window_size=288)
    interest = build_interest_features(market, base)
    canonical = pd.concat([base, interest], axis=1).loc[:, lambda x: ~x.columns.duplicated(keep="last")]
    canonical = kimchi_features(market, bidirectional_features(market, canonical))
    expected_long, expected_short, diagnostics = fresh_candidate_masks(market, canonical)
    actual_features = build_fresh_kimchi_feature_frame(market, runtime_features)
    actual_long = vector_gate_pass(actual_features, cfg["long_gates"])
    actual_short = vector_gate_pass(actual_features, cfg["short_gates"])

    availability = {"funding_available", "usdkrw_available", "kimchi_available"}
    gate_features = {
        str(gate["feature"])
        for key in ("long_gates", "short_gates")
        for gate in cfg[key]
    }
    expected_feature_subset = canonical.loc[:, sorted(gate_features | availability)].copy()
    for column in availability:
        expected_feature_subset[column] = pd.to_numeric(market[column], errors="coerce").to_numpy(float)
    feature_report = _feature_parity(
        expected_feature_subset,
        actual_features,
        gate_features | availability,
    )

    expected_schedule, actual_schedule, schedule_report = _schedule_parity(
        market["date"],
        hold_bars=int(cfg["hold_bars"]),
        stride_bars=int(cfg["stride_bars"]),
        stride_offset_bars=int(cfg["stride_offset_bars"]),
    )
    _, domain = historical_schedule(len(market), int(cfg["hold_bars"]), int(cfg["stride_bars"]))
    expected_side = np.where(expected_long ^ expected_short, np.where(expected_long, 1, -1), 0)
    actual_side = np.where(actual_long ^ actual_short, np.where(actual_long, 1, -1), 0)
    decision_report = compare_signals(
        market["date"],
        np.where(expected_schedule, expected_side, 0),
        np.where(actual_schedule, actual_side, 0),
        domain,
    )
    masks_report = {
        "long": compare_signals(market["date"], expected_long, actual_long, np.ones(len(market), bool)),
        "short": compare_signals(
            market["date"],
            -expected_short.astype(np.int8),
            -actual_short.astype(np.int8),
            np.ones(len(market), bool),
        ),
    }
    passed = feature_report["passed"] and schedule_report["passed"] and decision_report["passed"] and all(
        item["passed"] for item in masks_report.values()
    )
    return {
        "passed": bool(passed),
        "scope": "scheduled candidate side only; TP/SL and non-overlap lifecycle excluded",
        "historical_diagnostics": diagnostics,
        "feature_parity": feature_report,
        "candidate_mask_parity": masks_report,
        "schedule_parity": schedule_report,
        "decision_parity": decision_report,
    }


def _historical_markov_transition_keys(market: pd.DataFrame, spec: dict[str, Any]) -> np.ndarray:
    _, hourly = hourly_features(market)
    trend = np.where(
        hourly["trend24"] <= float(spec["trend_low"]),
        0,
        np.where(hourly["trend24"] >= float(spec["trend_high"]), 2, 1),
    )
    volatility = (hourly["vol24"] >= float(spec["vol_median"])).astype(int)
    flow = (hourly["flow24"] >= float(spec["flow_median"])).astype(int)
    state = trend * 4 + volatility * 2 + flow
    previous = pd.Series(state, index=hourly.index).shift(1).fillna(-1).astype(int)
    transitions = previous * 12 + state
    mapped = pd.merge_asof(
        pd.DataFrame({"date": pd.to_datetime(market["date"]), "position": np.arange(len(market))}),
        pd.DataFrame({"date": hourly.index.to_numpy(), "transition": transitions.to_numpy()}),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta("2h"),
    ).sort_values("position")
    return mapped["transition"].fillna(-1).to_numpy(dtype=int)


def _markov_audit(market: pd.DataFrame, runtime_features: pd.DataFrame) -> dict[str, Any]:
    cfg = _load_json("configs/shadow/markov_transition_long_2026-07-16.json")
    contract = cfg["feature_contract"]
    canonical_features = historical_markov_feature_frame(market)
    canonical_base = new_alpha._alpha_active(canonical_features, "long_minimal_funding_premium")
    expected_transitions = _historical_markov_transition_keys(market, cfg["state_model"])
    expected_active = canonical_base & np.isin(
        expected_transitions,
        np.asarray(cfg["state_model"]["allowed_transition_keys"], dtype=int),
    )
    canonical_reference = historical_markov_active(market, canonical_features)
    reference_report = compare_signals(
        market["date"], expected_active, canonical_reference, np.ones(len(market), bool)
    )

    actual_features = build_markov_feature_frame(
        market,
        runtime_features,
        window_size=int(contract["window_size"]),
        zscore_window=int(contract["zscore_window"]),
        volume_window=int(contract["volume_window"]),
    )
    actual_base = vector_gate_clauses(actual_features, cfg["gate_clauses"])
    actual_transitions = observable_markov_transition_keys(market, cfg["state_model"])
    actual_active = actual_base & np.isin(
        actual_transitions,
        np.asarray(cfg["state_model"]["allowed_transition_keys"], dtype=int),
    )
    gate_features = {str(gate["feature"]) for clause in cfg["gate_clauses"] for gate in clause}
    availability_features: set[str] = set()
    for feature in list(gate_features):
        for flag in _required_availability_flags(feature):
            availability_features.add(flag)
    expected_feature_subset = canonical_features.loc[:, sorted(gate_features)].copy()
    for flag in availability_features:
        expected_feature_subset[flag] = pd.to_numeric(market[flag], errors="coerce").to_numpy(float)
    gate_features |= availability_features
    feature_report = _feature_parity(expected_feature_subset, actual_features, gate_features)

    expected_schedule, actual_schedule, schedule_report = _schedule_parity(
        market["date"],
        hold_bars=int(cfg["hold_bars"]),
        stride_bars=int(cfg["stride_bars"]),
        stride_offset_bars=int(cfg["stride_offset_bars"]),
    )
    _, domain = historical_schedule(len(market), int(cfg["hold_bars"]), int(cfg["stride_bars"]))
    base_report = compare_signals(market["date"], canonical_base, actual_base, np.ones(len(market), bool))
    transition_report = compare_integer_vectors(
        market["date"],
        expected_transitions,
        actual_transitions,
        np.ones(len(market), bool),
    )
    decision_report = compare_signals(
        market["date"],
        np.where(expected_schedule & expected_active, 1, 0),
        np.where(actual_schedule & actual_active, 1, 0),
        domain,
    )
    passed = all(
        report["passed"]
        for report in (
            reference_report,
            feature_report,
            base_report,
            transition_report,
            schedule_report,
            decision_report,
        )
    )
    return {
        "passed": bool(passed),
        "scope": "scheduled candidate side only; fixed-hold lifecycle and non-overlap excluded",
        "historical_reference_self_check": reference_report,
        "feature_parity": feature_report,
        "base_gate_parity": base_report,
        "transition_parity": transition_report,
        "schedule_parity": schedule_report,
        "decision_parity": decision_report,
    }


def _frozen_rex_veto_row() -> dict[str, Any]:
    report = legacy_all.load_json(legacy_all.SCAN_FILES["rex_veto"])
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for bucket in ("top", "tte_top"):
        for row in report.get(bucket, [])[:50]:
            key = json.dumps(row.get("gates", []), sort_keys=True)
            if key not in seen:
                seen.add(key)
                rows.append(row)
    if len(rows) <= 7:
        raise RuntimeError("frozen cand_rex_veto_7 source row is missing")
    return rows[7]


def _source_signal(rows: list[dict[str, Any]], length: int, *, row_kind: str) -> np.ndarray:
    signal = np.zeros(length, dtype=np.int8)
    for row in rows:
        position = int(row["signal_pos"])
        if row_kind == "taker":
            side = str((row.get("action") or {}).get("side", "")).upper()
        else:
            side = str((row.get("base_event") or {}).get("base_side", "")).upper()
        value = 1 if side == "LONG" else -1 if side == "SHORT" else 0
        if value == 0:
            raise RuntimeError(f"invalid REX source side at position {position}: {side}")
        if signal[position] not in {0, value}:
            raise RuntimeError(f"conflicting REX source sides at position {position}")
        signal[position] = value
    return signal


def _rex_strength_side_parity(
    rows: list[dict[str, Any]],
    strength: np.ndarray,
    direction: np.ndarray,
    *,
    row_kind: str,
) -> dict[str, Any]:
    strength_mismatches: list[dict[str, Any]] = []
    side_mismatches: list[dict[str, Any]] = []
    max_abs_diff = 0.0
    for row in rows:
        position = int(row["signal_pos"])
        if row_kind == "taker":
            expected_strength = float((row.get("feature_snapshot") or {})["rex_candidate_strength"])
            expected_side = str((row.get("action") or {}).get("side", "")).upper()
        else:
            expected_strength = float((row.get("base_event") or {})["strength"])
            expected_side = str((row.get("base_event") or {}).get("base_side", "")).upper()
        actual_strength = float(strength[position])
        actual_side = "LONG" if direction[position] > 0 else "SHORT" if direction[position] < 0 else "FLAT"
        difference = abs(expected_strength - actual_strength)
        max_abs_diff = max(max_abs_diff, difference)
        if difference != 0.0 and len(strength_mismatches) < 10:
            strength_mismatches.append(
                {"position": position, "expected": expected_strength, "actual": actual_strength}
            )
        if expected_side != actual_side and len(side_mismatches) < 10:
            side_mismatches.append(
                {"position": position, "expected": expected_side, "actual": actual_side}
            )
    strength_mismatch_count = sum(
        float(
            (row.get("feature_snapshot") or {}).get(
                "rex_candidate_strength", (row.get("base_event") or {}).get("strength")
            )
        )
        != float(strength[int(row["signal_pos"])])
        for row in rows
    )
    side_mismatch_count = sum(
        (
            str(
                (row.get("action") or {}).get(
                    "side", (row.get("base_event") or {}).get("base_side", "")
                )
            ).upper()
            != (
                "LONG"
                if direction[int(row["signal_pos"])] > 0
                else "SHORT"
                if direction[int(row["signal_pos"])] < 0
                else "FLAT"
            )
        )
        for row in rows
    )
    return {
        "passed": strength_mismatch_count == 0 and side_mismatch_count == 0,
        "rows": len(rows),
        "strength_mismatch_count": int(strength_mismatch_count),
        "side_mismatch_count": int(side_mismatch_count),
        "max_abs_strength_diff": float(max_abs_diff),
        "first_strength_mismatches": strength_mismatches,
        "first_side_mismatches": side_mismatches,
    }


def _rex_audit(
    market: pd.DataFrame,
    runtime_features: pd.DataFrame,
    *,
    name: str,
    config_path: str,
    source_paths: tuple[Path, ...],
    row_kind: str,
    source_start: pd.Timestamp,
) -> dict[str, Any]:
    cfg = _load_json(config_path)
    rows = _load_jsonl(source_paths, deduplicate=row_kind == "taker")
    dates = pd.to_datetime(market["date"])
    for row in rows:
        position = int(row["signal_pos"])
        if pd.Timestamp(row["date"]) != pd.Timestamp(dates.iloc[position]):
            raise RuntimeError(f"{name} source row/date mismatch at position {position}")

    policy = cfg["rex_policy"]
    policy_features = _rex_policy_features(market, runtime_features, str(policy["feature_contract"]))
    if str(policy["feature_contract"]) == "rex_event_reasoning_20260712":
        strength, direction = _rex_pullback_reclaim_arrays(policy_features)
    else:
        strength, direction = _feature_candidates(policy_features)[str(policy["family"])]
    threshold = float(policy["strength_threshold"])
    runtime_schedule = interval_slots(dates, int(cfg["stride_bars"]), int(cfg["stride_offset_bars"]))
    active_from = pd.Timestamp(str(cfg.get("active_from", source_start)))
    activation = np.asarray(dates >= active_from, dtype=bool)
    runtime_base = (strength > threshold) & (direction != 0) & runtime_schedule & activation
    runtime_base_signal = np.where(runtime_base, np.sign(direction), 0).astype(np.int8)
    expected_base_signal = _source_signal(rows, len(market), row_kind=row_kind)

    domain = np.asarray(
        (dates >= PORTFOLIO_START)
        & (np.arange(len(market)) < len(market) - int(cfg["hold_bars"]) - 2),
        dtype=bool,
    )
    base_report = compare_signals(dates, expected_base_signal, runtime_base_signal, domain)
    strength_side_report = _rex_strength_side_parity(rows, strength, direction, row_kind=row_kind)

    if row_kind == "taker":
        historical_gate = np.asarray([rex_gate_match(row, list(REX_GATES)) for row in rows], dtype=bool)
        frozen_gate_identity = list(REX_GATES) == cfg["gates"]
    else:
        frozen_row = _frozen_rex_veto_row()
        frozen_gate_identity = frozen_row.get("gates", []) == cfg["gates"]
        light_features = legacy_all._build_light_rex_features(market)
        historical_gate = np.asarray(
            [legacy_all._rex_row_matches(frozen_row.get("gates", []), light_features, row) for row in rows],
            dtype=bool,
        )
        del light_features

    runtime_gate_all = vector_gate_pass(policy_features, cfg["gates"])
    positions = np.asarray([int(row["signal_pos"]) for row in rows], dtype=int)
    runtime_gate = runtime_gate_all[positions]
    in_scope_rows = np.asarray([pd.Timestamp(row["date"]) >= PORTFOLIO_START for row in rows], dtype=bool)
    gate_mismatch = historical_gate != runtime_gate
    in_scope_gate_mismatches = int(np.count_nonzero(gate_mismatch & in_scope_rows))
    prehistory_mismatch_positions = positions[gate_mismatch & ~in_scope_rows]
    availability_exception_ok = True
    if len(prehistory_mismatch_positions):
        availability_exception_ok = bool(
            np.all(historical_gate[gate_mismatch & ~in_scope_rows])
            and not np.any(runtime_gate[gate_mismatch & ~in_scope_rows])
            and np.all(dates.iloc[prehistory_mismatch_positions].to_numpy() < PORTFOLIO_START.to_datetime64())
            and np.all(
                pd.to_numeric(
                    policy_features.iloc[prehistory_mismatch_positions]["open_interest_available"],
                    errors="coerce",
                ).fillna(0.0).to_numpy(float)
                <= 0.5
            )
        )

    expected_gated_signal = expected_base_signal.copy()
    expected_gated_signal[positions[~historical_gate]] = 0
    runtime_gated_signal = np.where(runtime_base & runtime_gate_all, np.sign(direction), 0).astype(np.int8)
    decision_report = compare_signals(dates, expected_gated_signal, runtime_gated_signal, domain)

    coverage_gap = np.asarray((dates >= PORTFOLIO_START) & (dates < source_start), dtype=bool)
    gap_runtime_signals = int(np.count_nonzero(runtime_gated_signal[coverage_gap]))
    coverage = {
        "portfolio_start": str(PORTFOLIO_START),
        "source_contract_start": str(source_start),
        "runtime_active_from": str(active_from),
        "source_first_signal": str(rows[0]["date"]),
        "source_last_signal": str(rows[-1]["date"]),
        "pre_source_domain_rows": int(np.count_nonzero(coverage_gap)),
        "runtime_gated_signals_in_pre_source_gap": gap_runtime_signals,
        "note": "runtime active_from must suppress every pre-source candidate",
    }
    passed = bool(
        frozen_gate_identity
        and strength_side_report["passed"]
        and base_report["passed"]
        and in_scope_gate_mismatches == 0
        and availability_exception_ok
        and decision_report["passed"]
    )
    return {
        "passed": passed,
        "scope": "scheduled REX candidate and frozen gate only; non-overlap/fixed-hold lifecycle excluded",
        "feature_contract": policy["feature_contract"],
        "source_rows": len(rows),
        "source_coverage": coverage,
        "frozen_gate_identity": bool(frozen_gate_identity),
        "strength_side_parity": strength_side_report,
        "base_candidate_parity": base_report,
        "gate_parity": {
            "passed_in_portfolio_scope": in_scope_gate_mismatches == 0,
            "historical_pass_count": int(historical_gate.sum()),
            "runtime_pass_count": int(runtime_gate.sum()),
            "in_scope_mismatch_count": in_scope_gate_mismatches,
            "prehistory_fail_closed_mismatch_count": int(len(prehistory_mismatch_positions)),
            "prehistory_availability_exception_valid": bool(availability_exception_ok),
            "prehistory_first_positions": [int(position) for position in prehistory_mismatch_positions[:10]],
        },
        "decision_parity": decision_report,
    }


def run(
    output: str | Path = DEFAULT_OUTPUT,
    *,
    artifact_root: str | Path | None = None,
    fail_on_mismatch: bool = True,
) -> dict[str, Any]:
    source_paths, source_evidence = _verify_source_manifest(artifact_root=artifact_root)
    portfolio = _load_json("configs/live/portfolio_added_alpha_shadow_candidate_2026-07-16.json")
    configured_lookback = int(portfolio.get("minimum_feature_history_minutes", 0))
    if configured_lookback < REQUIRED_LOOKBACK_MINUTES:
        raise RuntimeError(
            "shadow candidate history contract is too short: "
            f"{configured_lookback} < {REQUIRED_LOOKBACK_MINUTES} minutes"
        )
    market, _, _, _ = _prep()
    market["date"] = pd.to_datetime(market["date"])
    runtime_features = _build_portfolio_feature_frame(
        market,
        LiveDbFeatureConfig(),
        include_activity_flow=False,
    )

    sleeves: dict[str, Any] = {}
    sleeves["fresh_kimchi_fx"] = _fresh_audit(market, runtime_features)
    gc.collect()
    sleeves["markov_transition_long"] = _markov_audit(market, runtime_features)
    gc.collect()
    sleeves["rex_taker_low_range_position"] = _rex_audit(
        market,
        runtime_features,
        name="rex_taker_low_range_position",
        config_path="configs/shadow/rex_taker_low_range_position_2026-07-16.json",
        source_paths=(
            source_paths["rex_taker_train"],
            source_paths["rex_taker_test"],
            source_paths["rex_taker_eval"],
        ),
        row_kind="taker",
        source_start=pd.Timestamp(
            _load_json("configs/shadow/rex_taker_low_range_position_2026-07-16.json")[
                "active_from"
            ]
        ),
    )
    gc.collect()
    sleeves["cand_rex_veto_7"] = _rex_audit(
        market,
        runtime_features,
        name="cand_rex_veto_7",
        config_path="configs/live/rex_veto_7_candidate.json",
        source_paths=(source_paths["rex_veto_source"],),
        row_kind="cand",
        source_start=PORTFOLIO_START,
    )

    source_contract_passed = all(result["passed"] for result in sleeves.values())
    full_interval_passed = bool(
        source_contract_passed
        and all(
            int(result.get("source_coverage", {}).get("runtime_gated_signals_in_pre_source_gap", 0))
            == 0
            for result in sleeves.values()
        )
    )
    passed = bool(source_contract_passed and full_interval_passed)
    report = {
        "schema_version": 1,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "passed": bool(passed),
        "source_contract_signal_parity_passed": bool(source_contract_passed),
        "full_portfolio_interval_signal_parity_passed": full_interval_passed,
        "candidate_signal_parity_passed_count": int(sum(result["passed"] for result in sleeves.values())),
        "candidate_signal_parity_total": len(sleeves),
        "market": {
            "rows": len(market),
            "start": str(market["date"].iloc[0]),
            "end": str(market["date"].iloc[-1]),
            "portfolio_evaluation_start": str(PORTFOLIO_START),
        },
        "scope": {
            "proved": "historical/live feature, stride, candidate-side, and frozen gate parity in each declared source domain",
            "not_proved": [
                "fills, slippage, order lifecycle, position netting, and portfolio PnL parity",
                "Fresh Kimchi TP/SL barrier exits",
                "frozen_annual_rank7 scoring or exits",
                "absence of research contamination or alpha generalization",
            ],
            "active_window_contract": (
                "rex_taker_low_range_position active_from=2021-01-01 suppresses every "
                "2020-09-01..2020-12-31 pre-source candidate"
            ),
            "orders_enabled": False,
        },
        "live_history_contract": {
            "required_history_bars_5m": REQUIRED_HISTORY_BARS,
            "required_lookback_minutes": REQUIRED_LOOKBACK_MINUTES,
            "configured_minimum_feature_history_minutes": configured_lookback,
            "reason": "market_features_v1 htf_3d completed-bar guard requires 17,280 source rows",
            "passed": configured_lookback >= REQUIRED_LOOKBACK_MINUTES,
        },
        "sleeves": sleeves,
        "source_artifacts": source_evidence,
        "artifact_hashes": {
            SOURCE_MANIFEST: _sha256_file(SOURCE_MANIFEST),
            "configs/live/portfolio_added_alpha_shadow_candidate_2026-07-16.json": _sha256_file(
                "configs/live/portfolio_added_alpha_shadow_candidate_2026-07-16.json"
            ),
            "configs/shadow/fresh_kimchi_fx_2026-07-16.json": _sha256_file(
                "configs/shadow/fresh_kimchi_fx_2026-07-16.json"
            ),
            "configs/shadow/markov_transition_long_2026-07-16.json": _sha256_file(
                "configs/shadow/markov_transition_long_2026-07-16.json"
            ),
            "configs/shadow/rex_taker_low_range_position_2026-07-16.json": _sha256_file(
                "configs/shadow/rex_taker_low_range_position_2026-07-16.json"
            ),
            "configs/live/rex_veto_7_candidate.json": _sha256_file(
                "configs/live/rex_veto_7_candidate.json"
            ),
        },
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if fail_on_mismatch and not passed:
        failed = [name for name, result in sleeves.items() if not result["passed"]]
        raise RuntimeError(f"shadow signal parity failed: {failed}; report={output_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--artifact-root", default="")
    parser.add_argument("--allow-mismatch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(
        args.output,
        artifact_root=str(args.artifact_root) or None,
        fail_on_mismatch=not args.allow_mismatch,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "candidate_signal_parity": (
                    f"{report['candidate_signal_parity_passed_count']}/"
                    f"{report['candidate_signal_parity_total']}"
                ),
                "output": args.output,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
