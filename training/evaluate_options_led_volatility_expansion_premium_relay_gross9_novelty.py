"""Evaluate frozen OVEPR-24 structural novelty against all Gross9 sleeves.

Only the frozen OVEPR clock, the Gross9 structural-clock bundle, and their
JSON control artifacts are opened.  BTC prices, returns, funding cash flows,
portfolio outcomes, and PnL remain unopened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from training import export_gross9_structural_clocks as gross9


PROTOCOL_VERSION = "ovepr_24_gross9_novelty_v1"
POLICY_ID = "OVEPR-24"
PREREGISTRATION = Path("results/options_led_volatility_expansion_premium_relay_preregistration_2026-08-08.json")
PREREGISTRATION_SHA256 = "180e6be7f6889024896303be511a07b3a95b44dc225f4566a2edab7127022dd6"
PREREGISTRATION_MANIFEST_HASH = "0433453fd6895045e031f62e4c20f5b3591b2b74a9d36aad2ea6b18f2a0f932f"
SOURCE_SUPPORT = Path("results/options_led_volatility_expansion_premium_relay_support_2026-08-08.json")
SOURCE_SUPPORT_SHA256 = "afbf8157c2c85aec0470563cdfba1b45afe18472a617633d2140b6ec6c1c15a7"
PRIMARY_CLOCK = Path("data/options_led_volatility_expansion_premium_relay_clocks_2023_2026.csv.gz")
PRIMARY_CLOCK_SHA256 = "b79bd105784db59980a83d1e1e75e3334e954f76e0f06a6d44eca1dc017e6bf1"
GROSS9_MANIFEST = gross9.DEFAULT_MANIFEST
DEFAULT_OUTPUT = Path("results/options_led_volatility_expansion_premium_relay_gross9_novelty_2026-08-08.json")
WINDOW_START = pd.Timestamp("2023-07-01T00:00:00Z")
WINDOW_END = pd.Timestamp("2025-01-01T00:00:00Z")
BAR = pd.Timedelta(minutes=5)
NEAR_LIMIT = pd.Timedelta(hours=6)
LIMITS = {
    "exact_entry_jaccard": 0.10,
    "one_to_one_6h_max_matched_share": 0.45,
    "occupied_5m_bar_jaccard": 0.30,
    "absolute_signed_exposure_pearson": 0.35,
}
CLOCK_COLUMNS = {"entry_time", "exit_time", "side"}


def sha256_file(path: str | Path) -> str:
    return gross9.sha256_file(path)


def canonical_hash(payload: Any) -> str:
    return gross9.canonical_hash(payload)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON artifact is not an object: {path}")
    return payload


def _verify_manifest(payload: dict[str, Any], label: str) -> None:
    observed = payload.get("manifest_hash")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if observed != canonical_hash(core):
        raise RuntimeError(f"{label} canonical manifest hash drift")


def _load_frozen_controls() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("OVEPR preregistration hash drift")
    preregistration = _load_json(PREREGISTRATION)
    _verify_manifest(preregistration, "OVEPR preregistration")
    if preregistration.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("OVEPR preregistration manifest binding drift")
    frozen_limits = preregistration.get("novelty", {}).get(
        "requirements_each_comparator_and_each_gross9_sleeve"
    )
    expected_limits = {
        "exact_entry_jaccard_max": "1/10",
        "one_to_one_6h_max_matched_share_max": "9/20",
        "occupied_5m_bar_jaccard_max": "3/10",
        "absolute_signed_exposure_pearson_max": "7/20",
    }
    if frozen_limits != expected_limits:
        raise RuntimeError("OVEPR Gross9 novelty limits drift")

    if sha256_file(SOURCE_SUPPORT) != SOURCE_SUPPORT_SHA256:
        raise RuntimeError("OVEPR source-support artifact hash drift")
    support = _load_json(SOURCE_SUPPORT)
    _verify_manifest(support, "OVEPR source support")
    if support.get("policy_id") != POLICY_ID or support.get("gross9_novelty_status") != "pending":
        raise RuntimeError("OVEPR source-support predecessor state drift")
    if support.get("advance_to_economic_outcomes") is not False:
        raise RuntimeError("OVEPR outcomes were already authorized")
    if sha256_file(PRIMARY_CLOCK) != PRIMARY_CLOCK_SHA256:
        raise RuntimeError("OVEPR primary clock hash drift")
    if support.get("clock", {}).get("sha256") != PRIMARY_CLOCK_SHA256:
        raise RuntimeError("OVEPR source-support clock binding drift")

    manifest = _load_json(GROSS9_MANIFEST)
    _verify_manifest(manifest, "Gross9 structural clocks")
    if manifest.get("protocol_version") != gross9.PROTOCOL_VERSION:
        raise RuntimeError("Gross9 structural-clock protocol drift")
    authority = manifest.get("authority", {})
    if authority.get("sha256") != gross9.ANCHOR_SHA256 or authority.get("weights") != gross9.EXPECTED_WEIGHTS:
        raise RuntimeError("Gross9 authority binding drift")
    if manifest.get("all_authoritative_counts_verified") is not True:
        raise RuntimeError("Gross9 authoritative counts were not verified")
    if set(manifest.get("clocks", {})) != set(gross9.EXPECTED_WEIGHTS):
        raise RuntimeError("Gross9 complete sleeve roster drift")
    return preregistration, support, manifest


def load_clock(path: str | Path, *, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path, compression="gzip")
    if not CLOCK_COLUMNS.issubset(frame.columns):
        raise RuntimeError(f"{label} clock schema drift")
    result = frame[["entry_time", "exit_time", "side"]].copy()
    result["entry_time"] = pd.to_datetime(result["entry_time"], utc=True, errors="raise")
    result["exit_time"] = pd.to_datetime(result["exit_time"], utc=True, errors="raise")
    result["side"] = pd.to_numeric(result["side"], errors="raise").astype(int)
    if set(result.side) - {-1, 1}:
        raise RuntimeError(f"{label} side drift")
    if not (result.exit_time >= result.entry_time).all():
        raise RuntimeError(f"{label} negative interval")
    for column in ("entry_time", "exit_time"):
        nanoseconds = result[column].astype("int64")
        if (nanoseconds % BAR.value != 0).any():
            raise RuntimeError(f"{label} is not aligned to the UTC 5m grid")
    if result.duplicated().any():
        raise RuntimeError(f"{label} duplicate interval")
    return result.sort_values(["entry_time", "exit_time", "side"], kind="mergesort").reset_index(drop=True)


def fully_contained(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[(frame.entry_time >= WINDOW_START) & (frame.exit_time <= WINDOW_END)].reset_index(drop=True)


def exact_entry_jaccard(candidate: Sequence[pd.Timestamp], comparator: Sequence[pd.Timestamp]) -> float:
    left, right = set(candidate), set(comparator)
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def optimal_near_matches(
    candidate: Sequence[pd.Timestamp], comparator: Sequence[pd.Timestamp]
) -> tuple[tuple[tuple[pd.Timestamp, pd.Timestamp], ...], int]:
    """Return max-cardinality, then minimum-total-lag non-crossing matches."""

    left = tuple(sorted(set(candidate)))
    right = tuple(sorted(set(comparator)))
    rows, columns = len(left), len(right)
    cardinality = np.zeros((rows + 1, columns + 1), dtype=np.int32)
    lag = np.zeros((rows + 1, columns + 1), dtype=np.int64)
    choice = np.full((rows, columns), 1, dtype=np.int8)
    limit_seconds = int(NEAR_LIMIT.total_seconds())

    def rank(item: tuple[int, int, int]) -> tuple[int, int, int]:
        count, total_lag, priority = item
        return (-count, total_lag, priority)

    for i in range(rows - 1, -1, -1):
        for j in range(columns - 1, -1, -1):
            options = [
                (int(cardinality[i + 1, j]), int(lag[i + 1, j]), 1),
                (int(cardinality[i, j + 1]), int(lag[i, j + 1]), 2),
            ]
            seconds = int(abs((left[i] - right[j]).total_seconds()))
            if seconds <= limit_seconds:
                options.append(
                    (
                        1 + int(cardinality[i + 1, j + 1]),
                        seconds + int(lag[i + 1, j + 1]),
                        0,
                    )
                )
            best = min(options, key=rank)
            cardinality[i, j], lag[i, j], choice[i, j] = best

    matches: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    i = j = 0
    while i < rows and j < columns:
        selected = int(choice[i, j])
        if selected == 0:
            matches.append((left[i], right[j]))
            i += 1
            j += 1
        elif selected == 1:
            i += 1
        else:
            j += 1
    return tuple(matches), int(lag[0, 0])


def signed_exposure(frame: pd.DataFrame) -> np.ndarray:
    bar_count = int((WINDOW_END - WINDOW_START) / BAR)
    changes = np.zeros(bar_count + 1, dtype=np.int32)
    for row in frame.itertuples(index=False):
        start = int((row.entry_time - WINDOW_START) / BAR)
        end = int((row.exit_time - WINDOW_START) / BAR)
        changes[start] += int(row.side)
        changes[end] -= int(row.side)
    return np.cumsum(changes[:-1], dtype=np.int32)


def occupied_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left_occupied, right_occupied = left != 0, right != 0
    union = left_occupied | right_occupied
    return float(np.count_nonzero(left_occupied & right_occupied) / np.count_nonzero(union)) if union.any() else 0.0


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    x = left.astype(np.float64, copy=False)
    y = right.astype(np.float64, copy=False)
    count = len(x)
    numerator = count * float(np.dot(x, y)) - float(x.sum() * y.sum())
    left_scale = count * float(np.dot(x, x)) - float(x.sum() ** 2)
    right_scale = count * float(np.dot(y, y)) - float(y.sum() ** 2)
    denominator = math.sqrt(left_scale * right_scale)
    if denominator == 0.0:
        raise RuntimeError("signed-exposure Pearson is undefined")
    return numerator / denominator


def evaluate_pair(candidate: pd.DataFrame, comparator: pd.DataFrame) -> dict[str, Any]:
    candidate = fully_contained(candidate)
    comparator = fully_contained(comparator)
    candidate_entries = tuple(candidate.entry_time)
    comparator_entries = tuple(comparator.entry_time)
    if not candidate_entries or not comparator_entries:
        raise RuntimeError("Gross9 novelty common-window clock is empty")
    matches, total_lag_seconds = optimal_near_matches(candidate_entries, comparator_entries)
    denominator = min(len(set(candidate_entries)), len(set(comparator_entries)))
    candidate_exposure = signed_exposure(candidate)
    comparator_exposure = signed_exposure(comparator)
    correlation = pearson(candidate_exposure, comparator_exposure)
    metrics = {
        "exact_entry_jaccard": exact_entry_jaccard(candidate_entries, comparator_entries),
        "one_to_one_6h_max_matched_share": len(matches) / denominator,
        "occupied_5m_bar_jaccard": occupied_jaccard(candidate_exposure, comparator_exposure),
        "signed_exposure_pearson": correlation,
        "absolute_signed_exposure_pearson": abs(correlation),
    }
    checks = {name: metrics[name] <= limit for name, limit in LIMITS.items()}
    match_rows = [
        {
            "candidate_entry": left.isoformat(),
            "comparator_entry": right.isoformat(),
            "absolute_lag_seconds": int(abs((left - right).total_seconds())),
        }
        for left, right in matches
    ]
    return {
        "common_window": ["2023-07-01T00:00:00Z", "2025-01-01T00:00:00Z"],
        "containment_policy": "entry >= start AND exit <= end; no clip, shift, or split",
        "candidate_rows": len(candidate),
        "comparator_rows": len(comparator),
        "matching": {
            "objective": "maximum cardinality, then minimum total absolute lag, deterministic non-crossing UTC order",
            "maximum_cardinality": len(matches),
            "minimum_total_absolute_lag_seconds": total_lag_seconds,
            "matched_pairs_sha256": canonical_hash(match_rows),
        },
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
    }


def run(output: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    preregistration, support, manifest = _load_frozen_controls()
    candidate = load_clock(PRIMARY_CLOCK, label="OVEPR primary")
    sleeve_results: dict[str, Any] = {}
    for sleeve in gross9.EXPECTED_WEIGHTS:
        record = manifest["clocks"][sleeve]
        path = Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Gross9 clock hash drift: {sleeve}")
        comparator = load_clock(path, label=f"Gross9 {sleeve}")
        expected_total = sum(gross9.EXPECTED_COUNTS[sleeve].values())
        if len(comparator) != record["rows"] or len(comparator) != expected_total:
            raise RuntimeError(f"Gross9 clock count drift: {sleeve}")
        sleeve_results[sleeve] = evaluate_pair(candidate, comparator)

    every_gross9_sleeve_passed = all(result["passed"] for result in sleeve_results.values())
    source_support_passed = support.get("support_passed") is True
    named_novelty_passed = support.get("named_family_novelty_passed") is True
    advance = source_support_passed and named_novelty_passed and every_gross9_sleeve_passed
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "preregistration": {
            "path": PREREGISTRATION.as_posix(),
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": preregistration["manifest_hash"],
        },
        "supersedes_source_support_result": {
            "path": SOURCE_SUPPORT.as_posix(),
            "sha256": SOURCE_SUPPORT_SHA256,
            "manifest_hash": support["manifest_hash"],
            "predecessor_gross9_novelty_status": "pending",
            "predecessor_mutated": False,
        },
        "gross9_structural_clocks": {
            "path": GROSS9_MANIFEST.as_posix(),
            "sha256": sha256_file(GROSS9_MANIFEST),
            "manifest_hash": manifest["manifest_hash"],
            "authority_sha256": gross9.ANCHOR_SHA256,
            "complete_roster": list(gross9.EXPECTED_WEIGHTS),
        },
        "evidence_boundary": {
            "ovepr_clock_rows_opened": len(candidate),
            "gross9_structural_clock_rows_opened": sum(record["rows"] for record in manifest["clocks"].values()),
            "btc_execution_rows_opened": 0,
            "btc_price_or_return_rows_opened": 0,
            "funding_rows_opened": 0,
            "economic_outcome_rows_opened": 0,
            "portfolio_return_or_pnl_metrics_computed": False,
            "outcomes_opened": False,
        },
        "limits": LIMITS,
        "gross9_sleeves": sleeve_results,
        "source_support_passed": source_support_passed,
        "named_family_novelty_passed": named_novelty_passed,
        "every_gross9_sleeve_passed": every_gross9_sleeve_passed,
        "gross9_novelty_status": "passed" if every_gross9_sleeve_passed else "failed",
        "advance_to_economic_outcomes": advance,
        "failure_action": None if advance else "reject OVEPR-24 unchanged before economic outcomes",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    result = run(arguments.output)
    print(json.dumps({"output": str(arguments.output), "gross9_novelty_status": result["gross9_novelty_status"], "advance_to_economic_outcomes": result["advance_to_economic_outcomes"]}))


if __name__ == "__main__":
    main()
