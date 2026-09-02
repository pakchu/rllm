"""Train-only Gross9 structural novelty gate for G9ASYNCPAIR-8.

This stage is evidence-boundary strict: it opens committed control artifacts,
source-supported async pair clocks, and frozen Gross9 structural clocks only.
It represents all 36 preregistered candidates, evaluates only the seven pairs
that passed source support, and never opens market prices, funding, returns,
PnL, or economic outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import build_gross9_async_pair_train_clocks as source_support
from training import export_gross9_structural_clocks as gross9
from training import preregister_gross9_async_pair_search as prereg
from training import evaluate_options_led_volatility_expansion_premium_relay_gross9_novelty as metric


POLICY_ID = prereg.POLICY_ID
PROTOCOL_VERSION = "gross9_async_pair_train_gross9_novelty_v1"
PREREGISTRATION = prereg.DEFAULT_OUTPUT
PREREGISTRATION_SHA256 = "868ea5d80bc8e15adeff1303c142553c21c2ceedc8d425523fc85ae57371a622"
PREREGISTRATION_MANIFEST_HASH = "a9aafc9e8c563a399e5970beccce9aa683e38f17c343904a805d66185fbc467c"
SOURCE_SUPPORT = source_support.RESULT
SOURCE_SUPPORT_SHA256 = "c6d3929f282ba1075c2ebc091e4bc62164b923a038bce94de32884aaf7ff0009"
SOURCE_SUPPORT_MANIFEST_HASH = "b92d3afb7a3539cdd194eddc1ab09bc65068716135d0bca575db0531ac450011"
GROSS9_MANIFEST = gross9.DEFAULT_MANIFEST
GROSS9_MANIFEST_SHA256 = "5433812da786a959cda1cfcf4825bc2e4a228ea8152a4b8cce1e867f29adf073"
GROSS9_MANIFEST_HASH = "c1f7c2096cea035d053dd3d7b887b13f3220b6d96ddb99893b5be26cb44ae650"
OUTPUT = Path("results/gross9_async_pair_train_gross9_novelty_2026-09-02.json")
TRAIN_START = pd.Timestamp("2023-07-01T00:00:00Z")
TRAIN_END = pd.Timestamp("2024-01-01T00:00:00Z")
BAR = pd.Timedelta(minutes=5)
LIMITS = {
    "exact_entry_jaccard": 0.10,
    "one_to_one_6h_max_matched_share": 0.35,
    "occupied_5m_bar_jaccard": 0.25,
    "absolute_signed_exposure_pearson": 0.35,
}
CLOCK_COLUMNS = {"split", "entry_time", "exit_time", "side"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return gross9.canonical_hash(value)


def _load_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{POLICY_ID} JSON artifact is not an object: {path}")
    return value


def _verify_manifest_hash(value: Mapping[str, Any], label: str) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError(f"{POLICY_ID} {label} manifest drift")


def _verify_sha(path: Path, expected: str, label: str) -> None:
    if sha256_file(path) != expected:
        raise RuntimeError(f"{POLICY_ID} {label} hash drift")


def _iso_z(ts: pd.Timestamp) -> str:
    return ts.isoformat().replace("+00:00", "Z")


def _expected_gross9_gate_payload() -> dict[str, Any]:
    return {
        "exact_entry_jaccard_max": 0.1,
        "candidate_near_6h_share_max": 0.35,
        "occupied_5m_jaccard_max": 0.25,
        "absolute_signed_exposure_pearson_max": 0.35,
        "must_pass_before_economics": True,
    }


def load_frozen_controls() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _verify_sha(PREREGISTRATION, PREREGISTRATION_SHA256, "preregistration file")
    registration = _load_json_object(PREREGISTRATION)
    _verify_manifest_hash(registration, "preregistration")
    if registration.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError(f"{POLICY_ID} preregistration manifest binding drift")
    prereg.validate(registration)
    if registration != prereg.build():
        raise RuntimeError(f"{POLICY_ID} preregistration artifact differs from code")
    gates = dict(registration.get("gross9_novelty_gates", {}))
    comparator = gates.pop("comparator", {})
    if gates != _expected_gross9_gate_payload():
        raise RuntimeError(f"{POLICY_ID} Gross9 novelty gate drift")
    if (
        comparator.get("path") != GROSS9_MANIFEST.as_posix()
        or comparator.get("sha256") != GROSS9_MANIFEST_SHA256
        or comparator.get("manifest_hash") != GROSS9_MANIFEST_HASH
        or comparator.get("authority_sha256") != gross9.ANCHOR_SHA256
    ):
        raise RuntimeError(f"{POLICY_ID} preregistered Gross9 comparator drift")

    _verify_sha(SOURCE_SUPPORT, SOURCE_SUPPORT_SHA256, "source-support file")
    support = _load_json_object(SOURCE_SUPPORT)
    _verify_manifest_hash(support, "source support")
    if support.get("manifest_hash") != SOURCE_SUPPORT_MANIFEST_HASH:
        raise RuntimeError(f"{POLICY_ID} source-support manifest binding drift")
    if (
        support.get("policy_id") != POLICY_ID
        or support.get("protocol_version") != source_support.PROTOCOL_VERSION
        or support.get("preregistration", {}).get("sha256") != PREREGISTRATION_SHA256
        or support.get("candidate_family") != list(prereg.CANDIDATE_FAMILY)
        or support.get("candidate_family_size") != prereg.FAMILY_SIZE
        or support.get("decision") != "pass_supported_pairs_to_gross9_novelty"
    ):
        raise RuntimeError(f"{POLICY_ID} source-support eligibility drift")
    boundary = support.get("evidence_boundary", {})
    if (
        boundary.get("gross9_rows_opened") is not False
        or boundary.get("market_rows_opened") is not False
        or boundary.get("entry_exit_prices_opened") is not False
        or boundary.get("funding_opened") is not False
        or boundary.get("pair_combination_returns_or_pnl_opened") is not False
        or boundary.get("pair_combination_economic_outcomes_opened") is not False
    ):
        raise RuntimeError(f"{POLICY_ID} source-support boundary already opened outcomes")
    pairs = support.get("pairs")
    if not isinstance(pairs, dict) or set(pairs) != set(prereg.CANDIDATE_FAMILY) or len(pairs) != prereg.FAMILY_SIZE:
        raise RuntimeError(f"{POLICY_ID} source-support 36-row family drift")
    recomputed_passed: list[str] = []
    for candidate in prereg.CANDIDATE_FAMILY:
        row = pairs[candidate]
        source_pass = row.get("support_passed") is True
        if source_pass:
            recomputed_passed.append(candidate)
            if row.get("advance_to_gross9_novelty") is not True or row.get("advance_to_economic_outcomes") is not False:
                raise RuntimeError(f"{POLICY_ID} source-supported advancement drift: {candidate}")
        elif row.get("advance_to_gross9_novelty") is True or row.get("advance_to_economic_outcomes") is True:
            raise RuntimeError(f"{POLICY_ID} unsupported pair advancement drift: {candidate}")
    if support.get("passed_pairs") != recomputed_passed or len(recomputed_passed) != 7:
        raise RuntimeError(f"{POLICY_ID} source-supported roster drift")

    _verify_sha(GROSS9_MANIFEST, GROSS9_MANIFEST_SHA256, "Gross9 structural manifest file")
    manifest = _load_json_object(GROSS9_MANIFEST)
    _verify_manifest_hash(manifest, "Gross9 structural clocks")
    if manifest.get("manifest_hash") != GROSS9_MANIFEST_HASH:
        raise RuntimeError(f"{POLICY_ID} Gross9 manifest binding drift")
    if (
        manifest.get("protocol_version") != gross9.PROTOCOL_VERSION
        or manifest.get("authority", {}).get("sha256") != gross9.ANCHOR_SHA256
        or manifest.get("authority", {}).get("weights") != gross9.EXPECTED_WEIGHTS
        or manifest.get("all_authoritative_counts_verified") is not True
        or set(manifest.get("clocks", {})) != set(gross9.EXPECTED_WEIGHTS)
    ):
        raise RuntimeError(f"{POLICY_ID} Gross9 structural clock authority drift")
    return registration, support, manifest


def load_clock(path: str | Path, *, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path, compression="gzip")
    if not CLOCK_COLUMNS.issubset(frame.columns):
        raise RuntimeError(f"{label} clock schema drift")
    result = frame[["split", "entry_time", "exit_time", "side"]].copy()
    result["split"] = result["split"].astype(str)
    result["entry_time"] = pd.to_datetime(result["entry_time"], utc=True, errors="raise")
    result["exit_time"] = pd.to_datetime(result["exit_time"], utc=True, errors="raise")
    result["side"] = pd.to_numeric(result["side"], errors="raise").astype(int)
    if set(result["side"]) - {-1, 1}:
        raise RuntimeError(f"{label} side drift")
    if not result["exit_time"].ge(result["entry_time"]).all():
        raise RuntimeError(f"{label} negative interval")
    for column in ("entry_time", "exit_time"):
        if (result[column].astype("int64") % BAR.value).any():
            raise RuntimeError(f"{label} is not aligned to the UTC 5m grid")
    if result.duplicated().any():
        raise RuntimeError(f"{label} duplicate interval")
    return result.sort_values(["entry_time", "exit_time", "side", "split"], kind="mergesort").reset_index(drop=True)


def train_contained(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[
        frame["split"].eq("train")
        & frame["entry_time"].ge(TRAIN_START)
        & frame["entry_time"].lt(TRAIN_END)
        & frame["exit_time"].le(TRAIN_END)
    ].reset_index(drop=True)


def _split_value_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {str(key): int(value) for key, value in frame["split"].value_counts().sort_index().items()}


def load_candidate_clock(clock_record: Mapping[str, Any], candidate: str) -> pd.DataFrame:
    path = Path(str(clock_record["path"]))
    if sha256_file(path) != clock_record.get("sha256"):
        raise RuntimeError(f"{POLICY_ID} candidate clock hash drift: {candidate}")
    frame = load_clock(path, label=f"{POLICY_ID} {candidate}")
    common = train_contained(frame)
    if len(frame) != int(clock_record.get("rows", -1)) or len(common) != int(clock_record.get("rows", -1)):
        raise RuntimeError(f"{POLICY_ID} candidate clock row/window drift: {candidate}")
    return common


def load_gross9_train_clock(sleeve: str, record: Mapping[str, Any]) -> tuple[pd.DataFrame, int, int]:
    path = Path(str(record["path"]))
    if sha256_file(path) != record.get("sha256"):
        raise RuntimeError(f"{POLICY_ID} Gross9 clock hash drift: {sleeve}")
    frame = load_clock(path, label=f"Gross9 {sleeve}")
    expected_counts = {key: int(value) for key, value in gross9.EXPECTED_COUNTS[sleeve].items()}
    if (
        len(frame) != int(record.get("rows", -1))
        or len(frame) != sum(expected_counts.values())
        or dict(record.get("counts", {})) != expected_counts
        or _split_value_counts(frame) != expected_counts
    ):
        raise RuntimeError(f"{POLICY_ID} Gross9 split count drift: {sleeve}")
    common = train_contained(frame)
    if common.empty:
        raise RuntimeError(f"{POLICY_ID} Gross9 train common-window clock is empty: {sleeve}")
    return common, len(frame), len(common)


def signed_exposure_train(frame: pd.DataFrame) -> np.ndarray:
    bar_count = int((TRAIN_END - TRAIN_START) / BAR)
    changes = np.zeros(bar_count + 1, dtype=np.int32)
    for row in frame.itertuples(index=False):
        start = int((row.entry_time - TRAIN_START) / BAR)
        end = int((row.exit_time - TRAIN_START) / BAR)
        if start < 0 or end > bar_count or end < start:
            raise RuntimeError(f"{POLICY_ID} train exposure received invalid interval")
        changes[start] += int(row.side)
        changes[end] -= int(row.side)
    return np.cumsum(changes[:-1], dtype=np.int32)


def evaluate_pair_train(candidate_clock: pd.DataFrame, comparator: pd.DataFrame) -> dict[str, Any]:
    candidate = train_contained(candidate_clock)
    comp = train_contained(comparator)
    if candidate.empty or comp.empty:
        raise RuntimeError(f"{POLICY_ID} Gross9 novelty train-window clock is empty")
    candidate_entries = tuple(candidate["entry_time"])
    comparator_entries = tuple(comp["entry_time"])
    matches, total_lag_seconds = metric.optimal_near_matches(candidate_entries, comparator_entries)
    denominator = min(len(set(candidate_entries)), len(set(comparator_entries)))
    candidate_exposure = signed_exposure_train(candidate)
    comparator_exposure = signed_exposure_train(comp)
    correlation = metric.pearson(candidate_exposure, comparator_exposure)
    metrics = {
        "exact_entry_jaccard": metric.exact_entry_jaccard(candidate_entries, comparator_entries),
        "one_to_one_6h_max_matched_share": len(matches) / denominator,
        "occupied_5m_bar_jaccard": metric.occupied_jaccard(candidate_exposure, comparator_exposure),
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
        "common_window": [_iso_z(TRAIN_START), _iso_z(TRAIN_END)],
        "containment_policy": "split == train AND entry >= start AND entry < end AND exit <= end; no clip, shift, or split",
        "candidate_rows": int(len(candidate)),
        "comparator_rows": int(len(comp)),
        "matching": {
            "objective": "maximum cardinality, then minimum total absolute lag, deterministic non-crossing UTC order",
            "maximum_cardinality": int(len(matches)),
            "minimum_total_absolute_lag_seconds": int(total_lag_seconds),
            "matched_pairs_sha256": canonical_hash(match_rows),
        },
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
    }


def evaluate_candidate(candidate: str, clock_record: Mapping[str, Any], gross9_clocks: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    candidate_clock = load_candidate_clock(clock_record, candidate)
    sleeve_results = {sleeve: evaluate_pair_train(candidate_clock, gross9_clocks[sleeve]) for sleeve in gross9.EXPECTED_WEIGHTS}
    every_passed = all(item["passed"] for item in sleeve_results.values())
    return {
        "candidate_clock_rows_opened": int(len(candidate_clock)),
        "gross9_sleeves": sleeve_results,
        "gross9_pass": every_passed,
        "gross9_novelty_status": "passed" if every_passed else "failed",
        "advance_to_train_economics": every_passed,
        "decision": "pass_to_train_economics" if every_passed else "terminal_gross9_novelty_reject",
    }


def run(output: str | Path = OUTPUT) -> dict[str, Any]:
    registration, support, manifest = load_frozen_controls()
    gross9_clocks: dict[str, pd.DataFrame] = {}
    gross9_counts: dict[str, Any] = {}
    for sleeve in gross9.EXPECTED_WEIGHTS:
        record = manifest["clocks"][sleeve]
        clock, total_rows, common_rows = load_gross9_train_clock(sleeve, record)
        gross9_clocks[sleeve] = clock
        gross9_counts[sleeve] = {
            "path": record["path"],
            "sha256": record["sha256"],
            "manifest_total_rows": int(record["rows"]),
            "full_file_rows_verified": int(total_rows),
            "split_counts_verified": {key: int(value) for key, value in record["counts"].items()},
            "train_common_window_rows_evaluated": int(common_rows),
        }

    pairs: dict[str, Any] = {}
    evaluated_pairs: list[str] = []
    passed_pairs: list[str] = []
    for candidate in prereg.CANDIDATE_FAMILY:
        support_record = support["pairs"][candidate]
        source_pass = support_record.get("support_passed") is True
        row: dict[str, Any] = {
            "candidate": candidate,
            "components": support_record["components"],
            "clock": support_record["clock"],
            "source_pass": source_pass,
            "source_support": support_record["support"],
            "source_support_checks": support_record["support_checks"],
            "gross9_pass": False,
            "gross9_novelty_status": "not_evaluated_source_support_failed",
            "advance_to_train_economics": False,
            "decision": "terminal_source_support_reject",
        }
        if source_pass:
            evaluated_pairs.append(candidate)
            row.update(evaluate_candidate(candidate, support_record["clock"], gross9_clocks))
            if row["gross9_pass"] is True:
                passed_pairs.append(candidate)
        pairs[candidate] = row

    source_supported_rows = sum(int(support["pairs"][candidate]["clock"]["rows"]) for candidate in support["passed_pairs"])
    evaluated_rows = sum(int(pairs[candidate].get("candidate_clock_rows_opened", 0)) for candidate in support["passed_pairs"])
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "as_of_date": "2026-09-02",
        "implementation": {"evaluator": {"path": Path(__file__).as_posix(), "sha256": sha256_file(Path(__file__))}},
        "preregistration": {"path": str(PREREGISTRATION), "sha256": PREREGISTRATION_SHA256, "manifest_hash": registration["manifest_hash"]},
        "source_support": {"path": str(SOURCE_SUPPORT), "sha256": SOURCE_SUPPORT_SHA256, "manifest_hash": support["manifest_hash"], "predecessor_mutated": False},
        "gross9_structural_clocks": {
            "path": str(GROSS9_MANIFEST),
            "sha256": GROSS9_MANIFEST_SHA256,
            "manifest_hash": manifest["manifest_hash"],
            "authority_sha256": gross9.ANCHOR_SHA256,
            "complete_roster": list(gross9.EXPECTED_WEIGHTS),
            "counts": gross9_counts,
        },
        "candidate_family": list(prereg.CANDIDATE_FAMILY),
        "candidate_family_size": prereg.FAMILY_SIZE,
        "source_supported_pairs": list(support["passed_pairs"]),
        "source_supported_pair_count": len(support["passed_pairs"]),
        "source_supported_pair_rows_opened": source_supported_rows,
        "novelty_evaluated_pairs": evaluated_pairs,
        "gross9_novelty_evaluated_pair_count": len(evaluated_pairs),
        "gross9_novelty_passed_pairs": passed_pairs,
        "gross9_novelty_passed_pair_count": len(passed_pairs),
        "gross9_passed_any_pair": bool(passed_pairs),
        "window": {
            "start_inclusive": _iso_z(TRAIN_START),
            "end_exclusive": _iso_z(TRAIN_END),
            "containment_policy": "split == train AND entry >= start AND entry < end AND exit <= end; no clip, shift, or split",
        },
        "limits": LIMITS,
        "pairs": pairs,
        "evidence_boundary": {
            "candidate_family_rows_counted": prereg.FAMILY_SIZE,
            "source_supported_pair_clock_rows_opened": source_supported_rows,
            "source_supported_pair_clock_rows_evaluated": evaluated_rows,
            "unsupported_pair_clock_rows_opened_for_novelty": 0,
            "gross9_structural_clock_rows_opened_full_files_for_count_hash_split_verification": sum(item["full_file_rows_verified"] for item in gross9_counts.values()),
            "gross9_structural_clock_rows_evaluated_after_train_filter": sum(item["train_common_window_rows_evaluated"] for item in gross9_counts.values()),
            "btc_execution_rows_opened": 0,
            "btc_price_or_return_rows_opened": 0,
            "entry_exit_prices_opened": 0,
            "funding_rows_opened": 0,
            "economic_outcome_rows_opened": 0,
            "portfolio_return_or_pnl_metrics_computed": False,
            "outcomes_opened": False,
        },
        "advance_to_economic_outcomes": bool(passed_pairs),
        "decision": "pass_gross9_novel_pairs_to_train_economics" if passed_pairs else "terminal_no_gross9_novel_pairs",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "evaluated_pairs": result["novelty_evaluated_pairs"],
                "passed_pairs": result["gross9_novelty_passed_pairs"],
                "decision": result["decision"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
