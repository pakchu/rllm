"""Train-only Gross9 structural novelty gate for G9ASYNCACTIVEVETO-8.

Evidence-boundary strict evaluator for the active-opposite-veto family.  It
opens committed control artifacts, source-supported train clocks, and frozen
Gross9 structural clocks only.  It represents all 72 preregistered candidates,
evaluates only the 14 candidates that pass source + exact-duplicate gates, and
never opens market prices, funding, returns, PnL, or economic outcomes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import build_gross9_async_active_veto_train_clocks as source_support
from training import evaluate_gross9_async_pair_train_novelty as pair_novelty
from training import export_gross9_structural_clocks as gross9
from training import preregister_gross9_async_active_veto_search as prereg


POLICY_ID = prereg.POLICY_ID
PROTOCOL_VERSION = "gross9_async_active_veto_train_gross9_novelty_v1"
PREREGISTRATION = prereg.DEFAULT_OUTPUT
PREREGISTRATION_SHA256 = "5bb0abae46a5716451b07268a268cdd112a78829786772c4aeec8bc43f383f25"
PREREGISTRATION_MANIFEST_HASH = "871c7fb8c8825cb30c0967cab46a2a8cc7342f46f37c673372b45d2501d6aa6e"
SOURCE_SUPPORT = source_support.RESULT
SOURCE_SUPPORT_SHA256 = "ee966e59e219886b561a23e605cf225f44d393128f210a360048addfeba42f20"
SOURCE_SUPPORT_MANIFEST_HASH = "ec32caa65a0945fc73b6d863cb1b3fa810f4c58ffd8aed68408fff949e4d6f32"
GROSS9_MANIFEST = gross9.DEFAULT_MANIFEST
GROSS9_MANIFEST_SHA256 = pair_novelty.GROSS9_MANIFEST_SHA256
GROSS9_MANIFEST_HASH = pair_novelty.GROSS9_MANIFEST_HASH
OUTPUT = Path("results/gross9_async_active_veto_train_gross9_novelty_2026-09-02.json")
TRAIN_START = pair_novelty.TRAIN_START
TRAIN_END = pair_novelty.TRAIN_END
LIMITS = dict(pair_novelty.LIMITS)
EXPECTED_SUPPORTED_COUNT = 14


def sha256_file(path: str | Path) -> str:
    return pair_novelty.sha256_file(path)


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


def _assert_source_boundary_closed(boundary: Mapping[str, Any]) -> None:
    closed_false_flags = (
        "gross9_rows_opened",
        "market_rows_opened",
        "entry_exit_prices_opened",
        "funding_opened",
        "returns_or_pnl_opened",
        "economic_outcomes_opened",
        "base_control_economic_outcomes_opened",
    )
    for flag in closed_false_flags:
        if boundary.get(flag) is not False:
            raise RuntimeError(f"{POLICY_ID} source-support boundary already opened outcomes: {flag}")
    if boundary.get("oos_component_rows_materialized") != 0:
        raise RuntimeError(f"{POLICY_ID} source-support boundary materialized OOS component rows")
    if boundary.get("component_clock_rows_materialized_train_prefix_only") is not True:
        raise RuntimeError(f"{POLICY_ID} source-support was not train-prefix-only")


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
    prereg_receipt = support.get("preregistration", {})
    if (
        support.get("policy_id") != POLICY_ID
        or support.get("protocol_version") != source_support.PROTOCOL_VERSION
        or prereg_receipt.get("sha256") != PREREGISTRATION_SHA256
        or prereg_receipt.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH
        or prereg_receipt.get("status") != "validated_against_committed_preregistration"
        or prereg_receipt.get("prior_source_support_artifacts_cross_checked") is not True
        or prereg_receipt.get("research_boundary_disclosure_cross_checked") is not True
        or support.get("candidate_family") != list(prereg.CANDIDATE_FAMILY)
        or support.get("candidate_family_size") != prereg.FAMILY_SIZE
        or support.get("decision") != "pass_supported_active_veto_candidates_to_gross9_novelty"
    ):
        raise RuntimeError(f"{POLICY_ID} source-support eligibility drift")
    _assert_source_boundary_closed(support.get("evidence_boundary", {}))
    receipt = support.get("preliminary_source_materialization_receipt", {})
    expected_receipt = prereg.PRELIMINARY_SOURCE_MATERIALIZATION_RECEIPT
    if not isinstance(receipt, Mapping):
        raise RuntimeError(f"{POLICY_ID} preliminary source-support receipt missing")
    for key in ("commit", "path", "sha256", "manifest_hash", "builder", "placeholder_builder_value"):
        if receipt.get(key) != expected_receipt.get(key):
            raise RuntimeError(f"{POLICY_ID} preliminary source-support receipt drift: {key}")
    placeholder = expected_receipt.get("preregistration_artifact_with_placeholder_builder_binding", {})
    if (
        receipt.get("placeholder_preregistration_sha256") != placeholder.get("sha256")
        or receipt.get("placeholder_preregistration_manifest_hash") != placeholder.get("manifest_hash")
        or receipt.get("passed_candidates") != expected_receipt.get("support_count_disclosure", {}).get("passed_candidates")
    ):
        raise RuntimeError(f"{POLICY_ID} preliminary source-support receipt drift")

    candidates = support.get("candidates")
    if not isinstance(candidates, dict) or set(candidates) != set(prereg.CANDIDATE_FAMILY) or len(candidates) != prereg.FAMILY_SIZE:
        raise RuntimeError(f"{POLICY_ID} source-support 72-row family drift")
    recomputed_passed: list[str] = []
    for candidate in prereg.CANDIDATE_FAMILY:
        row = candidates[candidate]
        duplicate_gate = row.get("duplicate_gate", {})
        exact_duplicate_pass = duplicate_gate.get("rejected") is False
        source_pass = row.get("support_passed") is True
        if source_pass:
            recomputed_passed.append(candidate)
            if not exact_duplicate_pass:
                raise RuntimeError(f"{POLICY_ID} source-supported duplicate-gate projection drift: {candidate}")
            if row.get("advance_to_gross9_novelty") is not True or row.get("advance_to_economic_outcomes") is not False:
                raise RuntimeError(f"{POLICY_ID} source-supported advancement drift: {candidate}")
            checks = row.get("support_checks", {})
            if any(value is not True for value in checks.values()):
                raise RuntimeError(f"{POLICY_ID} source-supported check projection drift: {candidate}")
            if int(row.get("construction_diagnostics", {}).get("opposite_latest_veto_suppressions", 0)) < 1:
                raise RuntimeError(f"{POLICY_ID} source-supported suppression gate projection drift: {candidate}")
        elif row.get("advance_to_gross9_novelty") is True or row.get("advance_to_economic_outcomes") is True:
            raise RuntimeError(f"{POLICY_ID} unsupported candidate advancement drift: {candidate}")
    duplicate_summary = support.get("duplicate_gate_summary", {})
    rejected = set(duplicate_summary.get("rejected_candidates", []))
    if rejected != {name for name, row in candidates.items() if row.get("duplicate_gate", {}).get("rejected") is True}:
        raise RuntimeError(f"{POLICY_ID} duplicate-gate summary projection drift")
    if support.get("passed_candidates") != recomputed_passed or len(recomputed_passed) != EXPECTED_SUPPORTED_COUNT:
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


def load_candidate_clock(clock_record: Mapping[str, Any], candidate: str) -> pd.DataFrame:
    path = Path(str(clock_record["path"]))
    if sha256_file(path) != clock_record.get("sha256"):
        raise RuntimeError(f"{POLICY_ID} candidate clock hash drift: {candidate}")
    frame = pair_novelty.load_clock(path, label=f"{POLICY_ID} {candidate}")
    common = pair_novelty.train_contained(frame)
    if len(frame) != int(clock_record.get("rows", -1)) or len(common) != int(clock_record.get("rows", -1)):
        raise RuntimeError(f"{POLICY_ID} candidate clock row/window drift: {candidate}")
    return common


def load_gross9_train_clock(sleeve: str, record: Mapping[str, Any]) -> tuple[pd.DataFrame, int, int]:
    return pair_novelty.load_gross9_train_clock(sleeve, record)


def evaluate_candidate_train(candidate_clock: pd.DataFrame, comparator: pd.DataFrame) -> dict[str, Any]:
    return pair_novelty.evaluate_pair_train(candidate_clock, comparator)


def evaluate_candidate(candidate: str, clock_record: Mapping[str, Any], gross9_clocks: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    candidate_clock = load_candidate_clock(clock_record, candidate)
    sleeve_results = {sleeve: evaluate_candidate_train(candidate_clock, gross9_clocks[sleeve]) for sleeve in gross9.EXPECTED_WEIGHTS}
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

    candidates: dict[str, Any] = {}
    evaluated_candidates: list[str] = []
    passed_candidates: list[str] = []
    for candidate in prereg.CANDIDATE_FAMILY:
        support_record = support["candidates"][candidate]
        duplicate_rejected = support_record["duplicate_gate"]["rejected"] is True
        source_pass = support_record.get("support_passed") is True
        row: dict[str, Any] = {
            "candidate": candidate,
            "components": support_record["components"],
            "clock": support_record["clock"],
            "construction_diagnostics": support_record["construction_diagnostics"],
            "duplicate_gate": support_record["duplicate_gate"],
            "exact_duplicate_pass": not duplicate_rejected,
            "source_pass": source_pass,
            "source_support": support_record["support"],
            "source_support_checks": support_record["support_checks"],
            "gross9_pass": False,
            "gross9_novelty_status": "not_evaluated_source_or_exact_duplicate_failed",
            "advance_to_train_economics": False,
            "decision": "terminal_source_or_exact_duplicate_reject",
        }
        if source_pass:
            evaluated_candidates.append(candidate)
            row.update(evaluate_candidate(candidate, support_record["clock"], gross9_clocks))
            if row["gross9_pass"] is True:
                passed_candidates.append(candidate)
        candidates[candidate] = row

    source_supported_rows = sum(int(support["candidates"][candidate]["clock"]["rows"]) for candidate in support["passed_candidates"])
    evaluated_rows = sum(int(candidates[candidate].get("candidate_clock_rows_opened", 0)) for candidate in support["passed_candidates"])
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
        "source_supported_candidates": list(support["passed_candidates"]),
        "source_supported_candidate_count": len(support["passed_candidates"]),
        "source_supported_candidate_rows_opened": source_supported_rows,
        "novelty_evaluated_candidates": evaluated_candidates,
        "gross9_novelty_evaluated_candidate_count": len(evaluated_candidates),
        "gross9_novelty_passed_candidates": passed_candidates,
        "gross9_novelty_passed_candidate_count": len(passed_candidates),
        "gross9_passed_any_candidate": bool(passed_candidates),
        "window": {
            "start_inclusive": _iso_z(TRAIN_START),
            "end_exclusive": _iso_z(TRAIN_END),
            "containment_policy": "split == train AND entry >= start AND entry < end AND exit <= end; no clip, shift, or split",
        },
        "limits": LIMITS,
        "candidates": candidates,
        "evidence_boundary": {
            "candidate_family_rows_counted": prereg.FAMILY_SIZE,
            "source_supported_candidate_clock_rows_opened": source_supported_rows,
            "source_supported_candidate_clock_rows_evaluated": evaluated_rows,
            "unsupported_candidate_clock_rows_opened_for_novelty": 0,
            "exact_duplicate_gate_projected_for_all_72": True,
            "source_and_exact_duplicate_supported_candidates_expected": EXPECTED_SUPPORTED_COUNT,
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
        "advance_to_economic_outcomes": bool(passed_candidates),
        "decision": "pass_gross9_novel_candidates_to_train_economics" if passed_candidates else "terminal_no_gross9_novel_candidates",
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
                "evaluated_candidates": result["novelty_evaluated_candidates"],
                "passed_candidates": result["gross9_novelty_passed_candidates"],
                "decision": result["decision"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
