"""Train-only Gross9 structural novelty gate for G9ASYNCHANDOFF-8.

This stage is evidence-boundary strict: it opens committed handoff control
artifacts, source-supported handoff train clocks, and frozen Gross9 structural
clocks only.  It represents all 36 preregistered candidates, evaluates only
source-supported pairs, and never opens market prices, funding, returns, PnL,
or economic outcomes.
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

from training import build_gross9_async_opposition_handoff_train_clocks as source_support
from training import evaluate_gross9_async_pair_train_novelty as pair_novelty
from training import export_gross9_structural_clocks as gross9
from training import preregister_gross9_async_opposition_handoff_search as prereg


POLICY_ID = prereg.POLICY_ID
PROTOCOL_VERSION = "gross9_async_opposition_handoff_train_gross9_novelty_v1"
PREREGISTRATION = prereg.DEFAULT_OUTPUT
PREREGISTRATION_SHA256 = "b6a98fc4d24fb255baccfe6128b5c322ebaa7eddc1cb7b1e15584bac5ffaa903"
PREREGISTRATION_MANIFEST_HASH = "95e2f274b34afa6b4417c29a91b914aa29b5cf2df10f28bf5e4fff3dd558f477"
SOURCE_SUPPORT = source_support.RESULT
SOURCE_SUPPORT_SHA256 = "a8982c1b6e155f65f76af4559ca2d01b2a7824cb5c58524a260b72beb997f754"
SOURCE_SUPPORT_MANIFEST_HASH = "92501aa4c921bba20d05378b6f658f33d6c712e8b3adb9f095940dd44ac3f3b0"
GROSS9_MANIFEST = gross9.DEFAULT_MANIFEST
GROSS9_MANIFEST_SHA256 = pair_novelty.GROSS9_MANIFEST_SHA256
GROSS9_MANIFEST_HASH = pair_novelty.GROSS9_MANIFEST_HASH
OUTPUT = Path("results/gross9_async_opposition_handoff_train_gross9_novelty_2026-09-02.json")
TRAIN_START = pair_novelty.TRAIN_START
TRAIN_END = pair_novelty.TRAIN_END
LIMITS = dict(pair_novelty.LIMITS)


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
        diagnostics = row.get("construction_diagnostics", {})
        if diagnostics.get("same_side_pre_reservation_entry_intersection") != 0:
            raise RuntimeError(f"{POLICY_ID} same-side pre-reservation intersection drift: {candidate}")
        if source_pass:
            recomputed_passed.append(candidate)
            if row.get("advance_to_gross9_novelty") is not True or row.get("advance_to_economic_outcomes") is not False:
                raise RuntimeError(f"{POLICY_ID} source-supported advancement drift: {candidate}")
            checks = row.get("support_checks", {})
            if any(value is not True for value in checks.values()):
                raise RuntimeError(f"{POLICY_ID} source-supported check projection drift: {candidate}")
        elif row.get("advance_to_gross9_novelty") is True or row.get("advance_to_economic_outcomes") is True:
            raise RuntimeError(f"{POLICY_ID} unsupported pair advancement drift: {candidate}")
    if support.get("passed_pairs") != recomputed_passed:
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


def evaluate_pair_train(candidate_clock: pd.DataFrame, comparator: pd.DataFrame) -> dict[str, Any]:
    return pair_novelty.evaluate_pair_train(candidate_clock, comparator)


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
            "construction_diagnostics": support_record["construction_diagnostics"],
            "same_side_pre_reservation_intersection_pass": support_record["construction_diagnostics"]["same_side_pre_reservation_entry_intersection"] == 0,
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
            "same_side_pre_reservation_intersection_required": 0,
            "same_side_pre_reservation_intersection_verified_for_all_36": True,
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
