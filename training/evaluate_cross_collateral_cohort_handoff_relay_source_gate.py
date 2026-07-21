"""Evaluate the outcome-blind prerequisite gate for CCHR-288.

The frozen comparator metadata is sufficient to reject an empty required
comparator before CCHR source incidence or any economic outcome is opened.
This evaluator therefore stops at that earliest irreversible gate and records
the exact failed members without reading a clock row, source value, price,
funding value, return, or PnL.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, cast

from training import cchr_comparator_clock_common as common
from training import (
    freeze_cross_collateral_cohort_handoff_relay_comparators as comparator_freeze,
)
from training import preregister_cchr_pure_clock_exports as export_prereg
from training import preregister_cross_collateral_cohort_handoff_relay as cchr


PROTOCOL_VERSION = "cross_collateral_cohort_handoff_relay_source_gate_v1"
POLICY_ID = cchr.POLICY_ID
AS_OF_DATE = "2026-07-21"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPOSITORY_ROOT / "results"
EVALUATOR_SOURCE = Path(
    "training/evaluate_cross_collateral_cohort_handoff_relay_source_gate.py"
)
DEFAULT_OUTPUT = Path(
    "results/cross_collateral_cohort_handoff_relay_source_gate_2026-07-21.json"
)

OUTCOME_BOUNDARY = {
    "master_preregistration_json_parsed": True,
    "comparator_freeze_json_parsed": True,
    "family_preregistration_json_parsed": True,
    "export_manifest_json_parsed": True,
    "published_artifact_json_readback": True,
    "pure_clock_bytes_hashed": True,
    "pure_clock_rows_read": 0,
    "legacy_comparator_rows_read": 0,
    "cchr_source_bytes_hashed": True,
    "cchr_source_values_read": 0,
    "cchr_incidence_rows_derived": 0,
    "market_rows_loaded": 0,
    "funding_rows_loaded": 0,
    "outcome_artifacts_parsed": 0,
    "return_or_pnl_fields_read": 0,
    "post_2023_rows_loaded": 0,
    "network_calls": 0,
    "subprocess_calls": 0,
}

TOP_LEVEL_KEYS = frozenset(
    {
        "protocol_version",
        "policy_id",
        "as_of_date",
        "preregistration",
        "comparator_freeze",
        "evaluator",
        "gate_contract",
        "family_precheck",
        "failed_required_members",
        "decision",
        "authorization",
        "outcomes_opened",
        "outcome_boundary",
        "manifest_hash",
    }
)


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return Path(os.path.abspath(candidate))


def _load_master() -> tuple[dict[str, Any], str]:
    path = export_prereg.MASTER_PREREGISTRATION
    before = common.sha256_file(path)
    payload = cchr._read_json(path)
    after = common.sha256_file(path)
    if before != after or before != export_prereg.MASTER_PREREGISTRATION_SHA256:
        raise RuntimeError("CCHR source gate master preregistration drift")
    cchr.validate_manifest(payload, verify_sources=False, expected_output=path)
    return payload, before


def _family_precheck(
    frozen: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    families = cast(Mapping[str, Mapping[str, Any]], frozen["generated_families"])
    summary: dict[str, Any] = {}
    failed_members: list[str] = []
    for family in export_prereg.FAMILIES:
        binding = families[family]
        metadata = cast(Mapping[str, Any], binding["clock_metadata"])
        required = cast(Mapping[str, Any], binding["required_bindings"])
        counts = cast(Mapping[str, Any], metadata["rows_by_candidate"])
        empty_members = sorted(
            candidate_id for candidate_id, rows in counts.items() if rows == 0
        )
        failed_members.extend(empty_members)
        coverage = cast(Mapping[str, Mapping[str, Any]], required["coverage"])
        summary[family] = {
            "required_member_count": required["member_count"],
            "clock_rows": metadata["rows"],
            "zero_row_member_count": len(empty_members),
            "zero_row_members": empty_members,
            "coverage_rows": {split: item["rows"] for split, item in coverage.items()},
            "precheck_pass": not empty_members,
        }
    return summary, sorted(failed_members)


def build_report() -> dict[str, Any]:
    master, master_sha256 = _load_master()
    frozen = comparator_freeze.load_freeze()
    if frozen["master_preregistration"] != {
        "path": str(export_prereg.MASTER_PREREGISTRATION),
        "sha256": master_sha256,
        "manifest_hash": master["manifest_hash"],
    }:
        raise RuntimeError("CCHR source gate comparator freeze/master drift")

    family_precheck, failed_members = _family_precheck(frozen)
    comparator_contract = cast(Mapping[str, Any], master["policy"])[
        "comparator_contract"
    ]
    if comparator_contract["zero_variance_action"] != "fail closed":
        raise RuntimeError("CCHR source gate zero-variance contract drift")
    if not failed_members:
        raise RuntimeError(
            "CCHR source gate precheck passed; real incidence evaluator is required"
        )

    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "as_of_date": AS_OF_DATE,
        "preregistration": {
            "path": str(export_prereg.MASTER_PREREGISTRATION),
            "sha256": master_sha256,
            "manifest_hash": master["manifest_hash"],
            "policy_hash": master["policy_hash"],
        },
        "comparator_freeze": {
            "path": str(comparator_freeze.DEFAULT_OUTPUT),
            "sha256": common.sha256_file(comparator_freeze.DEFAULT_OUTPUT),
            "manifest_hash": frozen["manifest_hash"],
        },
        "evaluator": {
            "path": str(EVALUATOR_SOURCE),
            "sha256": common.sha256_file(EVALUATOR_SOURCE),
        },
        "gate_contract": {
            "empty_required_comparator_action": "fail closed",
            "zero_variance_action": comparator_contract["zero_variance_action"],
            "failure_action": master["policy"]["support_gates"]["failure_action"],
            "repair_after_failure": False,
            "earliest_failure_short_circuit": True,
        },
        "family_precheck": family_precheck,
        "failed_required_members": failed_members,
        "decision": {
            "status": "retired_before_real_incidence",
            "pass": False,
            "reason": (
                "at least one required comparator member has an empty frozen "
                "clock, so novelty exposure is zero-variance and undefined"
            ),
            "failed_family": "far",
            "failed_member_count": len(failed_members),
            "cchr_source_incidence_opened": False,
            "economic_outcomes_opened": False,
            "repair_authorized": False,
        },
        "authorization": {
            "cchr_source_incidence_after_this_artifact": False,
            "outcome_evaluator": False,
            "post_2023_source_access": False,
            "next_action": "new independently preregistered alpha only",
        },
        "outcomes_opened": False,
        "outcome_boundary": dict(OUTCOME_BOUNDARY),
    }
    payload["manifest_hash"] = common.canonical_hash(payload)
    validate_report(payload, verify_files=False)
    return payload


def validate_report(
    payload: Mapping[str, Any],
    *,
    verify_files: bool = True,
) -> None:
    if frozenset(payload) != TOP_LEVEL_KEYS:
        raise RuntimeError("CCHR source gate top-level schema drift")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("CCHR source gate protocol drift")
    if payload.get("policy_id") != POLICY_ID or payload.get("as_of_date") != AS_OF_DATE:
        raise RuntimeError("CCHR source gate identity drift")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != common.canonical_hash(core):
        raise RuntimeError("CCHR source gate manifest hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("CCHR source gate opened outcomes")
    if payload.get("outcome_boundary") != OUTCOME_BOUNDARY:
        raise RuntimeError("CCHR source gate outcome boundary drift")
    decision = payload.get("decision")
    if not isinstance(decision, dict) or decision != {
        "status": "retired_before_real_incidence",
        "pass": False,
        "reason": (
            "at least one required comparator member has an empty frozen clock, "
            "so novelty exposure is zero-variance and undefined"
        ),
        "failed_family": "far",
        "failed_member_count": 12,
        "cchr_source_incidence_opened": False,
        "economic_outcomes_opened": False,
        "repair_authorized": False,
    }:
        raise RuntimeError("CCHR source gate decision drift")
    failed = payload.get("failed_required_members")
    expected_maps = {
        family: {
            candidate_id
            for candidate_id, definition in cchr.comparator_candidate_map().items()
            if definition["family"] == family
        }
        for family in export_prereg.FAMILIES
    }
    expected_failed = sorted(expected_maps["far"])
    if failed != expected_failed:
        raise RuntimeError("CCHR source gate failed-member drift")
    family_precheck = payload.get("family_precheck")
    if not isinstance(family_precheck, dict) or set(family_precheck) != set(
        export_prereg.FAMILIES
    ):
        raise RuntimeError("CCHR source gate family-precheck schema drift")
    observed_failed: list[str] = []
    for family, expected_members in expected_maps.items():
        item = family_precheck[family]
        if not isinstance(item, dict) or set(item) != {
            "required_member_count",
            "clock_rows",
            "zero_row_member_count",
            "zero_row_members",
            "coverage_rows",
            "precheck_pass",
        }:
            raise RuntimeError(f"{family} CCHR source gate precheck schema drift")
        zero_members = item["zero_row_members"]
        if not isinstance(zero_members, list) or not set(zero_members).issubset(
            expected_members
        ):
            raise RuntimeError(f"{family} CCHR source gate zero-member drift")
        expected_zero_members = expected_failed if family == "far" else []
        expected_pass = not zero_members
        if (
            zero_members != expected_zero_members
            or item["zero_row_member_count"] != len(zero_members)
            or item["required_member_count"] != len(expected_members)
            or item["precheck_pass"] is not expected_pass
        ):
            raise RuntimeError(f"{family} CCHR source gate precheck drift")
        if type(item["clock_rows"]) is not int or item["clock_rows"] < 0:
            raise RuntimeError(f"{family} CCHR source gate clock rows invalid")
        coverage_rows = item["coverage_rows"]
        if (
            not isinstance(coverage_rows, dict)
            or set(coverage_rows) != {"train", "selection"}
            or any(type(rows) is not int or rows < 0 for rows in coverage_rows.values())
            or sum(coverage_rows.values()) != item["clock_rows"]
        ):
            raise RuntimeError(f"{family} CCHR source gate coverage drift")
        observed_failed.extend(zero_members)
    if sorted(observed_failed) != expected_failed:
        raise RuntimeError("CCHR source gate failed-member/precheck mismatch")
    authorization = payload.get("authorization")
    if (
        not isinstance(authorization, dict)
        or authorization.get("outcome_evaluator") is not False
    ):
        raise RuntimeError("CCHR source gate authorization drift")
    if verify_files and payload != build_report():
        raise RuntimeError("CCHR source gate file binding drift")


def _validated_output_target() -> Path:
    target = _repository_path(DEFAULT_OUTPUT)
    results_root = _repository_path(RESULTS_ROOT)
    if target.resolve().parent != results_root.resolve() or target.suffix != ".json":
        raise ValueError("CCHR source gate output must be a results JSON child")
    if results_root.is_symlink() or target.is_symlink():
        raise ValueError("CCHR source gate output path cannot be a symlink")
    protected = {
        _repository_path(export_prereg.MASTER_PREREGISTRATION).resolve(),
        _repository_path(comparator_freeze.DEFAULT_OUTPUT).resolve(),
        _repository_path(EVALUATOR_SOURCE).resolve(),
    }
    if target.resolve() in protected:
        raise ValueError("CCHR source gate output aliases a protected input")
    return target.resolve()


def run() -> dict[str, Any]:
    target = _validated_output_target()
    if target.exists():
        raise FileExistsError("CCHR source gate artifact is immutable")
    directory_identity = comparator_freeze._directory_identity(target.parent)
    payload = build_report()
    validate_report(payload, verify_files=True)
    comparator_freeze._publish_json_create_only(
        payload,
        target,
        expected_directory_identity=directory_identity,
    )
    published = cchr._read_json(target)
    if published != payload:
        raise RuntimeError("published CCHR source gate artifact differs from payload")
    validate_report(published, verify_files=False)
    return payload


def main() -> None:
    payload = run()
    print(
        json.dumps(
            {
                "path": str(DEFAULT_OUTPUT),
                "manifest_hash": payload["manifest_hash"],
                "decision": payload["decision"],
                "outcomes_opened": payload["outcomes_opened"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
