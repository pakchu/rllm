#!/usr/bin/env python3
"""Select the synthetic-only PSIM-D8 relation-subcard mechanism."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from training import (
    audit_protocol_specification_intent_maturity_d7_gate5_rejection
    as d7_audit,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(
    "results/protocol_specification_intent_maturity_d8_mechanism_probe_"
    "2026-07-27.json"
)
PROTOCOL_VERSION = "psim_d8_relation_subcard_mechanism_probe_v1"
MECHANISM_VERSION = (
    "PSIM_D8_LOGICAL_DAY_CARD_WITH_ORDERED_RELATION_SUBCARDS_V1"
)
MAX_RELATION_UNITS_PER_SUBCARD = 64

D7_TERMINAL_COMMIT = "6d286943566dd13f591aeea41cbc767233822adf"
D7_TERMINAL_PATH = d7_audit.runner.DEFAULT_REJECTION_PATH
D7_TERMINAL_SHA256 = d7_audit.TERMINAL_SHA256
D7_TERMINAL_RESULT_HASH = d7_audit.TERMINAL_RESULT_HASH

D7_FORENSIC_COMMIT = "eb6bbc0ff8b83a38c9bcb56b817e7483fee7661b"
D7_FORENSIC_PATH = d7_audit.DEFAULT_OUTPUT
D7_FORENSIC_SHA256 = (
    "35f961d2bde8a71045209698eee1c5508108218726b73fd2d3ceff35de85ab9b"
)
D7_FORENSIC_RESULT_HASH = (
    "620d81baadafaa9d5cee1e5c38883846d1ac2df60acd00b67117241d87184144"
)
D7_FORENSIC_SCRIPT_PATH = Path(
    "training/"
    "audit_protocol_specification_intent_maturity_d7_gate5_rejection.py"
)
D7_FORENSIC_SCRIPT_SHA256 = (
    "99c54480221eb7f934e3fac59880b3a07e28201c8f4e554eb3bc425b22d1d7bd"
)
D7_FORENSIC_TEST_PATH = Path(
    "tests/"
    "test_audit_protocol_specification_intent_maturity_d7_gate5_rejection.py"
)
D7_FORENSIC_TEST_SHA256 = (
    "9c8dfb7a0eb7adfe3f8273b62c9271b1e23c9dffa6405b61a2c477ec5ad7f864"
)
D7_FORENSIC_DOCUMENT_PATH = Path(
    "docs/psim-d7-gate5-post-terminal-forensic-2026-07-27.md"
)
D7_FORENSIC_DOCUMENT_SHA256 = (
    "7c32f061af8f60e332e043a2df3ae38b39510e09acd94fea010e0ce2e9297740"
)


class D8SubcardError(ValueError):
    """Typed synthetic failure for the D8 subcard mechanism."""


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    return d7_audit._sha256_file(repository_path(path))


def canonical_json_bytes(payload: Any, *, pretty: bool = True) -> bytes:
    return d7_audit.runner.canonical_json_bytes(payload, pretty=pretty)


def canonical_hash(payload: Any) -> str:
    return d7_audit.runner.canonical_hash(payload)


def _read_canonical_json(path: str | Path) -> dict[str, Any]:
    target = repository_path(path)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"PSIM-D8 authority is unsafe: {path}")
    raw = target.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"PSIM-D8 authority is unreadable: {path}"
        ) from error
    if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
        raise RuntimeError(f"PSIM-D8 authority is noncanonical: {path}")
    return payload


def _load_d7_authority() -> dict[str, Any]:
    terminal = _read_canonical_json(D7_TERMINAL_PATH)
    terminal_core = {
        key: value
        for key, value in terminal.items()
        if key != "result_hash"
    }
    forensic = _read_canonical_json(D7_FORENSIC_PATH)
    forensic_core = {
        key: value
        for key, value in forensic.items()
        if key != "result_hash"
    }
    if (
        sha256_file(D7_TERMINAL_PATH) != D7_TERMINAL_SHA256
        or terminal.get("result_hash") != D7_TERMINAL_RESULT_HASH
        or terminal.get("result_hash") != canonical_hash(terminal_core)
        or terminal.get("decision") != "reject"
        or terminal.get("first_failure")
        != {
            "gate_id": 5,
            "name": "split_annual_quarterly_unique_day_support",
        }
        or terminal.get("outcomes_opened") is not False
        or terminal.get("profitability_result") is not False
        or sha256_file(D7_FORENSIC_PATH) != D7_FORENSIC_SHA256
        or forensic.get("result_hash") != D7_FORENSIC_RESULT_HASH
        or forensic.get("result_hash") != canonical_hash(forensic_core)
        or forensic.get("failure") != d7_audit.EXPECTED_EXCEPTION
        or forensic.get("cardinality", {}).get(
            "maximum_model_events_per_card"
        )
        != MAX_RELATION_UNITS_PER_SUBCARD
        or forensic.get("cardinality", {})
        .get("maximum_cardinality", {})
        .get("relation_units")
        != 1_221
        or forensic.get("boundary", {}).get("network_commands") != 0
        or forensic.get("boundary", {}).get(
            "market_model_or_outcomes_accessed"
        )
        is not False
    ):
        raise RuntimeError("PSIM-D7 terminal/forensic authority changed")
    if (
        sha256_file(D7_FORENSIC_SCRIPT_PATH)
        != D7_FORENSIC_SCRIPT_SHA256
        or sha256_file(D7_FORENSIC_TEST_PATH) != D7_FORENSIC_TEST_SHA256
        or sha256_file(D7_FORENSIC_DOCUMENT_PATH)
        != D7_FORENSIC_DOCUMENT_SHA256
    ):
        raise RuntimeError("PSIM-D7 forensic producer changed")
    return {
        "terminal": {
            "commit": D7_TERMINAL_COMMIT,
            "path": D7_TERMINAL_PATH.as_posix(),
            "sha256": D7_TERMINAL_SHA256,
            "result_hash": D7_TERMINAL_RESULT_HASH,
        },
        "forensic": {
            "commit": D7_FORENSIC_COMMIT,
            "path": D7_FORENSIC_PATH.as_posix(),
            "sha256": D7_FORENSIC_SHA256,
            "result_hash": D7_FORENSIC_RESULT_HASH,
            "script": {
                "path": D7_FORENSIC_SCRIPT_PATH.as_posix(),
                "sha256": D7_FORENSIC_SCRIPT_SHA256,
            },
            "test": {
                "path": D7_FORENSIC_TEST_PATH.as_posix(),
                "sha256": D7_FORENSIC_TEST_SHA256,
            },
            "document": {
                "path": D7_FORENSIC_DOCUMENT_PATH.as_posix(),
                "sha256": D7_FORENSIC_DOCUMENT_SHA256,
            },
        },
        "observed_source_cardinality": {
            "overflow_card_cells": forensic["cardinality"][
                "overflow_card_cells"
            ],
            "first_overflow_relation_units": forensic["cardinality"][
                "first_overflow"
            ]["relation_units"],
            "maximum_relation_units": forensic["cardinality"][
                "maximum_cardinality"
            ]["relation_units"],
        },
    }


def build_relation_subcard_manifest(
    relation_units: Sequence[Mapping[str, Any]],
    *,
    schedule: str,
    decision_at: str,
) -> dict[str, Any]:
    units = [dict(unit) for unit in relation_units]
    if not units:
        raise D8SubcardError("ERROR_EMPTY_RELATION_ROSTER")
    if not schedule or not decision_at:
        raise D8SubcardError("ERROR_LOGICAL_DAY_IDENTITY")

    roster_hash = canonical_hash(units)
    subcard_count = math.ceil(
        len(units) / MAX_RELATION_UNITS_PER_SUBCARD
    )
    prior_subcard_hash = canonical_hash(
        {
            "schedule": schedule,
            "decision_at": decision_at,
            "complete_relation_roster_sha256": roster_hash,
            "state": "PSIM_D8_SUBCARD_CHAIN_START",
        }
    )
    subcards: list[dict[str, Any]] = []
    for ordinal in range(subcard_count):
        start = ordinal * MAX_RELATION_UNITS_PER_SUBCARD
        end_exclusive = min(
            start + MAX_RELATION_UNITS_PER_SUBCARD,
            len(units),
        )
        payload_sha256 = canonical_hash(units[start:end_exclusive])
        subcard_core = {
            "schedule": schedule,
            "decision_at": decision_at,
            "subcard_ordinal": ordinal,
            "subcard_count": subcard_count,
            "start": start,
            "end_exclusive": end_exclusive,
            "relation_unit_count": end_exclusive - start,
            "subcard_payload_sha256": payload_sha256,
            "prior_subcard_hash": prior_subcard_hash,
            "complete_relation_roster_sha256": roster_hash,
        }
        subcard = {
            **subcard_core,
            "subcard_hash": canonical_hash(subcard_core),
        }
        subcards.append(subcard)
        prior_subcard_hash = subcard["subcard_hash"]

    manifest_core = {
        "schedule": schedule,
        "decision_at": decision_at,
        "relation_unit_count": len(units),
        "maximum_relation_units_per_subcard": (
            MAX_RELATION_UNITS_PER_SUBCARD
        ),
        "complete_relation_roster_sha256": roster_hash,
        "subcard_count": subcard_count,
        "subcards": subcards,
    }
    return {
        **manifest_core,
        "manifest_hash": canonical_hash(manifest_core),
    }


def validate_relation_subcard_manifest(
    relation_units: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    try:
        expected = build_relation_subcard_manifest(
            relation_units,
            schedule=str(manifest["schedule"]),
            decision_at=str(manifest["decision_at"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise D8SubcardError("ERROR_SUBCARD_MANIFEST_SHAPE") from error
    if dict(manifest) != expected:
        raise D8SubcardError("ERROR_SUBCARD_MANIFEST_MISMATCH")


def build_logical_daily_card_envelope(
    *,
    schedule: str,
    decision_at: str,
    split: str,
    prior_card_hash: str,
    protocol_state: Mapping[str, Any],
    new_events: Sequence[Mapping[str, Any]],
    relation_units: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    units = [dict(unit) for unit in relation_units]
    manifest = build_relation_subcard_manifest(
        units,
        schedule=schedule,
        decision_at=decision_at,
    )
    local_payload = {
        "protocol_state": dict(protocol_state),
        "new_events": [dict(event) for event in new_events],
        "relation_units": units,
        "relation_subcard_manifest": manifest,
    }
    local_payload_sha256 = canonical_hash(local_payload)
    card_core = {
        "schedule": schedule,
        "decision_at": decision_at,
        "prior_card_hash": prior_card_hash,
        "local_payload_sha256": local_payload_sha256,
    }
    return {
        "schedule": schedule,
        "decision_at": decision_at,
        "split": split,
        "local_payload": local_payload,
        "local_payload_sha256": local_payload_sha256,
        "prior_card_hash": prior_card_hash,
        "card_hash": canonical_hash(card_core),
    }


def validate_logical_daily_card_envelope(
    card: Mapping[str, Any],
) -> None:
    try:
        local_payload = card["local_payload"]
        if not isinstance(local_payload, Mapping):
            raise TypeError
        relation_units = local_payload["relation_units"]
        manifest = local_payload["relation_subcard_manifest"]
        if not isinstance(relation_units, Sequence) or not isinstance(
            manifest,
            Mapping,
        ):
            raise TypeError
        if (
            manifest.get("schedule") != card.get("schedule")
            or manifest.get("decision_at") != card.get("decision_at")
        ):
            raise D8SubcardError("ERROR_LOGICAL_CARD_IDENTITY_MISMATCH")
        validate_relation_subcard_manifest(relation_units, manifest)
        expected_local_hash = canonical_hash(local_payload)
        expected_card_hash = canonical_hash(
            {
                "schedule": card["schedule"],
                "decision_at": card["decision_at"],
                "prior_card_hash": card["prior_card_hash"],
                "local_payload_sha256": expected_local_hash,
            }
        )
    except D8SubcardError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise D8SubcardError("ERROR_LOGICAL_CARD_SHAPE") from error
    if (
        card.get("local_payload_sha256") != expected_local_hash
        or card.get("card_hash") != expected_card_hash
    ):
        raise D8SubcardError("ERROR_LOGICAL_CARD_HASH_MISMATCH")


def _synthetic_units(count: int) -> list[dict[str, str]]:
    return [
        {
            "left": canonical_hash({"side": "left", "ordinal": index}),
            "right": canonical_hash({"side": "right", "ordinal": index}),
        }
        for index in range(count)
    ]


def _success_scenario(count: int) -> dict[str, Any]:
    units = _synthetic_units(count)
    manifest = build_relation_subcard_manifest(
        units,
        schedule="ARCHIVE_D90",
        decision_at="2023-01-01T12:05:00Z",
    )
    validate_relation_subcard_manifest(units, manifest)
    expected_count = math.ceil(count / MAX_RELATION_UNITS_PER_SUBCARD)
    passed = (
        manifest["subcard_count"] == expected_count
        and sum(
            row["relation_unit_count"]
            for row in manifest["subcards"]
        )
        == count
        and all(
            1
            <= row["relation_unit_count"]
            <= MAX_RELATION_UNITS_PER_SUBCARD
            for row in manifest["subcards"]
        )
    )
    return {
        "scenario": f"relation_count_{count}",
        "relation_unit_count": count,
        "expected_subcard_count": expected_count,
        "observed_subcard_count": manifest["subcard_count"],
        "manifest_hash": manifest["manifest_hash"],
        "passed": passed,
    }


def _tamper_scenario(field: str) -> dict[str, Any]:
    units = _synthetic_units(70)
    manifest = build_relation_subcard_manifest(
        units,
        schedule="ARCHIVE_D90",
        decision_at="2023-01-01T12:05:00Z",
    )
    tampered = deepcopy(manifest)
    if field == "range":
        tampered["subcards"][1]["start"] += 1
    elif field == "payload":
        tampered["subcards"][0]["subcard_payload_sha256"] = "0" * 64
    elif field == "chain":
        tampered["subcards"][1]["prior_subcard_hash"] = "0" * 64
    elif field == "roster":
        tampered["complete_relation_roster_sha256"] = "0" * 64
    else:
        raise AssertionError(field)
    rejected = False
    try:
        validate_relation_subcard_manifest(units, tampered)
    except D8SubcardError as error:
        rejected = str(error) == "ERROR_SUBCARD_MANIFEST_MISMATCH"
    return {
        "scenario": f"tampered_{field}_rejected",
        "expected": "ERROR_SUBCARD_MANIFEST_MISMATCH",
        "passed": rejected,
    }


def _logical_card_binding_scenario() -> dict[str, Any]:
    units = _synthetic_units(70)
    card = build_logical_daily_card_envelope(
        schedule="ARCHIVE_D90",
        decision_at="2023-01-01T12:05:00Z",
        split="eval",
        prior_card_hash="1" * 64,
        protocol_state={
            "ethereum": "NEW_EVENT",
            "bitcoin": "NO_NEW_EVENT",
        },
        new_events=[{"synthetic": "event"}],
        relation_units=units,
    )
    validate_logical_daily_card_envelope(card)
    tampered = deepcopy(card)
    tampered["local_payload"]["relation_subcard_manifest"]["subcards"][0][
        "end_exclusive"
    ] -= 1
    rejected = False
    try:
        validate_logical_daily_card_envelope(tampered)
    except D8SubcardError:
        rejected = True
    cross_day = deepcopy(card)
    cross_day["decision_at"] = "2023-01-02T12:05:00Z"
    cross_day["card_hash"] = canonical_hash(
        {
            "schedule": cross_day["schedule"],
            "decision_at": cross_day["decision_at"],
            "prior_card_hash": cross_day["prior_card_hash"],
            "local_payload_sha256": cross_day["local_payload_sha256"],
        }
    )
    cross_day_rejected = False
    try:
        validate_logical_daily_card_envelope(cross_day)
    except D8SubcardError as error:
        cross_day_rejected = (
            str(error) == "ERROR_LOGICAL_CARD_IDENTITY_MISMATCH"
        )
    return {
        "scenario": "single_logical_card_binds_complete_subcard_manifest",
        "top_level_daily_card_count": 1,
        "logical_control_denominator_days": 1,
        "subcard_count": card["local_payload"][
            "relation_subcard_manifest"
        ]["subcard_count"],
        "manifest_tamper_rejected": rejected,
        "cross_day_transplant_rejected": cross_day_rejected,
        "passed": (
            rejected
            and cross_day_rejected
            and card["local_payload"]["relation_subcard_manifest"][
                "subcard_count"
            ]
            == 2
        ),
    }


def build_probe() -> dict[str, Any]:
    scenarios = [
        *(
            _success_scenario(count)
            for count in (1, 64, 65, 70, 143, 1_221)
        ),
        *(
            _tamper_scenario(field)
            for field in ("range", "payload", "chain", "roster")
        ),
        _logical_card_binding_scenario(),
    ]
    empty_rejected = False
    try:
        build_relation_subcard_manifest(
            [],
            schedule="ARCHIVE_D90",
            decision_at="2023-01-01T12:05:00Z",
        )
    except D8SubcardError as error:
        empty_rejected = str(error) == "ERROR_EMPTY_RELATION_ROSTER"
    scenarios.append(
        {
            "scenario": "empty_roster_rejected",
            "expected": "ERROR_EMPTY_RELATION_ROSTER",
            "passed": empty_rejected,
        }
    )

    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": "PSIM-D8-SYNTHETIC-RELATION-SUBCARD-PROBE",
        "mechanism_version": MECHANISM_VERSION,
        "synthetic_only": True,
        "selection_scope": (
            "AUTHORIZE_D8_PREREGISTRATION_ONLY_NOT_OFFICIAL_SOURCE_EXECUTION"
        ),
        "candidate": {
            "id": "PSIM-D8",
            "name": "LOGICAL_DAY_CARD_WITH_ORDERED_RELATION_SUBCARDS",
            "last_source_representation_successor": True,
            "d9_allowed_after_source_failure": False,
        },
        "d7_authority": _load_d7_authority(),
        "mechanism_contract": {
            "logical_daily_card_count": (
                "EXACTLY_ONE_PER_SCHEDULE_AND_DECISION_DAY"
            ),
            "logical_daily_relation_roster": (
                "EXACT_D7_ORDERED_COMPLETE_RELATION_UNITS"
            ),
            "maximum_model_relation_units_per_subcard": (
                MAX_RELATION_UNITS_PER_SUBCARD
            ),
            "subcard_partition": (
                "CONTIGUOUS_GREEDY_SLICES_IN_ORIGINAL_ORDER"
            ),
            "subcard_coverage": (
                "COMPLETE_NONOVERLAPPING_NO_GAP_NO_DUPLICATION"
            ),
            "complete_relation_roster_hash_required": True,
            "subcard_payload_hash_required": True,
            "subcard_chain_required": True,
            "logical_card_hash_binds_completed_manifest": True,
            "audit_manifest_model_visible": False,
            "logical_card_payload_model_visible": False,
            "later_model_input": (
                "VERIFIED_SUBCARD_SLICE_ONLY_UNDER_SEPARATE_PREREGISTRATION"
            ),
            "later_model_aggregation": (
                "UNDECIDED_NOT_AUTHORIZED_BY_D8_SOURCE_PREREGISTRATION"
            ),
            "dropping_sampling_summarization_allowed": False,
            "cap_raise_allowed": False,
            "market_or_outcome_dependent_partition_allowed": False,
            "control_denominator": "UNIQUE_LOGICAL_DECISION_DAYS",
        },
        "option_comparison": {
            "deterministic_subcards": "SELECTED_LOSSLESS_BOUNDED",
            "raise_card_cap": "REJECT_POST_INCIDENCE_UNBOUNDED_MODEL_PAYLOAD",
            "daily_summary": "REJECT_LOSSY_SEMANTIC_SELECTION",
            "top_level_multiple_daily_cards": (
                "REJECT_BREAKS_LOGICAL_DAY_IDENTITY_AND_CONTROL_DENOMINATOR"
            ),
        },
        "synthetic_battery": {
            "scenario_count": len(scenarios),
            "all_passed": all(row["passed"] is True for row in scenarios),
            "scenario_roster_hash": canonical_hash(scenarios),
            "scenarios": scenarios,
        },
        "access_boundary": {
            "d7_terminal_artifact_read": True,
            "d7_forensic_artifact_read": True,
            "d7_forensic_source_root_accessed": False,
            "d7_run_invoked": False,
            "external_network_accessed": False,
            "historical_proposal_text_accessed": False,
            "market_data_accessed": False,
            "model_accessed": False,
            "outcomes_accessed": False,
            "reward_trade_pnl_accessed": False,
        },
    }
    return {**core, "result_hash": canonical_hash(core)}


def _safe_output_path(path: str | Path) -> Path:
    requested = Path(path)
    results_root = REPO_ROOT.resolve() / "results"
    target = results_root / requested.name
    if (
        requested.is_absolute()
        or requested.parent != Path("results")
        or requested.suffix != ".json"
        or results_root.is_symlink()
        or not results_root.is_dir()
        or target.is_symlink()
        or (target.exists() and not target.is_file())
    ):
        raise RuntimeError(
            "PSIM-D8 mechanism output must be a safe flat JSON result"
        )
    return target


def write_probe(path: str | Path = DEFAULT_OUTPUT) -> Path:
    target = _safe_output_path(path)
    raw = canonical_json_bytes(build_probe())
    if target.exists() and target.read_bytes() != raw:
        raise RuntimeError("existing PSIM-D8 mechanism artifact differs")
    if not target.exists():
        temporary = target.with_name(target.name + ".tmp")
        if os.path.lexists(temporary):
            raise RuntimeError("unsafe PSIM-D8 mechanism temporary path")
        temporary.write_bytes(raw)
        os.replace(temporary, target)
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    path = write_probe(arguments.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
