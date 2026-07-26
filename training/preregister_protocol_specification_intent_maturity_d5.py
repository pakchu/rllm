"""Preregister outcome-blind PSIM-D5 path/text-delta source semantics."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from training import preregister_protocol_specification_intent_maturity as d1
from training import (
    preregister_protocol_specification_intent_maturity_d4 as d4,
)
from training import (
    probe_protocol_specification_intent_maturity_d5_event_semantics as probe,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(
    "training/preregister_protocol_specification_intent_maturity_d5.py"
)
DEFAULT_OUTPUT = Path(
    "results/protocol_specification_intent_maturity_d5_preregistration_"
    "2026-07-26.json"
)

DECISION_PATH = Path(
    "docs/psim-d5-event-semantics-selection-2026-07-26.md"
)
DECISION_COMMIT = "0e62ec05e6861b2619e6737dd594e7306ad7c93a"
DECISION_SHA256 = (
    "364302ddada267c7252c37cd211f088893597917ab6ea3bbe99f896c647beba1"
)

D4_PREREGISTRATION_PATH = d4.DEFAULT_OUTPUT
D4_PREREGISTRATION_COMMIT = (
    "7731f8322b1700550ff1aa46d8a6c6898c31eef0"
)
D4_PREREGISTRATION_SHA256 = (
    "52d77eafef0e9e79f1d7a47b9c262aad148765a34ac1928b26992cfafce4d515"
)
D4_PREREGISTRATION_MANIFEST_HASH = (
    "b37fe58cf7a043d2164f2e3b08856a75fefad87aef85c02083873e7f3cffb1c8"
)

D4_TERMINAL_PATH = probe.D4_TERMINAL_PATH
D4_TERMINAL_COMMIT = probe.D4_TERMINAL_COMMIT
D4_TERMINAL_SHA256 = probe.D4_TERMINAL_SHA256
D4_TERMINAL_RESULT_HASH = probe.D4_TERMINAL_RESULT_HASH
D4_TERMINAL_ACTION = (
    "REJECT_PSIM_D4_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES"
)

SEMANTICS_PROBE_PATH = probe.DEFAULT_OUTPUT
SEMANTICS_PROBE_COMMIT = DECISION_COMMIT
SEMANTICS_PROBE_SHA256 = (
    "42265a1ed0899366047732e1fa5dad24d961bb4b0bd7fb7bb58479a77bc8894b"
)
SEMANTICS_PROBE_RESULT_HASH = (
    "467f4272bc7276879c0087662a70d99c57d9cef421647f1a679e2fce65de4871"
)
SEMANTICS_PROBE_PROTOCOL_VERSION = probe.PROTOCOL_VERSION
SEMANTICS_VERSION = probe.SEMANTICS_VERSION
MODEL_TEXT_SECTIONS = tuple(probe.d4.core.MODEL_SECTION_ORDER)
SEMANTICS_PROBE_SCRIPT_PATH = Path(
    "training/"
    "probe_protocol_specification_intent_maturity_d5_event_semantics.py"
)
SEMANTICS_PROBE_SCRIPT_SHA256 = (
    "d1aaf55effec3df8f38854992b4c60bd39d612e4bd6cd00fe705f60b5cac9d85"
)
SEMANTICS_PROBE_TEST_PATH = Path(
    "tests/"
    "test_probe_protocol_specification_intent_maturity_d5_event_semantics.py"
)
SEMANTICS_PROBE_TEST_SHA256 = (
    "68919a7e02975557443e7f3e5d38c912e2fbdd74eeaa95e95922fa842cd0b1c1"
)

POLICY_ID = "PSIM-D5"
PROTOCOL_VERSION = "psim_d5_source_preregistration_v1"
SOURCE_ROOT = "/tmp/psim-d5-source"
FAILURE_ACTION = (
    "REJECT_PSIM_D5_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES"
)
MEMORIZATION_FAILURE_ACTION = (
    "REJECT_PSIM_D5_UNCHANGED_BEFORE_MARKET_OR_OUTCOMES"
)
SEALED_REF = "refs/psim-d5/sealed-tip"

ARTIFACT_PATHS = {
    "result": (
        "results/protocol_specification_intent_maturity_d5_source_support_"
        "2026-07-26.json"
    ),
    "rejection": (
        "results/protocol_specification_intent_maturity_d5_source_rejection_"
        "2026-07-26.json"
    ),
    "events": (
        "data/protocol_specification_intent_maturity_d5_events_"
        "2020_2023.jsonl.gz"
    ),
    "cards": (
        "data/protocol_specification_intent_maturity_d5_cards_"
        "2020_2024q1.jsonl.gz"
    ),
    "controls": (
        "results/protocol_specification_intent_maturity_d5_source_controls_"
        "2026-07-26.json"
    ),
}

PROBE_SEMANTICS_CONTRACT = {
    "administrative_quarantine": (
        "EXACT_ONE_LINE_ETHEREUM_ERC_MOVE_STUB_WITH_TARGET_NUMBER_"
        "EQUAL_TO_PROPOSAL_GROUP_PATH_NUMBER"
    ),
    "administrative_quarantine_invalid_metadata_audit": (
        "PRESERVE_EXPLICIT_INVALID_STATES_WHILE_MODEL_TEXT_IS_EMPTY"
    ),
    "administrative_text_model_visible": False,
    "bip_parser": "UNCHANGED_D4_STRICT_PARSER",
    "dependency_when_metadata_invalid": "UNKNOWN_WITH_NULL_COUNT_NO_REPAIR",
    "invalid_metadata_states": [
        "INVALID_DUPLICATE_CONFLICTING",
        "INVALID_DUPLICATE_IDENTICAL",
        "INVALID_MALFORMED_HEADER",
        "INVALID_SELF_DEPENDENCY",
        "INVALID_UNKNOWN",
    ],
    "known_invalid_metadata_text_model_visible": (
        "TRUE_FOR_NONADMINISTRATIVE_EVENTS"
    ),
    "metadata_resolution": (
        "NONE_NO_FIRST_LAST_MERGE_DEDUP_RENAME_OR_SELF_EDGE_DROP"
    ),
    "model_metadata_lines_visible": False,
    "model_text_field": "normalized_text_delta",
    "model_text_sections": list(MODEL_TEXT_SECTIONS),
    "normalized_text_delta_is_causal_semantics_claim": False,
    "normalized_text_delta_order": (
        "SEQUENCE_MATCHER_OPCODE_ORDER_REMOVE_THEN_ADD_SOURCE_ORDER"
    ),
    "path_identity": (
        "PROTOCOL_PLUS_EXACT_OLD_NEW_GROUP_PATHS_PLUS_NUMBER_"
        "CANONICAL_HASH_BOUND"
    ),
    "raw_normalization": "UNCHANGED_D1_NORMALIZE_BLOB_BYTES",
    "reverse_administrative_transition": "FAIL_CLOSED",
    "unknown_grammar": "FAIL_CLOSED_BEFORE_MODEL_OR_OUTCOMES",
}

SEMANTICS_PROBE_BINDING = {
    "path": SEMANTICS_PROBE_PATH.as_posix(),
    "commit": SEMANTICS_PROBE_COMMIT,
    "sha256": SEMANTICS_PROBE_SHA256,
    "result_hash": SEMANTICS_PROBE_RESULT_HASH,
    "protocol_version": SEMANTICS_PROBE_PROTOCOL_VERSION,
    "semantics_version": SEMANTICS_VERSION,
    "script_path": SEMANTICS_PROBE_SCRIPT_PATH.as_posix(),
    "script_sha256": SEMANTICS_PROBE_SCRIPT_SHA256,
    "test_path": SEMANTICS_PROBE_TEST_PATH.as_posix(),
    "test_sha256": SEMANTICS_PROBE_TEST_SHA256,
    "synthetic_only": True,
    "official_historical_proposal_source_opened": False,
    "d4_forensic_root_opened": False,
    "market_model_outcomes_opened": False,
    "official_reference_notes_model_visible": False,
}

EVENT_SEMANTICS_CONTRACT = {
    "semantics_version": SEMANTICS_VERSION,
    "synthetic_probe_binding": copy.deepcopy(SEMANTICS_PROBE_BINDING),
    "probe_semantics": copy.deepcopy(PROBE_SEMANTICS_CONTRACT),
    "historical_ethereum_blob_class_counts": {
        "D4_DUPLICATE_IDENTICAL_HEADER": 7,
        "D4_MALFORMED_HEADER_LINE": 20,
        "D4_SELF_DEPENDENCY": 9,
        "D4_VALID": 4_440,
        "ERC_MIGRATION_REDIRECT_LOWER_PATH": 365,
        "ERC_MIGRATION_REDIRECT_UPPER_PATH": 365,
    },
    "full_normalized_delta_audit": {
        "fields": [
            "audit_diff_hash",
            "audit_line_change_count",
            "exact_old_path",
            "exact_new_path",
            "path_identity_hash",
            "old_metadata_state",
            "new_metadata_state",
            "invalid_metadata_states",
        ],
        "model_visible": False,
        "raw_or_normalized_lines_persisted": False,
    },
    "model_card_integration": {
        "administrative_events_retained_in_source_event_artifact": True,
        "administrative_events_retained_in_model_cards": False,
        "exact_paths_or_path_identity_hash_model_visible": False,
        "known_invalid_state_model_visible": True,
        "normalized_text_delta_sections": list(MODEL_TEXT_SECTIONS),
        "raw_header_other_or_copyright_lines_model_visible": False,
    },
}

BATCH_HYDRATION_CONTRACT = copy.deepcopy(d4.BATCH_HYDRATION_CONTRACT)
BATCH_HYDRATION_CONTRACT["trace_child_argv_ambiguity_action"] = (
    FAILURE_ACTION
)
BATCH_HYDRATION_CONTRACT["post_hydration_read"][
    "missing_object_action"
] = FAILURE_ACTION
BATCH_HYDRATION_CONTRACT["first_failure_action"] = FAILURE_ACTION
BATCH_HYDRATION_CONTRACT["forbidden_transports"] = [
    (
        "D1, D2, D3, or D4 source-object reuse"
        if value == "D1, D2, or D3 source-object reuse"
        else value
    )
    for value in BATCH_HYDRATION_CONTRACT["forbidden_transports"]
]

# Frozen after the recursive D4 -> D5 delta is reviewed.
AUTHORIZED_DELTA_PATHS = (
    "candidate.id",
    "candidate.name",
    "candidate.selection_commit",
    "daily_relation_contract.administrative_events_retained_in_model_cards",
    (
        "daily_relation_contract."
        "administrative_events_retained_in_source_event_artifact"
    ),
    "daily_relation_contract.model_text_field",
    "decision_binding.commit",
    "decision_binding.path",
    "decision_binding.sha256",
    "event_contract.d5_source_semantics",
    "memorization_contract.administrative_events_excluded_from_challenges",
    "memorization_contract.first_failure_action",
    "memorization_contract.model_text_field",
    "next_authorized_step",
    "parser_contract.metadata_parse_failure_action",
    "protocol_version",
    "representation_contract.administrative_events_model_visible",
    "representation_contract.dependency_delta_states",
    "representation_contract.deterministic_source_tokens",
    (
        "representation_contract."
        "exact_paths_or_path_identity_hash_model_visible"
    ),
    "representation_contract.known_invalid_metadata_state_model_visible",
    "representation_contract.legacy_intent_text_field_allowed",
    "representation_contract.model_text_field",
    "representation_contract.model_text_sections",
    (
        "representation_contract."
        "raw_header_other_or_copyright_text_model_visible"
    ),
    "source_contract.artifact_paths.cards",
    "source_contract.artifact_paths.controls",
    "source_contract.artifact_paths.events",
    "source_contract.artifact_paths.rejection",
    "source_contract.artifact_paths.result",
    "source_contract.bare_repository_contract.ref_roster[1]",
    "source_contract.bare_repository_contract.sealed_ref",
    "source_contract.bare_repository_contract.source_traversal_ref",
    "source_contract.batch_hydration_contract.first_failure_action",
    "source_contract.batch_hydration_contract.forbidden_transports[7]",
    (
        "source_contract.batch_hydration_contract.post_hydration_read."
        "missing_object_action"
    ),
    (
        "source_contract.batch_hydration_contract."
        "trace_child_argv_ambiguity_action"
    ),
    "source_contract.repositories[0].sealed_ref",
    "source_contract.repositories[1].sealed_ref",
    "source_contract.source_root",
    "source_support_contract.administrative_events_retained_in_model_cards",
    (
        "source_support_contract."
        "administrative_events_retained_in_source_artifact"
    ),
    "source_support_contract.blob_semantics_total_fraction_required",
    (
        "source_support_contract.control_sensitivity_metric."
        "administrative_events_in_model_payload"
    ),
    (
        "source_support_contract.control_sensitivity_metric."
        "first_failure_action"
    ),
    "source_support_contract.ethereum_historical_blob_class_counts",
    "source_support_contract.first_failure_action",
    "source_support_contract.gate_four_semantics",
    "source_support_contract.parser_success_fraction_required",
    (
        "source_support_contract.relation_control_transforms."
        "dependency_edge_direction_reverse"
    ),
    (
        "source_support_contract.relation_control_transforms."
        "old_new_direction_reverse"
    ),
    (
        "source_support_contract.relation_control_transforms."
        "protocol_label_swap"
    ),
)
AUTHORIZED_DELTA_HASH = (
    "81ec7b54b199801bf5f68f78c03de1c96583eb6c2a061124a2367966457c190d"
)
EVENT_SEMANTICS_CONTRACT_HASH = (
    "d73f97f980009b199d918bce662876f29e3559ee618a38679ec5209aa8404dcf"
)
BATCH_HYDRATION_CONTRACT_HASH = (
    "e2cff3df57a398ba65072b4243077c68f4ba71e44e5c11f182c7d884c4721381"
)


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(repository_path(path).read_bytes())


def canonical_json_bytes(payload: Any, *, pretty: bool = True) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_hash(payload: Any) -> str:
    return sha256_bytes(
        canonical_json_bytes(payload, pretty=False).rstrip(b"\n")
    )


def _read_canonical_json(path: str | Path) -> dict[str, Any]:
    target = repository_path(path)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"PSIM-D5 authority is absent or unsafe: {path}")
    raw = target.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"PSIM-D5 authority is unreadable: {path}") from error
    if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
        raise RuntimeError(f"PSIM-D5 authority is noncanonical: {path}")
    return payload


def _validate_authority() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    if (
        re.fullmatch(r"[0-9a-f]{40}", DECISION_COMMIT) is None
        or sha256_file(DECISION_PATH) != DECISION_SHA256
    ):
        raise RuntimeError("PSIM-D5 decision authority changed")

    d4_registration = _read_canonical_json(D4_PREREGISTRATION_PATH)
    if (
        sha256_file(D4_PREREGISTRATION_PATH)
        != D4_PREREGISTRATION_SHA256
        or d4_registration.get("manifest_hash")
        != D4_PREREGISTRATION_MANIFEST_HASH
        or d4_registration != d4.build_preregistration()
    ):
        raise RuntimeError("PSIM-D4 preregistration authority changed")

    d4_terminal = _read_canonical_json(D4_TERMINAL_PATH)
    ledger = d4_terminal.get("access_ledger", {})
    gates = d4_terminal.get("gates")
    if (
        sha256_file(D4_TERMINAL_PATH) != D4_TERMINAL_SHA256
        or d4_terminal.get("result_hash") != D4_TERMINAL_RESULT_HASH
        or d4_terminal.get("decision") != "reject"
        or d4_terminal.get("first_failure")
        != {
            "gate_id": 4,
            "name": "historical_blob_preamble_dependency_integrity",
        }
        or d4_terminal.get("terminal_action") != D4_TERMINAL_ACTION
        or d4_terminal.get("outcomes_opened") is not False
        or d4_terminal.get("profitability_result") is not False
        or not isinstance(ledger, dict)
        or any(
            ledger.get(name) != 0
            for name in probe.d4.FORBIDDEN_ACCESS_FIELDS
        )
        or ledger.get("proposal_blobs_opened") != 5_206
        or ledger.get("proposal_text_rows_opened") != 44
        or not isinstance(gates, list)
        or len(gates) != 4
        or [row.get("passed") for row in gates]
        != [True, True, True, False]
    ):
        raise RuntimeError("PSIM-D4 terminal authority changed")

    semantics = _read_canonical_json(SEMANTICS_PROBE_PATH)
    if (
        sha256_file(SEMANTICS_PROBE_PATH) != SEMANTICS_PROBE_SHA256
        or sha256_file(SEMANTICS_PROBE_SCRIPT_PATH)
        != SEMANTICS_PROBE_SCRIPT_SHA256
        or sha256_file(SEMANTICS_PROBE_TEST_PATH)
        != SEMANTICS_PROBE_TEST_SHA256
        or semantics.get("result_hash") != SEMANTICS_PROBE_RESULT_HASH
        or semantics.get("protocol_version")
        != SEMANTICS_PROBE_PROTOCOL_VERSION
        or semantics.get("semantics_version") != SEMANTICS_VERSION
        or semantics.get("semantics_contract")
        != PROBE_SEMANTICS_CONTRACT
        or semantics.get("synthetic_only") is not True
        or semantics.get("selection_scope")
        != "AUTHORIZE_D5_PREREGISTRATION_ONLY_NOT_OFFICIAL_SOURCE_EXECUTION"
        or semantics.get("access_boundary")
        != {
            "d4_census_artifact_read": True,
            "d4_forensic_root_accessed": False,
            "d4_terminal_artifact_read": True,
            "external_network_accessed_by_probe": False,
            "market_data_accessed": False,
            "model_accessed": False,
            "official_reference_research_preexisted_probe": True,
            "official_historical_proposal_source_accessed": False,
            "outcomes_accessed": False,
        }
        or semantics.get("official_reference_provenance", {}).get(
            "model_visible"
        )
        is not False
        or semantics != probe.build_probe()
    ):
        raise RuntimeError("PSIM-D5 semantics probe authority changed")
    return d4_registration, d4_terminal, semantics


def _diff_values(
    left: Any,
    right: Any,
    *,
    path: str = "",
) -> dict[str, dict[str, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        rows: dict[str, dict[str, Any]] = {}
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else key
            if key not in left:
                rows[child] = {"before": None, "after": right[key]}
            elif key not in right:
                rows[child] = {"before": left[key], "after": None}
            else:
                rows.update(_diff_values(left[key], right[key], path=child))
        return rows
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return {path: {"before": left, "after": right}}
        rows: dict[str, dict[str, Any]] = {}
        for index, (left_value, right_value) in enumerate(zip(left, right)):
            rows.update(
                _diff_values(
                    left_value,
                    right_value,
                    path=f"{path}[{index}]",
                )
            )
        return rows
    return {} if left == right else {path: {"before": left, "after": right}}


def _contract_core(registration: dict[str, Any]) -> dict[str, Any]:
    core = copy.deepcopy(registration)
    core.pop("manifest_hash")
    core.pop("inheritance_proof")
    return core


def _successor_core(d4_registration: dict[str, Any]) -> dict[str, Any]:
    core = _contract_core(d4_registration)
    core["protocol_version"] = PROTOCOL_VERSION
    core["candidate"] = {
        **core["candidate"],
        "id": POLICY_ID,
        "name": (
            "Protocol Specification Intent-Maturity relation RLLM, "
            "exact path identity plus normalized text-delta semantics"
        ),
        "selection_commit": DECISION_COMMIT,
    }
    core["decision_binding"] = {
        "path": DECISION_PATH.as_posix(),
        "commit": DECISION_COMMIT,
        "sha256": DECISION_SHA256,
    }

    parser = core["parser_contract"]
    parser["metadata_parse_failure_action"] = (
        "CLASSIFY_D5_KNOWN_INVALID_WITHOUT_REPAIR_OR_REJECT_UNKNOWN"
    )

    event = core["event_contract"]
    event["d5_source_semantics"] = copy.deepcopy(
        EVENT_SEMANTICS_CONTRACT
    )

    representation = core["representation_contract"]
    representation["dependency_delta_states"] = [
        *representation["dependency_delta_states"],
        "UNKNOWN_INVALID_METADATA",
    ]
    representation["deterministic_source_tokens"] = [
        *representation["deterministic_source_tokens"],
        "old_metadata_state",
        "new_metadata_state",
        "invalid_metadata_present",
    ]
    representation["administrative_events_model_visible"] = False
    representation["exact_paths_or_path_identity_hash_model_visible"] = False
    representation["known_invalid_metadata_state_model_visible"] = True
    representation["legacy_intent_text_field_allowed"] = False
    representation["model_text_field"] = "normalized_text_delta"
    representation["model_text_sections"] = list(
        MODEL_TEXT_SECTIONS
    )
    representation["raw_header_other_or_copyright_text_model_visible"] = (
        False
    )

    daily = core["daily_relation_contract"]
    daily["administrative_events_retained_in_model_cards"] = False
    daily["administrative_events_retained_in_source_event_artifact"] = True
    daily["model_text_field"] = "normalized_text_delta"

    source = core["source_contract"]
    source["source_root"] = SOURCE_ROOT
    source["artifact_paths"] = dict(ARTIFACT_PATHS)
    source["batch_hydration_contract"] = copy.deepcopy(
        BATCH_HYDRATION_CONTRACT
    )
    bare = source["bare_repository_contract"]
    bare["sealed_ref"] = SEALED_REF
    bare["ref_roster"] = ["refs/heads/master", SEALED_REF]
    bare["source_traversal_ref"] = SEALED_REF
    for repository in source["repositories"]:
        repository["sealed_ref"] = SEALED_REF

    core["memorization_contract"]["first_failure_action"] = (
        MEMORIZATION_FAILURE_ACTION
    )
    core["memorization_contract"][
        "administrative_events_excluded_from_challenges"
    ] = True
    core["memorization_contract"]["model_text_field"] = (
        "normalized_text_delta"
    )

    support = core["source_support_contract"]
    support["first_failure_action"] = FAILURE_ACTION
    support["control_sensitivity_metric"]["first_failure_action"] = (
        FAILURE_ACTION
    )
    support["control_sensitivity_metric"][
        "administrative_events_in_model_payload"
    ] = False
    support.pop("parser_success_fraction_required")
    support["blob_semantics_total_fraction_required"] = "1.0"
    support["ethereum_historical_blob_class_counts"] = copy.deepcopy(
        EVENT_SEMANTICS_CONTRACT[
            "historical_ethereum_blob_class_counts"
        ]
    )
    support["gate_four_semantics"] = (
        "STRICT_D4_VALID_OR_EXACT_ADMINISTRATIVE_REDIRECT_OR_"
        "KNOWN_INVALID_METADATA_STATE_OTHERWISE_REJECT"
    )
    support["administrative_events_retained_in_source_artifact"] = True
    support["administrative_events_retained_in_model_cards"] = False
    support["relation_control_transforms"][
        "old_new_direction_reverse"
    ] = (
        "swap all old/new derived fields and exact side paths; map "
        "CREATE<->DELETE and retain UPDATE; swap explicit metadata states; "
        "reverse normalized_text_delta ADD/REMOVE; retain administrative "
        "quarantine as model-hidden"
    )
    support["relation_control_transforms"][
        "dependency_edge_direction_reverse"
    ] = (
        "swap ADDED<->REMOVED; keep NO_PRIOR,STABLE,MIXED,DELETED,"
        "UNKNOWN_INVALID_METADATA fixed"
    )
    support["relation_control_transforms"]["protocol_label_swap"] = (
        "swap ethereum<->bitcoin labels after extraction; preserve raw event "
        "identity, exact side-path audit hash, metadata state, and normalized "
        "text delta; rebuild relation cards; administrative quarantine "
        "remains model-hidden"
    )

    core["next_authorized_step"] = (
        "implement and seal synthetic-only PSIM-D5 path/text-delta "
        "source-support evaluator"
    )
    return core


def _transport_contract_rebased_to_d4(
    contract: dict[str, Any],
) -> dict[str, Any]:
    rebased = copy.deepcopy(contract)
    rebased["trace_child_argv_ambiguity_action"] = d4.FAILURE_ACTION
    rebased["post_hydration_read"][
        "missing_object_action"
    ] = d4.FAILURE_ACTION
    rebased["first_failure_action"] = d4.FAILURE_ACTION
    rebased["forbidden_transports"] = [
        (
            "D1, D2, or D3 source-object reuse"
            if value == "D1, D2, D3, or D4 source-object reuse"
            else value
        )
        for value in rebased["forbidden_transports"]
    ]
    return rebased


def build_preregistration() -> dict[str, Any]:
    d4_registration, d4_terminal, semantics = _validate_authority()
    d4_core = _contract_core(d4_registration)
    successor = _successor_core(d4_registration)
    delta = _diff_values(d4_core, successor)
    if tuple(sorted(delta)) != tuple(sorted(AUTHORIZED_DELTA_PATHS)):
        raise RuntimeError(
            "PSIM-D5 inherited-contract delta changed: "
            + ",".join(sorted(delta))
        )
    if (
        canonical_hash(delta) != AUTHORIZED_DELTA_HASH
        or canonical_hash(EVENT_SEMANTICS_CONTRACT)
        != EVENT_SEMANTICS_CONTRACT_HASH
        or canonical_hash(BATCH_HYDRATION_CONTRACT)
        != BATCH_HYDRATION_CONTRACT_HASH
    ):
        raise RuntimeError("PSIM-D5 authorized semantics delta hash changed")
    if _transport_contract_rebased_to_d4(
        BATCH_HYDRATION_CONTRACT
    ) != d4.BATCH_HYDRATION_CONTRACT:
        raise RuntimeError("PSIM-D5 changed D4 hydration mechanics")
    if successor["source_support_contract"]["gates_in_order"] != list(
        d1.SOURCE_ONLY_GATES
    ):
        raise RuntimeError("PSIM-D5 source gate roster changed")
    if successor["source_support_contract"]["relation_controls"] != list(
        d1.RELATION_CONTROLS
    ):
        raise RuntimeError("PSIM-D5 control roster changed")
    for key in (
        "availability_contract",
        "boundary_reset_contract",
        "bucket_contract",
        "excluded_feasibility_probe",
        "forbidden_access_contract",
        "official_sources",
        "split_contract",
    ):
        if successor[key] != d4_core[key]:
            raise RuntimeError(f"PSIM-D5 inherited {key} changed")

    inheritance = {
        "d4_preregistration": {
            "path": D4_PREREGISTRATION_PATH.as_posix(),
            "commit": D4_PREREGISTRATION_COMMIT,
            "sha256": D4_PREREGISTRATION_SHA256,
            "manifest_hash": D4_PREREGISTRATION_MANIFEST_HASH,
            "contract_core_hash": canonical_hash(d4_core),
        },
        "d4_terminal_rejection": {
            "path": D4_TERMINAL_PATH.as_posix(),
            "commit": D4_TERMINAL_COMMIT,
            "sha256": D4_TERMINAL_SHA256,
            "result_hash": D4_TERMINAL_RESULT_HASH,
            "first_failure_gate_id": 4,
            "proposal_blobs_opened": d4_terminal["access_ledger"][
                "proposal_blobs_opened"
            ],
            "proposal_text_rows_opened": d4_terminal["access_ledger"][
                "proposal_text_rows_opened"
            ],
            "outcomes_opened": d4_terminal["outcomes_opened"],
        },
        "d5_event_semantics_probe": {
            **copy.deepcopy(SEMANTICS_PROBE_BINDING),
            "selection_scope": semantics["selection_scope"],
        },
        "authorized_delta_paths": list(AUTHORIZED_DELTA_PATHS),
        "authorized_delta": delta,
        "authorized_delta_hash": AUTHORIZED_DELTA_HASH,
        "event_semantics_contract_hash": EVENT_SEMANTICS_CONTRACT_HASH,
        "batch_hydration_contract_hash": BATCH_HYDRATION_CONTRACT_HASH,
        "d4_transport_mechanics_byte_equal_after_namespace_rebase": True,
        "all_other_contract_paths_byte_equal": True,
        "official_selection_evidence": [
            "https://github.com/ethereum/EIPs/commit/"
            "0f44e2b94df4e504bb7b912f56ebd712db2ad396",
            "https://github.com/ethereum/EIPs/commit/"
            "47ce70257fae525a427780630bd8d1903cc96e75",
            "https://eips.ethereum.org/EIPS/eip-1",
            "https://yaml.org/spec/1.2.2/",
        ],
        "preregistration_access": {
            "git_commands": 0,
            "network_calls": 0,
            "d4_forensic_root_opened": False,
            "official_historical_proposal_source_opened": False,
            "market_model_outcomes_opened": False,
            "official_reference_notes_model_visible": False,
        },
    }
    core = {
        **successor,
        "inheritance_proof": inheritance,
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def write_preregistration(path: str | Path = DEFAULT_OUTPUT) -> Path:
    destination = repository_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(build_preregistration())
    if os.path.lexists(destination):
        if (
            destination.is_symlink()
            or not destination.is_file()
            or destination.read_bytes() != raw
        ):
            raise RuntimeError(
                f"existing PSIM-D5 preregistration differs: {destination}"
            )
        return destination
    temporary = destination.with_name(destination.name + ".tmp")
    if os.path.lexists(temporary):
        raise RuntimeError(
            f"unsafe PSIM-D5 preregistration temporary: {temporary}"
        )
    temporary.write_bytes(raw)
    os.replace(temporary, destination)
    return destination


def main() -> None:
    path = write_preregistration()
    payload = _read_canonical_json(path)
    print(
        json.dumps(
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "manifest_hash": payload["manifest_hash"],
                "authorized_delta_hash": payload["inheritance_proof"][
                    "authorized_delta_hash"
                ],
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
