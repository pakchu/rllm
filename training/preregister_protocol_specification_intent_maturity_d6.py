"""Preregister the outcome-blind PSIM-D6 source-representation successor.

This module reads only committed, canonical authority artifacts.  It does not
open the PSIM-D5 forensic root, execute a source runner, access a market or a
model, or inspect any outcome.  The preregistration permits a later reviewed
implementation and seal; it does not authorize an official PSIM-D6 run.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from training import (
    preregister_protocol_specification_intent_maturity_d5 as d5,
)
from training import (
    probe_protocol_specification_intent_maturity_d6_mechanism as mechanism,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(
    "training/preregister_protocol_specification_intent_maturity_d6.py"
)
DEFAULT_OUTPUT = Path(
    "results/protocol_specification_intent_maturity_d6_preregistration_"
    "2026-07-26.json"
)

DECISION_PATH = Path("docs/psim-d6-mechanism-selection-2026-07-26.md")
DECISION_COMMIT = "f985acb9821913e10325ed9487bdcea8fc2d39d9"
DECISION_SHA256 = (
    "d47448291b529442cc5dc18e0ccd86147a2c4cc837d63d025b91d39eca308959"
)

D5_PREREGISTRATION_PATH = d5.DEFAULT_OUTPUT
D5_PREREGISTRATION_COMMIT = (
    "4e2b403c1f369bf2e76b5edeb1e4166b9d2f8779"
)
D5_PREREGISTRATION_SHA256 = (
    "11465540d59181bc48ea28c5164579847cbd936bf005c69d874ec2c873c949b9"
)
D5_PREREGISTRATION_MANIFEST_HASH = (
    "f08eeb300fceb906cdcde485b4bce184c48d4cb14a1cd9028046e0c21a287309"
)
D5_PREREGISTRATION_SCRIPT_PATH = Path(
    "training/preregister_protocol_specification_intent_maturity_d5.py"
)
D5_PREREGISTRATION_SCRIPT_SHA256 = (
    "cc47bd574db47ee8857fc07cd9ff9a168b996e1d9a07ae0251d2ae85c1fdc7c6"
)
D5_PREREGISTRATION_TEST_PATH = Path(
    "tests/test_preregister_protocol_specification_intent_maturity_d5.py"
)
D5_PREREGISTRATION_TEST_SHA256 = (
    "e6e8b014dec1e3c9634218a026353abe7c35da53b04ee662de2f6f6145cab186"
)
D5_PREREGISTRATION_DOCUMENT_PATH = Path(
    "docs/psim-d5-source-support-preregistration-2026-07-26.md"
)
D5_PREREGISTRATION_DOCUMENT_SHA256 = (
    "ec949ba8aa47a3a67ea1849f80b5d6fb04f4b1dac24e1d0b15a87243d819e23a"
)

D5_TERMINAL_PATH = Path(
    "results/protocol_specification_intent_maturity_d5_source_rejection_"
    "2026-07-26.json"
)
D5_TERMINAL_COMMIT = "0f69f7472d89474052186bbb2b13fa8d6bf5d77f"
D5_TERMINAL_SHA256 = (
    "ffdebf2e5107f08345f16e21adc895d3bfc2f236d6b231322d03c372d4764ca1"
)
D5_TERMINAL_RESULT_HASH = (
    "0a23218e8784599f09e092d4f93942a48111c0af4f8e3ff85e2183eb84f56c56"
)
D5_TERMINAL_ACTION = (
    "REJECT_PSIM_D5_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES"
)
D5_TERMINAL_ZERO_LEDGER_FIELDS = (
    "btc_market_rows_read",
    "cagr_values_built",
    "daily_cards_built",
    "funding_rows_read",
    "future_return_rows_read",
    "model_outputs_built",
    "models_loaded",
    "pnl_rows_built",
    "post_2023_proposal_blobs_opened",
    "pre_2020_proposal_blobs_opened",
    "reward_rows_built",
    "strict_mdd_values_built",
    "trade_rows_built",
)

MECHANISM_PROBE_PATH = mechanism.DEFAULT_OUTPUT
MECHANISM_PROBE_COMMIT = DECISION_COMMIT
MECHANISM_PROBE_SHA256 = (
    "01b09218d71d83c6abc3c4225b708a1cae6fe9e426b9bbd98f4fe6e86579d60b"
)
MECHANISM_PROBE_RESULT_HASH = (
    "dda4b4786b34064a104178580f6cd33e56d5616c282515f6579105231b5dab38"
)
MECHANISM_PROBE_PROTOCOL_VERSION = (
    "psim_d6_source_mechanism_probe_v1"
)
MECHANISM_VERSION = (
    "PSIM_EXACT_MIGRATION_RECEIPT_PLUS_UTF8_CHUNK_TRANSPORT_V1"
)
MECHANISM_PROBE_SCRIPT_PATH = Path(
    "training/probe_protocol_specification_intent_maturity_d6_mechanism.py"
)
MECHANISM_PROBE_SCRIPT_SHA256 = (
    "fbe73520217e7e3119ee29cfff30c1bf7355678369a95fd5fde659eb3db68d91"
)
MECHANISM_PROBE_TEST_PATH = Path(
    "tests/test_probe_protocol_specification_intent_maturity_d6_mechanism.py"
)
MECHANISM_PROBE_TEST_SHA256 = (
    "3aeaaacf3f97d87771e7af7684b08915f5f1e501adab779436c6835b27d534b6"
)
MECHANISM_PROBE_DOCUMENT_PATH = DECISION_PATH
MECHANISM_PROBE_DOCUMENT_SHA256 = DECISION_SHA256

D5_CENSUS_COMMIT = mechanism.D5_CENSUS_COMMIT
D5_CENSUS_PATH = mechanism.D5_CENSUS_PATH
D5_CENSUS_SHA256 = mechanism.D5_CENSUS_SHA256
D5_CENSUS_RESULT_HASH = mechanism.D5_CENSUS_RESULT_HASH
D5_EPISODE_ROSTER_HASH = mechanism.D5_EPISODE_ROSTER_HASH
D5_EPISODE_RECEIPT_MANIFEST_HASH = (
    mechanism.D5_EPISODE_RECEIPT_MANIFEST_HASH
)
D5_MIGRATION_PROPOSAL_ROSTER_HASH = (
    mechanism.D5_MIGRATION_PROPOSAL_ROSTER_HASH
)
D5_TEXT_BOUND_EVENT_ROSTER_HASH = (
    mechanism.D5_TEXT_BOUND_EVENT_ROSTER_HASH
)

POLICY_ID = "PSIM-D6"
PROTOCOL_VERSION = "psim_d6_source_preregistration_v1"
SOURCE_ROOT = "/tmp/psim-d6-source"
FAILURE_ACTION = (
    "REJECT_PSIM_D6_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES"
)
MEMORIZATION_FAILURE_ACTION = (
    "REJECT_PSIM_D6_UNCHANGED_BEFORE_MARKET_OR_OUTCOMES"
)
SEALED_REF = "refs/psim-d6/sealed-tip"

ARTIFACT_PATHS = {
    "result": (
        "results/protocol_specification_intent_maturity_d6_source_support_"
        "2026-07-26.json"
    ),
    "rejection": (
        "results/protocol_specification_intent_maturity_d6_source_rejection_"
        "2026-07-26.json"
    ),
    "events": (
        "data/protocol_specification_intent_maturity_d6_events_"
        "2020_2023.jsonl.gz"
    ),
    "cards": (
        "data/protocol_specification_intent_maturity_d6_cards_"
        "2020_2024q1.jsonl.gz"
    ),
    "controls": (
        "results/protocol_specification_intent_maturity_d6_source_controls_"
        "2026-07-26.json"
    ),
}

MECHANISM_PROBE_CONTRACT = {
    "administrative_restoration": (
        "EXACT_THREE_STEP_CAUSAL_EPISODE_PLUS_PER_PROPOSAL_"
        "FROZEN_RECEIPT_HASH"
    ),
    "administrative_restoration_model_text_visible": False,
    "chunk_full_text_serialization": (
        "D5_CAUSAL_MODEL_ROWS_SECTION_DIRECTION_LINE_JOINED_BY_LF"
    ),
    "chunk_split": (
        "GREEDY_CONTIGUOUS_UTF8_BYTES_BACKTRACK_CONTINUATION_BOUNDARY"
    ),
    "chunk_transport_fields": [
        "chunk_count",
        "chunk_index",
        "normalized_text_delta_chunk",
    ],
    "chunk_transport_order": "ZERO_BASED_ASCENDING_CHUNK_INDEX",
    "full_text_reconstruction": "BYTE_FOR_BYTE_REQUIRED",
    "max_bytes_per_chunk": 8_192,
    "max_chunks_per_event": 8,
    "ninth_chunk": "FAIL_CLOSED_NO_TRUNCATION_OR_SUMMARIZATION",
    "receipt_authority_count": 365,
    "restoration_requires_prior_events_not_future_events": True,
    "unknown_episode": "FAIL_CLOSED_BEFORE_MODEL_OR_OUTCOMES",
}

MECHANISM_PROBE_BINDING = {
    "path": MECHANISM_PROBE_PATH.as_posix(),
    "commit": MECHANISM_PROBE_COMMIT,
    "sha256": MECHANISM_PROBE_SHA256,
    "result_hash": MECHANISM_PROBE_RESULT_HASH,
    "protocol_version": MECHANISM_PROBE_PROTOCOL_VERSION,
    "mechanism_version": MECHANISM_VERSION,
    "script_path": MECHANISM_PROBE_SCRIPT_PATH.as_posix(),
    "script_sha256": MECHANISM_PROBE_SCRIPT_SHA256,
    "test_path": MECHANISM_PROBE_TEST_PATH.as_posix(),
    "test_sha256": MECHANISM_PROBE_TEST_SHA256,
    "document_path": MECHANISM_PROBE_DOCUMENT_PATH.as_posix(),
    "document_sha256": MECHANISM_PROBE_DOCUMENT_SHA256,
    "synthetic_only": True,
    "selection_scope": (
        "AUTHORIZE_D6_PREREGISTRATION_ONLY_NOT_OFFICIAL_SOURCE_EXECUTION"
    ),
    "d5_forensic_root_accessed": False,
    "market_model_outcomes_accessed": False,
}

MODEL_TEXT_TRANSPORT_CONTRACT = {
    "audit_receipt_fields": [
        "chunk_count",
        "chunks",
        "full_text_line_count",
        "full_text_sha256",
        "full_text_utf8_bytes",
        "max_bytes_per_chunk",
        "max_chunks_per_event",
        "protocol_version",
        "receipt_hash",
        "reconstructed_sha256",
        "reconstruction_matches",
    ],
    "audit_receipt_model_visible": False,
    "canonical_partition_validation_required": True,
    "chunk_payload_fields": list(
        MECHANISM_PROBE_CONTRACT["chunk_transport_fields"]
    ),
    "chunk_payload_order": "ZERO_BASED_ASCENDING_CHUNK_INDEX",
    "chunk_payloads_are_transport_fragments_not_events_or_labels": True,
    "empty_text_chunk_count": 0,
    "event_container_field": "normalized_text_delta_chunks",
    "full_text_reconstruction": "BYTE_FOR_BYTE_REQUIRED",
    "full_text_serialization": MECHANISM_PROBE_CONTRACT[
        "chunk_full_text_serialization"
    ],
    "line_serialization_semantics": (
        "SECTION_AND_DIRECTION_ARE_THE_FIRST_TWO_DELIMITED_FIELDS;_"
        "LINE_IS_OPAQUE_AFTER_THE_SECOND_PIPE_AND_IS_NOT_REPARSED"
    ),
    "max_bytes_per_chunk": mechanism.MAX_MODEL_TEXT_BYTES_PER_CHUNK,
    "max_bytes_per_event": mechanism.MAX_MODEL_TEXT_BYTES_PER_EVENT,
    "max_chunks_per_event": mechanism.MAX_MODEL_TEXT_CHUNKS_PER_EVENT,
    "model_aggregation_policy": (
        "UNDECIDED_NOT_AUTHORIZED_BY_D6_PREREGISTRATION"
    ),
    "ninth_chunk_action": (
        "RECORD_TYPED_EVENT_ERROR_CONTINUE_COMPLETE_ROSTER_THEN_REJECT_"
        "NO_TRUNCATION_OR_SUMMARIZATION"
    ),
    "split": MECHANISM_PROBE_CONTRACT["chunk_split"],
    "strict_utf8_required": True,
}

MIGRATION_RESTORATION_CONTRACT = {
    "administrative_events_model_visible": False,
    "authority": {
        "episode_count": 365,
        "episode_receipt_manifest_hash": (
            D5_EPISODE_RECEIPT_MANIFEST_HASH
        ),
        "episode_roster_hash": D5_EPISODE_ROSTER_HASH,
        "proposal_roster_hash": D5_MIGRATION_PROPOSAL_ROSTER_HASH,
        "source_census_commit": D5_CENSUS_COMMIT,
        "source_census_path": D5_CENSUS_PATH.as_posix(),
        "source_census_result_hash": D5_CENSUS_RESULT_HASH,
        "source_census_sha256": D5_CENSUS_SHA256,
    },
    "authority_receipt_hashes_model_visible": False,
    "authorized_sequence": {
        "blob_classes": [
            list(row) for row in mechanism.MIGRATION_CLASS_SEQUENCE
        ],
        "commit_oids": list(mechanism.MIGRATION_COMMIT_SEQUENCE),
        "effective_days": list(mechanism.MIGRATION_DAY_SEQUENCE),
        "historical_d5_outcome_ids": list(mechanism.D5_OUTCOME_SEQUENCE),
    },
    "exact_path_and_blob_continuity_required": True,
    "generic_administrative_to_valid_transition_authorized": False,
    "model_payload_for_restoration": {
        "administrative_quarantined": True,
        "model_visibility": "ADMINISTRATIVE_QUARANTINE",
        "normalized_text_delta_chunks": [],
    },
    "restoration_authorization": (
        "EXACT_FROZEN_PROPOSAL_RECEIPT_PLUS_EXACT_THREE_STEP_CAUSAL_"
        "EPISODE_ONLY"
    ),
    "unknown_or_mutated_episode_action": (
        "RECORD_TYPED_EVENT_ERROR_CONTINUE_COMPLETE_ROSTER_THEN_REJECT_"
        "BEFORE_MODEL_OR_OUTCOMES"
    ),
}

GATE_FOUR_TOTALITY_CONTRACT = {
    "canonical_rejection_required_before_return_or_raise": True,
    "complete_roster_scope": (
        "ALL_RETAINED_2020_2023_PROPOSAL_GROUP_EVENTS_IN_ALL_FOUR_"
        "FRESH_REPLICAS_AFTER_SUCCESSFUL_HYDRATION"
    ),
    "decision_after_complete_roster_only": True,
    "error_report_raw_or_normalized_text_allowed": False,
    "event_semantics_exception_may_abort_roster_collection": False,
    "event_semantics_outcome_per_event_required": True,
    "replica_outcome_roster_identity_required": True,
    "semantic_error_examples": [
        "UNKNOWN_GRAMMAR",
        "UNAUTHORIZED_OR_MUTATED_MIGRATION_EPISODE",
        "NONCANONICAL_CHUNK_PARTITION",
        "MORE_THAN_EIGHT_MODEL_TEXT_CHUNKS",
        "STRICT_UTF8_FAILURE",
    ],
    "semantic_error_handling": (
        "ACCUMULATE_TYPED_AUDIT_ONLY_OUTCOME_CONTINUE_COMPLETE_ROSTER"
    ),
    "semantic_error_terminal_action": FAILURE_ACTION,
    "terminal_gate_result": (
        "PASS_ONLY_IF_COMPLETE_ROSTER_HAS_NO_UNAUTHORIZED_ERROR_OUTCOME"
    ),
}

SOURCE_MECHANISM_CONTRACT = {
    "base_semantics": (
        "PSIM_D5_UNCHANGED_EXCEPT_THIS_EXACT_FROZEN_D6_OVERLAY"
    ),
    "d5_failure_census": {
        "model_text_bound_error_events": 190,
        "model_text_bound_event_roster_hash": (
            D5_TEXT_BOUND_EVENT_ROSTER_HASH
        ),
        "reverse_administrative_migration_events": 365,
    },
    "gate_four_totality": copy.deepcopy(GATE_FOUR_TOTALITY_CONTRACT),
    "mechanism_probe_binding": copy.deepcopy(MECHANISM_PROBE_BINDING),
    "mechanism_probe_contract": copy.deepcopy(MECHANISM_PROBE_CONTRACT),
    "mechanism_version": MECHANISM_VERSION,
    "migration_restoration": copy.deepcopy(
        MIGRATION_RESTORATION_CONTRACT
    ),
    "model_text_transport": copy.deepcopy(
        MODEL_TEXT_TRANSPORT_CONTRACT
    ),
}

EXECUTION_AUTHORIZATION_CONTRACT = {
    "d5_forensic_or_source_root_reuse_allowed": False,
    "market_model_or_outcome_access_before_all_source_gates": False,
    "official_source_execution_authorized_by_this_preregistration": False,
    "required_before_official_source_execution": [
        "REVIEWED_D6_IMPLEMENTATION_COMMIT",
        "REVIEWED_D6_TEST_COMMIT",
        "CANONICAL_D6_EXECUTION_SEAL_BINDING_PREREGISTRATION_AND_CODE",
    ],
    "source_root_must_be_fresh": True,
    "synthetic_mechanism_probe_authorizes_official_execution": False,
}

BATCH_HYDRATION_CONTRACT = copy.deepcopy(d5.BATCH_HYDRATION_CONTRACT)
BATCH_HYDRATION_CONTRACT["trace_child_argv_ambiguity_action"] = (
    FAILURE_ACTION
)
BATCH_HYDRATION_CONTRACT["post_hydration_read"][
    "missing_object_action"
] = FAILURE_ACTION
BATCH_HYDRATION_CONTRACT["first_failure_action"] = FAILURE_ACTION
BATCH_HYDRATION_CONTRACT["forbidden_transports"] = [
    (
        "D1, D2, D3, D4, or D5 source-object reuse"
        if value == "D1, D2, D3, or D4 source-object reuse"
        else value
    )
    for value in BATCH_HYDRATION_CONTRACT["forbidden_transports"]
]

# Frozen after the recursive D5 -> D6 delta is reviewed.
AUTHORIZED_DELTA_PATHS = (
    "candidate.id",
    "candidate.name",
    "candidate.selection_commit",
    "daily_relation_contract.maximum_model_text_bytes_per_chunk",
    "daily_relation_contract.maximum_model_text_bytes_per_event",
    "daily_relation_contract.maximum_model_text_chunks_per_event",
    "daily_relation_contract.model_text_field",
    "daily_relation_contract.model_text_transport_contract",
    "decision_binding.commit",
    "decision_binding.path",
    "decision_binding.sha256",
    "event_contract.d6_source_mechanisms",
    "execution_authorization_contract",
    "memorization_contract.first_failure_action",
    "memorization_contract.model_text_chunk_aggregation_policy",
    "memorization_contract.model_text_field",
    "next_authorized_step",
    "parser_contract.metadata_parse_failure_action",
    "protocol_version",
    "representation_contract.model_text_field",
    "representation_contract.model_text_transport_contract",
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
    (
        "source_support_contract.control_sensitivity_metric."
        "first_failure_action"
    ),
    "source_support_contract.first_failure_action",
    "source_support_contract.gate_four_semantics",
    "source_support_contract.gate_four_totality_contract",
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
    "cb866ddfc173c294140725c391e0698cef779f6e3ee320dbdac6926f955bfbf0"
)
SOURCE_MECHANISM_CONTRACT_HASH = (
    "d03ea5ccacd415dab0f7d839b842d14e52492324c4fe432a723b6a2acb2c27b2"
)
BATCH_HYDRATION_CONTRACT_HASH = (
    "880d4cdc7775ff34faa06ebe0a67b9f6b10656739860eae2173500697be14104"
)
EXECUTION_AUTHORIZATION_CONTRACT_HASH = (
    "4fbbe236be3844a65cc4d65f0fd2420f3c8f464b4c0e2c2fba248171a327d0b2"
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


def _safe_output_path(path: str | Path) -> Path:
    requested = Path(path)
    results_root = REPO_ROOT.resolve() / "results"
    target = results_root / requested.name
    unsafe_existing_target = target.exists() and not target.is_file()
    if (
        requested.is_absolute()
        or requested.parent != Path("results")
        or results_root.is_symlink()
        or not results_root.is_dir()
        or requested.suffix != ".json"
        or target.is_symlink()
        or unsafe_existing_target
    ):
        raise RuntimeError(
            "PSIM-D6 preregistration output must be a safe repo-local result"
        )
    return target


def _read_canonical_json(path: str | Path) -> dict[str, Any]:
    target = repository_path(path)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"PSIM-D6 authority is absent or unsafe: {path}")
    raw = target.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"PSIM-D6 authority is unreadable: {path}"
        ) from error
    if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
        raise RuntimeError(f"PSIM-D6 authority is noncanonical: {path}")
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
        raise RuntimeError("PSIM-D6 decision authority changed")

    d5_registration = _read_canonical_json(D5_PREREGISTRATION_PATH)
    if (
        sha256_file(D5_PREREGISTRATION_PATH)
        != D5_PREREGISTRATION_SHA256
        or d5_registration.get("manifest_hash")
        != D5_PREREGISTRATION_MANIFEST_HASH
        or d5_registration != d5.build_preregistration()
        or sha256_file(D5_PREREGISTRATION_SCRIPT_PATH)
        != D5_PREREGISTRATION_SCRIPT_SHA256
        or sha256_file(D5_PREREGISTRATION_TEST_PATH)
        != D5_PREREGISTRATION_TEST_SHA256
        or sha256_file(D5_PREREGISTRATION_DOCUMENT_PATH)
        != D5_PREREGISTRATION_DOCUMENT_SHA256
    ):
        raise RuntimeError("PSIM-D5 preregistration authority changed")

    d5_terminal = _read_canonical_json(D5_TERMINAL_PATH)
    ledger = d5_terminal.get("access_ledger")
    gates = d5_terminal.get("gates")
    source_audit = d5_terminal.get("source_audit")
    if (
        sha256_file(D5_TERMINAL_PATH) != D5_TERMINAL_SHA256
        or d5_terminal.get("protocol_version")
        != "psim_d5_source_support_result_v1"
        or d5_terminal.get("result_hash") != D5_TERMINAL_RESULT_HASH
        or d5_terminal.get("decision") != "reject"
        or d5_terminal.get("first_failure")
        != {
            "gate_id": 4,
            "name": "historical_blob_preamble_dependency_integrity",
        }
        or d5_terminal.get("terminal_action") != D5_TERMINAL_ACTION
        or d5_terminal.get("outcomes_opened") is not False
        or d5_terminal.get("profitability_result") is not False
        or d5_terminal.get("error") != {"type": "ValueError"}
        or not isinstance(ledger, dict)
        or any(
            ledger.get(name) != 0
            for name in D5_TERMINAL_ZERO_LEDGER_FIELDS
        )
        or ledger.get("proposal_blobs_opened") != 5_206
        or ledger.get("proposal_text_rows_opened") != 5_206
        or not isinstance(gates, list)
        or len(gates) != 4
        or [row.get("passed") for row in gates]
        != [True, True, True, False]
        or not isinstance(source_audit, dict)
        or source_audit.get("source_root") != "/tmp/psim-d5-source"
        or source_audit.get("source_run_attempt") != 1
        or source_audit.get("repair_or_provider_swap_used") is not False
    ):
        raise RuntimeError("PSIM-D5 terminal authority changed")

    probe = _read_canonical_json(MECHANISM_PROBE_PATH)
    if (
        sha256_file(MECHANISM_PROBE_PATH) != MECHANISM_PROBE_SHA256
        or sha256_file(MECHANISM_PROBE_SCRIPT_PATH)
        != MECHANISM_PROBE_SCRIPT_SHA256
        or sha256_file(MECHANISM_PROBE_TEST_PATH)
        != MECHANISM_PROBE_TEST_SHA256
        or sha256_file(MECHANISM_PROBE_DOCUMENT_PATH)
        != MECHANISM_PROBE_DOCUMENT_SHA256
        or probe.get("result_hash") != MECHANISM_PROBE_RESULT_HASH
        or probe.get("protocol_version")
        != MECHANISM_PROBE_PROTOCOL_VERSION
        or probe.get("mechanism_version") != MECHANISM_VERSION
        or probe.get("mechanism_contract") != MECHANISM_PROBE_CONTRACT
        or probe.get("selection_scope")
        != MECHANISM_PROBE_BINDING["selection_scope"]
        or probe.get("synthetic_only") is not True
        or probe.get("access_boundary")
        != {
            "d5_census_artifact_read": True,
            "d5_forensic_root_accessed": False,
            "d5_run_invoked": False,
            "external_network_accessed_by_probe": False,
            "historical_proposal_text_accessed": False,
            "market_data_accessed": False,
            "model_accessed": False,
            "official_historical_proposal_source_accessed": False,
            "outcomes_accessed": False,
            "raw_official_text_published": False,
        }
        or probe != mechanism.build_probe()
    ):
        raise RuntimeError("PSIM-D6 mechanism probe authority changed")
    return d5_registration, d5_terminal, probe


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


def _successor_core(
    d5_registration: dict[str, Any],
) -> dict[str, Any]:
    core = _contract_core(d5_registration)
    core["protocol_version"] = PROTOCOL_VERSION
    core["candidate"] = {
        **core["candidate"],
        "id": POLICY_ID,
        "name": (
            "Protocol Specification Intent-Maturity source support, "
            "receipt-bound migration lifecycle plus lossless chunks"
        ),
        "selection_commit": DECISION_COMMIT,
    }
    core["decision_binding"] = {
        "path": DECISION_PATH.as_posix(),
        "commit": DECISION_COMMIT,
        "sha256": DECISION_SHA256,
    }

    core["parser_contract"]["metadata_parse_failure_action"] = (
        "RETAIN_D5_KNOWN_INVALID_WITHOUT_REPAIR_OR_EMIT_TYPED_D6_"
        "EVENT_ERROR"
    )

    core["event_contract"]["d6_source_mechanisms"] = copy.deepcopy(
        SOURCE_MECHANISM_CONTRACT
    )

    representation = core["representation_contract"]
    representation["model_text_field"] = "normalized_text_delta_chunks"
    representation["model_text_transport_contract"] = copy.deepcopy(
        MODEL_TEXT_TRANSPORT_CONTRACT
    )

    daily = core["daily_relation_contract"]
    daily["maximum_model_text_bytes_per_event"] = (
        mechanism.MAX_MODEL_TEXT_BYTES_PER_EVENT
    )
    daily["maximum_model_text_bytes_per_chunk"] = (
        mechanism.MAX_MODEL_TEXT_BYTES_PER_CHUNK
    )
    daily["maximum_model_text_chunks_per_event"] = (
        mechanism.MAX_MODEL_TEXT_CHUNKS_PER_EVENT
    )
    daily["model_text_field"] = "normalized_text_delta_chunks"
    daily["model_text_transport_contract"] = copy.deepcopy(
        MODEL_TEXT_TRANSPORT_CONTRACT
    )

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

    memorization = core["memorization_contract"]
    memorization["first_failure_action"] = MEMORIZATION_FAILURE_ACTION
    memorization["model_text_field"] = "normalized_text_delta_chunks"
    memorization["model_text_chunk_aggregation_policy"] = (
        MODEL_TEXT_TRANSPORT_CONTRACT["model_aggregation_policy"]
    )

    support = core["source_support_contract"]
    support["first_failure_action"] = FAILURE_ACTION
    support["control_sensitivity_metric"]["first_failure_action"] = (
        FAILURE_ACTION
    )
    support["gate_four_semantics"] = (
        "D5_BASE_PLUS_EXACT_FROZEN_365_RECEIPT_RESTORATIONS_PLUS_"
        "LOSSLESS_CANONICAL_UTF8_CHUNKS_OTHERWISE_COMPLETE_TYPED_"
        "ERROR_ROSTER_AND_REJECT"
    )
    support["gate_four_totality_contract"] = copy.deepcopy(
        GATE_FOUR_TOTALITY_CONTRACT
    )
    support["relation_control_transforms"]["old_new_direction_reverse"] = (
        "swap all old/new derived fields and exact side paths; map "
        "CREATE<->DELETE and retain UPDATE; swap explicit metadata states; "
        "reverse D5 normalized text-delta ADD/REMOVE rows before canonical "
        "D6 rechunking; retain administrative quarantine as model-hidden"
    )
    support["relation_control_transforms"]["protocol_label_swap"] = (
        "swap ethereum<->bitcoin labels after extraction; preserve raw event "
        "identity, exact side-path audit hash, metadata state, and underlying "
        "D5 normalized text-delta rows; rebuild canonical D6 chunks and "
        "relation cards; administrative quarantine remains model-hidden"
    )

    core["execution_authorization_contract"] = copy.deepcopy(
        EXECUTION_AUTHORIZATION_CONTRACT
    )
    core["next_authorized_step"] = (
        "implement, test, review, and seal a synthetic-only PSIM-D6 "
        "source-support evaluator; this preregistration does not authorize "
        "official source execution"
    )
    return core


def _transport_contract_rebased_to_d5(
    contract: dict[str, Any],
) -> dict[str, Any]:
    rebased = copy.deepcopy(contract)
    rebased["trace_child_argv_ambiguity_action"] = d5.FAILURE_ACTION
    rebased["post_hydration_read"][
        "missing_object_action"
    ] = d5.FAILURE_ACTION
    rebased["first_failure_action"] = d5.FAILURE_ACTION
    rebased["forbidden_transports"] = [
        (
            "D1, D2, D3, or D4 source-object reuse"
            if value == "D1, D2, D3, D4, or D5 source-object reuse"
            else value
        )
        for value in rebased["forbidden_transports"]
    ]
    return rebased


def build_preregistration() -> dict[str, Any]:
    d5_registration, d5_terminal, probe = _validate_authority()
    d5_core = _contract_core(d5_registration)
    successor = _successor_core(d5_registration)
    delta = _diff_values(d5_core, successor)
    if tuple(sorted(delta)) != tuple(sorted(AUTHORIZED_DELTA_PATHS)):
        raise RuntimeError(
            "PSIM-D6 inherited-contract delta changed: "
            + ",".join(sorted(delta))
        )
    if (
        canonical_hash(delta) != AUTHORIZED_DELTA_HASH
        or canonical_hash(SOURCE_MECHANISM_CONTRACT)
        != SOURCE_MECHANISM_CONTRACT_HASH
        or canonical_hash(BATCH_HYDRATION_CONTRACT)
        != BATCH_HYDRATION_CONTRACT_HASH
        or canonical_hash(EXECUTION_AUTHORIZATION_CONTRACT)
        != EXECUTION_AUTHORIZATION_CONTRACT_HASH
    ):
        raise RuntimeError("PSIM-D6 authorized source delta hash changed")
    if _transport_contract_rebased_to_d5(
        BATCH_HYDRATION_CONTRACT
    ) != d5.BATCH_HYDRATION_CONTRACT:
        raise RuntimeError("PSIM-D6 changed D5 hydration mechanics")
    if successor["source_support_contract"]["gates_in_order"] != (
        d5_core["source_support_contract"]["gates_in_order"]
    ):
        raise RuntimeError("PSIM-D6 source gate roster changed")
    if successor["source_support_contract"]["relation_controls"] != (
        d5_core["source_support_contract"]["relation_controls"]
    ):
        raise RuntimeError("PSIM-D6 relation-control roster changed")
    for key in (
        "availability_contract",
        "boundary_reset_contract",
        "bucket_contract",
        "excluded_feasibility_probe",
        "forbidden_access_contract",
        "official_sources",
        "split_contract",
    ):
        if successor[key] != d5_core[key]:
            raise RuntimeError(f"PSIM-D6 inherited {key} changed")
    if (
        successor["execution_authorization_contract"][
            "official_source_execution_authorized_by_this_preregistration"
        ]
        is not False
        or successor["event_contract"]["d6_source_mechanisms"][
            "migration_restoration"
        ]["generic_administrative_to_valid_transition_authorized"]
        is not False
        or successor["daily_relation_contract"][
            "model_text_transport_contract"
        ]["model_aggregation_policy"]
        != "UNDECIDED_NOT_AUTHORIZED_BY_D6_PREREGISTRATION"
    ):
        raise RuntimeError("PSIM-D6 execution or semantics scope expanded")

    census = probe["d5_census_binding"]
    inheritance = {
        "d5_preregistration": {
            "path": D5_PREREGISTRATION_PATH.as_posix(),
            "commit": D5_PREREGISTRATION_COMMIT,
            "sha256": D5_PREREGISTRATION_SHA256,
            "manifest_hash": D5_PREREGISTRATION_MANIFEST_HASH,
            "contract_core_hash": canonical_hash(d5_core),
            "producer": {
                "path": D5_PREREGISTRATION_SCRIPT_PATH.as_posix(),
                "sha256": D5_PREREGISTRATION_SCRIPT_SHA256,
            },
            "test": {
                "path": D5_PREREGISTRATION_TEST_PATH.as_posix(),
                "sha256": D5_PREREGISTRATION_TEST_SHA256,
            },
            "document": {
                "path": D5_PREREGISTRATION_DOCUMENT_PATH.as_posix(),
                "sha256": D5_PREREGISTRATION_DOCUMENT_SHA256,
            },
        },
        "d5_terminal_rejection": {
            "path": D5_TERMINAL_PATH.as_posix(),
            "commit": D5_TERMINAL_COMMIT,
            "sha256": D5_TERMINAL_SHA256,
            "result_hash": D5_TERMINAL_RESULT_HASH,
            "first_failure_gate_id": 4,
            "proposal_blobs_opened": d5_terminal["access_ledger"][
                "proposal_blobs_opened"
            ],
            "proposal_text_rows_opened": d5_terminal["access_ledger"][
                "proposal_text_rows_opened"
            ],
            "outcomes_opened": d5_terminal["outcomes_opened"],
        },
        "d5_post_terminal_census": copy.deepcopy(census),
        "d6_mechanism_probe": {
            **copy.deepcopy(MECHANISM_PROBE_BINDING),
            "mechanism_contract_hash": canonical_hash(
                MECHANISM_PROBE_CONTRACT
            ),
        },
        "authorized_delta_paths": list(AUTHORIZED_DELTA_PATHS),
        "authorized_delta": delta,
        "authorized_delta_hash": AUTHORIZED_DELTA_HASH,
        "source_mechanism_contract_hash": (
            SOURCE_MECHANISM_CONTRACT_HASH
        ),
        "batch_hydration_contract_hash": (
            BATCH_HYDRATION_CONTRACT_HASH
        ),
        "execution_authorization_contract_hash": (
            EXECUTION_AUTHORIZATION_CONTRACT_HASH
        ),
        "d5_transport_mechanics_byte_equal_after_namespace_rebase": True,
        "all_other_contract_paths_byte_equal": True,
        "preregistration_access": {
            "git_commands": 0,
            "network_calls": 0,
            "d5_preregistration_artifact_read": True,
            "d5_terminal_artifact_read": True,
            "d5_census_artifact_read": True,
            "d6_mechanism_probe_artifact_read": True,
            "d5_forensic_root_opened": False,
            "d5_source_runner_invoked": False,
            "d6_official_source_execution_invoked": False,
            "official_historical_proposal_source_opened": False,
            "market_model_outcomes_opened": False,
            "raw_official_text_published": False,
        },
    }
    core = {**successor, "inheritance_proof": inheritance}
    return {**core, "manifest_hash": canonical_hash(core)}


def write_preregistration(
    path: str | Path = DEFAULT_OUTPUT,
) -> Path:
    destination = _safe_output_path(path)
    raw = canonical_json_bytes(build_preregistration())
    if os.path.lexists(destination):
        if (
            destination.is_symlink()
            or not destination.is_file()
            or destination.read_bytes() != raw
        ):
            raise RuntimeError(
                f"existing PSIM-D6 preregistration differs: {destination}"
            )
        return destination
    temporary = destination.with_name(destination.name + ".tmp")
    if os.path.lexists(temporary):
        raise RuntimeError(
            f"unsafe PSIM-D6 preregistration temporary: {temporary}"
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
                "official_source_execution_authorized": payload[
                    "execution_authorization_contract"
                ][
                    "official_source_execution_authorized_by_this_"
                    "preregistration"
                ],
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
