"""Preregister the outcome-blind PSIM-D7 Bitcoin grammar successor.

This module reads only committed canonical authority artifacts. It does not
open a PSIM-D6 forensic/source root, execute a source runner, access a market
or model, or inspect an outcome. The preregistration permits a later reviewed
implementation and direct-child execution seal; it does not authorize an
official PSIM-D7 source run.
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
    preregister_protocol_specification_intent_maturity_d6 as d6,
)
from training import (
    probe_protocol_specification_intent_maturity_d7_bitcoin_grammar_mechanism
    as mechanism,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(
    "training/preregister_protocol_specification_intent_maturity_d7.py"
)
DEFAULT_OUTPUT = Path(
    "results/protocol_specification_intent_maturity_d7_preregistration_"
    "2026-07-26.json"
)

D6_PREREGISTRATION_PATH = d6.DEFAULT_OUTPUT
D6_PREREGISTRATION_COMMIT = (
    "a2ff036d03f01750da3527666e3be3d44737cbe2"
)
D6_PREREGISTRATION_SHA256 = (
    "9b6177ba02bf02783f7ddffe90cf4c5f1e385422ff658e17b28bf72d2f051d82"
)
D6_PREREGISTRATION_MANIFEST_HASH = (
    "0d6be5118ef7b34031af61bccc8a28944109db1a5411635ac2c822388e8895a6"
)
D6_PREREGISTRATION_CORE_HASH = (
    "630fa3f8f3c1f8b452fd8cd2857a7300c8188027c3ed043edf4b4096cb2f67ca"
)
D6_PREREGISTRATION_SCRIPT_PATH = d6.SCRIPT_PATH
D6_PREREGISTRATION_SCRIPT_SHA256 = (
    "e81cda6e88f298c3682d605ae8ab1b9e05ea1ae6cd50085eb1df9dea851a20b1"
)
D6_PREREGISTRATION_TEST_PATH = Path(
    "tests/test_preregister_protocol_specification_intent_maturity_d6.py"
)
D6_PREREGISTRATION_TEST_SHA256 = (
    "ced15cc7b7fafa1e1b60d27f979334eba39dc09c5eedfab0f99c370e0a31192b"
)
D6_PREREGISTRATION_DOCUMENT_PATH = Path(
    "docs/psim-d6-source-support-preregistration-2026-07-26.md"
)
D6_PREREGISTRATION_DOCUMENT_SHA256 = (
    "a4a8b2a9b0745e411d6290f0940c4af78aeb93790d34978a07295df3938626ff"
)

D6_IMPLEMENTATION_COMMIT = (
    "5c3f3f6d26046a8bc7b2f7ad09178d944d61e17b"
)
D6_RUNNER_PATH = Path(
    "training/build_protocol_specification_intent_maturity_d6_source_support.py"
)
D6_RUNNER_SHA256 = (
    "bc78fb2ff6ac0b4f0cebaedd01d03a75830f97be81cd9a736e47e6ead46a9f8f"
)
D6_TEST_PATH = Path(
    "tests/test_build_protocol_specification_intent_maturity_d6_source_support.py"
)
D6_TEST_SHA256 = (
    "eab0dad5e99d3480825f4f007d7b51b45d3f94552b7e1e81c80ca065a1a85fa3"
)

D6_SEAL_PATH = Path(
    "results/psim_d6_source_support_execution_seal_2026-07-26.json"
)
D6_SEAL_COMMIT = "8185e14b2e98fef6a4f8545828dc48b7d98417f2"
D6_SEAL_SHA256 = (
    "cf9bdbea467a499c6075059ef9275f00699fb0431fa27643751539ffdea64e1d"
)
D6_SEAL_HASH = (
    "5c9bb27b63375dd4e9bf7f7345115f8d8bf8910a84693a9c15b5c306c6bc2e54"
)

D6_TERMINAL_PATH = Path(
    "results/protocol_specification_intent_maturity_d6_source_rejection_"
    "2026-07-26.json"
)
D6_TERMINAL_COMMIT = "aef35e00f3ddcb91f6f4b6a37ff40d9d9f67a7a4"
D6_TERMINAL_SHA256 = (
    "f3e69893270be0d37299e78b651daa9208e1d05f07f24b39f6d1cf9a71c5d49f"
)
D6_TERMINAL_RESULT_HASH = (
    "052c8a0c5f3584a3c9a970f1fcfc434ebfd59a6aa25d5e087c6554aa3f2c31da"
)
D6_TERMINAL_ACTION = (
    "REJECT_PSIM_D6_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES"
)
D6_TERMINAL_ZERO_LEDGER_FIELDS = (
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

D6_CENSUS_COMMIT = mechanism.D6_CENSUS_COMMIT
D6_CENSUS_PATH = mechanism.D6_CENSUS_PATH
D6_CENSUS_SHA256 = mechanism.D6_CENSUS_SHA256
D6_CENSUS_RESULT_HASH = mechanism.D6_CENSUS_RESULT_HASH
D6_CENSUS_SCRIPT_PATH = mechanism.D6_CENSUS_SCRIPT_PATH
D6_CENSUS_SCRIPT_SHA256 = mechanism.D6_CENSUS_SCRIPT_SHA256
D6_CENSUS_TEST_PATH = mechanism.D6_CENSUS_TEST_PATH
D6_CENSUS_TEST_SHA256 = mechanism.D6_CENSUS_TEST_SHA256
D6_CENSUS_DOCUMENT_PATH = mechanism.D6_CENSUS_DOCUMENT_PATH
D6_CENSUS_DOCUMENT_SHA256 = mechanism.D6_CENSUS_DOCUMENT_SHA256

D6_MECHANISM_COMMIT = mechanism.D6_MECHANISM_COMMIT
D6_MECHANISM_PATH = mechanism.D6_MECHANISM_PATH
D6_MECHANISM_SHA256 = mechanism.D6_MECHANISM_SHA256
D6_MECHANISM_RESULT_HASH = mechanism.D6_MECHANISM_RESULT_HASH
D6_MECHANISM_VERSION = mechanism.D6_MECHANISM_VERSION
D6_MECHANISM_SCRIPT_PATH = mechanism.D6_MECHANISM_SCRIPT_PATH
D6_MECHANISM_SCRIPT_SHA256 = mechanism.D6_MECHANISM_SCRIPT_SHA256
D6_MECHANISM_TEST_PATH = mechanism.D6_MECHANISM_TEST_PATH
D6_MECHANISM_TEST_SHA256 = mechanism.D6_MECHANISM_TEST_SHA256
D6_MECHANISM_DOCUMENT_PATH = mechanism.D6_MECHANISM_DOCUMENT_PATH
D6_MECHANISM_DOCUMENT_SHA256 = (
    mechanism.D6_MECHANISM_DOCUMENT_SHA256
)
D6_SOURCE_MECHANISM_CONTRACT_HASH = (
    mechanism.D6_SOURCE_MECHANISM_CONTRACT_HASH
)

DECISION_PATH = Path(
    "docs/psim-d7-bitcoin-grammar-mechanism-selection-2026-07-26.md"
)
DECISION_COMMIT = "fe7f1d123eaf65af35d0d43e90f79faccbc53622"
DECISION_SHA256 = (
    "1fffa4b885006467742bffcaf33f41a0cac4300018f32f271db389876312d6d5"
)
MECHANISM_PROBE_PATH = mechanism.DEFAULT_OUTPUT
MECHANISM_PROBE_COMMIT = DECISION_COMMIT
MECHANISM_PROBE_SHA256 = (
    "2a549e6acfac2127527272ffe69986177b5e36f68f66623c6921ababac35ee94"
)
MECHANISM_PROBE_RESULT_HASH = (
    "832b1327d19b29f44f4fbd76dac312e001a7da19eb813ce41277d64a45492371"
)
MECHANISM_PROBE_PROTOCOL_VERSION = mechanism.PROTOCOL_VERSION
MECHANISM_VERSION = mechanism.MECHANISM_VERSION
MECHANISM_PROBE_SCRIPT_PATH = Path(
    "training/"
    "probe_protocol_specification_intent_maturity_d7_bitcoin_grammar_"
    "mechanism.py"
)
MECHANISM_PROBE_SCRIPT_SHA256 = (
    "f841ae20fb7db5e8129a4b4d6e5a712f8b4094c5ff51adfee917005e0feb53d5"
)
MECHANISM_PROBE_TEST_PATH = Path(
    "tests/"
    "test_probe_protocol_specification_intent_maturity_d7_bitcoin_grammar_"
    "mechanism.py"
)
MECHANISM_PROBE_TEST_SHA256 = (
    "9aace70243b2d5c2eaea940c3c410850c16e7f74775e2c995c270061bdebd5bf"
)
MECHANISM_PROBE_DOCUMENT_PATH = DECISION_PATH
MECHANISM_PROBE_DOCUMENT_SHA256 = DECISION_SHA256

POLICY_ID = "PSIM-D7"
PROTOCOL_VERSION = "psim_d7_source_preregistration_v1"
SOURCE_ROOT = "/tmp/psim-d7-source"
FAILURE_ACTION = (
    "REJECT_PSIM_D7_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES"
)
MEMORIZATION_FAILURE_ACTION = (
    "REJECT_PSIM_D7_UNCHANGED_BEFORE_MARKET_OR_OUTCOMES"
)
SEALED_REF = "refs/psim-d7/sealed-tip"

ARTIFACT_PATHS = {
    "result": (
        "results/protocol_specification_intent_maturity_d7_source_support_"
        "2026-07-26.json"
    ),
    "rejection": (
        "results/protocol_specification_intent_maturity_d7_source_rejection_"
        "2026-07-26.json"
    ),
    "events": (
        "data/protocol_specification_intent_maturity_d7_events_"
        "2020_2023.jsonl.gz"
    ),
    "cards": (
        "data/protocol_specification_intent_maturity_d7_cards_"
        "2020_2024q1.jsonl.gz"
    ),
    "controls": (
        "results/protocol_specification_intent_maturity_d7_source_controls_"
        "2026-07-26.json"
    ),
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
        "AUTHORIZE_D7_PREREGISTRATION_ONLY_NOT_OFFICIAL_SOURCE_EXECUTION"
    ),
    "d6_forensic_root_accessed": False,
    "market_model_outcomes_accessed": False,
}

D7_GRAMMAR_CONTRACT = {
    "base_semantics": (
        "PSIM_D6_UNCHANGED_EXCEPT_THIS_EXACT_FROZEN_D7_BITCOIN_OVERLAY"
    ),
    "bitcoin_only": True,
    "d6_exact_erc_migration_restoration_frozen": True,
    "d6_lossless_utf8_chunk_transport_frozen": True,
    "d6_max_bytes_per_chunk": 8_192,
    "d6_max_chunks_per_event": 8,
    "d6_ninth_chunk_action": (
        "FAIL_CLOSED_NO_TRUNCATION_OR_SUMMARIZATION"
    ),
    "d6_source_mechanism_contract_hash": (
        D6_SOURCE_MECHANISM_CONTRACT_HASH
    ),
    "d6_source_mechanisms_byte_equal": True,
    "ethereum_semantics_changed": False,
    "grammar_mechanism": copy.deepcopy(
        mechanism.build_probe()["mechanism_contract"]
    ),
    "identity_conditioned_allowlist": False,
    "mechanism_probe_binding": copy.deepcopy(MECHANISM_PROBE_BINDING),
    "mechanism_version": MECHANISM_VERSION,
    "unknown_or_ambiguous_grammar_action": (
        "RECORD_TYPED_EVENT_ERROR_CONTINUE_COMPLETE_ROSTER_THEN_REJECT_"
        "BEFORE_MODEL_OR_OUTCOMES"
    ),
}

MODEL_TEXT_TRANSPORT_CONTRACT = copy.deepcopy(
    d6.MODEL_TEXT_TRANSPORT_CONTRACT
)
MODEL_TEXT_TRANSPORT_CONTRACT["model_aggregation_policy"] = (
    "UNDECIDED_NOT_AUTHORIZED_BY_D7_PREREGISTRATION"
)

GATE_FOUR_TOTALITY_CONTRACT = copy.deepcopy(
    d6.GATE_FOUR_TOTALITY_CONTRACT
)
GATE_FOUR_TOTALITY_CONTRACT["semantic_error_terminal_action"] = (
    FAILURE_ACTION
)

BATCH_HYDRATION_CONTRACT = copy.deepcopy(d6.BATCH_HYDRATION_CONTRACT)
BATCH_HYDRATION_CONTRACT["trace_child_argv_ambiguity_action"] = (
    FAILURE_ACTION
)
BATCH_HYDRATION_CONTRACT["post_hydration_read"][
    "missing_object_action"
] = FAILURE_ACTION
BATCH_HYDRATION_CONTRACT["first_failure_action"] = FAILURE_ACTION
BATCH_HYDRATION_CONTRACT["forbidden_transports"] = [
    (
        "D1, D2, D3, D4, D5, or D6 source-object reuse"
        if value == "D1, D2, D3, D4, or D5 source-object reuse"
        else value
    )
    for value in BATCH_HYDRATION_CONTRACT["forbidden_transports"]
]

EXECUTION_AUTHORIZATION_CONTRACT = copy.deepcopy(
    d6.EXECUTION_AUTHORIZATION_CONTRACT
)
EXECUTION_AUTHORIZATION_CONTRACT.update(
    {
        "d6_forensic_or_source_root_reuse_allowed": False,
        "official_source_execution_authorized_by_this_preregistration": (
            False
        ),
        "required_before_official_source_execution": [
            "REVIEWED_D7_IMPLEMENTATION_COMMIT",
            "REVIEWED_D7_TEST_COMMIT",
            (
                "CANONICAL_D7_DIRECT_CHILD_EXECUTION_SEAL_BINDING_"
                "PREREGISTRATION_AND_CODE"
            ),
        ],
        "synthetic_mechanism_probe_authorizes_official_execution": False,
    }
)

# Frozen after the recursive D6 -> D7 delta was inspected.
AUTHORIZED_DELTA_PATHS = (
    "candidate.id",
    "candidate.name",
    "candidate.selection_commit",
    (
        "daily_relation_contract.model_text_transport_contract."
        "model_aggregation_policy"
    ),
    "decision_binding.commit",
    "decision_binding.path",
    "decision_binding.sha256",
    "event_contract.d7_bitcoin_grammar",
    (
        "execution_authorization_contract."
        "d6_forensic_or_source_root_reuse_allowed"
    ),
    (
        "execution_authorization_contract."
        "required_before_official_source_execution[0]"
    ),
    (
        "execution_authorization_contract."
        "required_before_official_source_execution[1]"
    ),
    (
        "execution_authorization_contract."
        "required_before_official_source_execution[2]"
    ),
    "memorization_contract.first_failure_action",
    "memorization_contract.model_text_chunk_aggregation_policy",
    "next_authorized_step",
    "parser_contract.metadata_parse_failure_action",
    "protocol_version",
    (
        "representation_contract.model_text_transport_contract."
        "model_aggregation_policy"
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
    (
        "source_support_contract.control_sensitivity_metric."
        "first_failure_action"
    ),
    "source_support_contract.first_failure_action",
    "source_support_contract.gate_four_semantics",
    (
        "source_support_contract.gate_four_totality_contract."
        "semantic_error_terminal_action"
    ),
)
AUTHORIZED_DELTA_HASH = (
    "b8295e977db265278d533d8fdc8f3dbf70e5905a04a95f7b936c191b0cd09440"
)
D7_GRAMMAR_CONTRACT_HASH = (
    "271b6b0447d392c341420d50f174aa8f5017c0c2f0fed99e453a4dacef00a977"
)
BATCH_HYDRATION_CONTRACT_HASH = (
    "98e9bb09e8d296d577020477c6e984c0333ab925e4aa68f22a13ab5d211cc492"
)
EXECUTION_AUTHORIZATION_CONTRACT_HASH = (
    "21c7a2722fa19a20d937a232ca658d11e3bd7c35d67c05f16d5ac0814ea039bb"
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
            "PSIM-D7 preregistration output must be a safe repo-local result"
        )
    return target


def _read_canonical_json(path: str | Path) -> dict[str, Any]:
    target = repository_path(path)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"PSIM-D7 authority is absent or unsafe: {path}")
    raw = target.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"PSIM-D7 authority is unreadable: {path}"
        ) from error
    if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
        raise RuntimeError(f"PSIM-D7 authority is noncanonical: {path}")
    return payload


def _validate_authority() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    d6_registration = _read_canonical_json(D6_PREREGISTRATION_PATH)
    d6_core = d6._contract_core(d6_registration)
    if (
        sha256_file(D6_PREREGISTRATION_PATH)
        != D6_PREREGISTRATION_SHA256
        or d6_registration.get("manifest_hash")
        != D6_PREREGISTRATION_MANIFEST_HASH
        or canonical_hash(d6_core) != D6_PREREGISTRATION_CORE_HASH
        or d6_registration != d6.build_preregistration()
        or sha256_file(D6_PREREGISTRATION_SCRIPT_PATH)
        != D6_PREREGISTRATION_SCRIPT_SHA256
        or sha256_file(D6_PREREGISTRATION_TEST_PATH)
        != D6_PREREGISTRATION_TEST_SHA256
        or sha256_file(D6_PREREGISTRATION_DOCUMENT_PATH)
        != D6_PREREGISTRATION_DOCUMENT_SHA256
    ):
        raise RuntimeError("PSIM-D6 preregistration authority changed")

    seal = _read_canonical_json(D6_SEAL_PATH)
    seal_authority = seal.get("authority")
    if (
        sha256_file(D6_SEAL_PATH) != D6_SEAL_SHA256
        or seal.get("protocol_version")
        != "psim_d6_source_support_execution_seal_v1"
        or seal.get("policy_id") != "PSIM-D6"
        or seal.get("seal_hash") != D6_SEAL_HASH
        or seal.get("shared_commit") != D6_IMPLEMENTATION_COMMIT
        or seal.get("runner")
        != {
            "commit": D6_IMPLEMENTATION_COMMIT,
            "path": D6_RUNNER_PATH.as_posix(),
            "sha256": D6_RUNNER_SHA256,
        }
        or seal.get("tests")
        != {
            "commit": D6_IMPLEMENTATION_COMMIT,
            "path": D6_TEST_PATH.as_posix(),
            "sha256": D6_TEST_SHA256,
        }
        or not isinstance(seal_authority, dict)
        or seal_authority.get("preregistration", {}).get("sha256")
        != D6_PREREGISTRATION_SHA256
        or seal_authority.get("preregistration_manifest_hash")
        != D6_PREREGISTRATION_MANIFEST_HASH
        or seal_authority.get("source_authority_hash")
        != "b63b42232d387af3ef9471ae6656857375ed4e94c548d510bdd4a874a9a9e963"
    ):
        raise RuntimeError("PSIM-D6 execution seal authority changed")

    terminal = _read_canonical_json(D6_TERMINAL_PATH)
    ledger = terminal.get("access_ledger")
    source_audit = terminal.get("source_audit")
    gates = terminal.get("gates")
    terminal_seal = terminal.get("authority", {}).get("execution_seal")
    if (
        sha256_file(D6_TERMINAL_PATH) != D6_TERMINAL_SHA256
        or terminal.get("protocol_version")
        != "psim_d6_source_support_result_v1"
        or terminal.get("policy_id") != "PSIM-D6"
        or terminal.get("result_hash") != D6_TERMINAL_RESULT_HASH
        or terminal.get("decision") != "reject"
        or terminal.get("first_failure")
        != {
            "gate_id": 4,
            "name": "historical_blob_preamble_dependency_integrity",
        }
        or terminal.get("terminal_action") != D6_TERMINAL_ACTION
        or terminal.get("error") is not None
        or terminal.get("outcomes_opened") is not False
        or terminal.get("profitability_result") is not False
        or not isinstance(ledger, dict)
        or any(
            ledger.get(name) != 0
            for name in D6_TERMINAL_ZERO_LEDGER_FIELDS
        )
        or ledger.get("proposal_blobs_opened") != 11_280
        or ledger.get("proposal_text_rows_opened") != 11_280
        or not isinstance(source_audit, dict)
        or source_audit.get("source_root") != "/tmp/psim-d6-source"
        or source_audit.get("source_run_attempt") != 1
        or source_audit.get("repair_or_provider_swap_used") is not False
        or not isinstance(gates, list)
        or len(gates) != 4
        or [row.get("passed") for row in gates]
        != [True, True, True, False]
        or terminal_seal
        != {
            "path": D6_SEAL_PATH.as_posix(),
            "runner": {
                "commit": D6_IMPLEMENTATION_COMMIT,
                "path": D6_RUNNER_PATH.as_posix(),
                "sha256": D6_RUNNER_SHA256,
            },
            "seal_hash": D6_SEAL_HASH,
            "sha256": D6_SEAL_SHA256,
            "shared_commit": D6_IMPLEMENTATION_COMMIT,
            "tests": {
                "commit": D6_IMPLEMENTATION_COMMIT,
                "path": D6_TEST_PATH.as_posix(),
                "sha256": D6_TEST_SHA256,
            },
        }
    ):
        raise RuntimeError("PSIM-D6 terminal authority changed")

    probe = _read_canonical_json(MECHANISM_PROBE_PATH)
    if (
        re.fullmatch(r"[0-9a-f]{40}", DECISION_COMMIT) is None
        or sha256_file(DECISION_PATH) != DECISION_SHA256
        or sha256_file(MECHANISM_PROBE_PATH)
        != MECHANISM_PROBE_SHA256
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
        or probe.get("selection_scope")
        != MECHANISM_PROBE_BINDING["selection_scope"]
        or probe.get("synthetic_only") is not True
        or probe.get("synthetic_battery", {}).get("all_passed") is not True
        or probe.get("synthetic_battery", {}).get("scenario_count") != 23
        or probe.get("access_boundary")
        != {
            "d6_census_artifact_read": True,
            "d6_forensic_root_accessed": False,
            "d6_run_invoked": False,
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
        raise RuntimeError("PSIM-D7 mechanism probe authority changed")

    census_binding = probe.get("d6_census_binding")
    d6_mechanism_binding = probe.get("d6_mechanism_binding")
    if (
        census_binding
        != {
            "commit": D6_CENSUS_COMMIT,
            "document": {
                "path": D6_CENSUS_DOCUMENT_PATH.as_posix(),
                "sha256": D6_CENSUS_DOCUMENT_SHA256,
            },
            "grammar_class_counts": {
                "BIP_LATER_EXACT_PRE_HEADER_AFTER_NONHEADER_PREFIX": 7,
                "BIP_PREFIXED_DECIMAL_DEPENDENCY_TOKEN": 1,
                "D4_VALID": 426,
            },
            "path": D6_CENSUS_PATH.as_posix(),
            "result_hash": D6_CENSUS_RESULT_HASH,
            "script": {
                "path": D6_CENSUS_SCRIPT_PATH.as_posix(),
                "sha256": D6_CENSUS_SCRIPT_SHA256,
            },
            "sha256": D6_CENSUS_SHA256,
            "test": {
                "path": D6_CENSUS_TEST_PATH.as_posix(),
                "sha256": D6_CENSUS_TEST_SHA256,
            },
            "unknown_grammar_count": 0,
        }
        or d6_mechanism_binding
        != {
            "commit": D6_MECHANISM_COMMIT,
            "document": {
                "path": D6_MECHANISM_DOCUMENT_PATH.as_posix(),
                "sha256": D6_MECHANISM_DOCUMENT_SHA256,
            },
            "mechanism_version": D6_MECHANISM_VERSION,
            "path": D6_MECHANISM_PATH.as_posix(),
            "result_hash": D6_MECHANISM_RESULT_HASH,
            "script": {
                "path": D6_MECHANISM_SCRIPT_PATH.as_posix(),
                "sha256": D6_MECHANISM_SCRIPT_SHA256,
            },
            "sha256": D6_MECHANISM_SHA256,
            "source_mechanism_contract_hash": (
                D6_SOURCE_MECHANISM_CONTRACT_HASH
            ),
            "test": {
                "path": D6_MECHANISM_TEST_PATH.as_posix(),
                "sha256": D6_MECHANISM_TEST_SHA256,
            },
        }
    ):
        raise RuntimeError("PSIM-D7 inherited source authority changed")
    return d6_registration, seal, terminal, probe


def _diff_values(
    left: Any,
    right: Any,
    *,
    path: str = "",
) -> dict[str, dict[str, Any]]:
    return d6._diff_values(left, right, path=path)


def _contract_core(registration: dict[str, Any]) -> dict[str, Any]:
    return d6._contract_core(registration)


def _successor_core(
    d6_registration: dict[str, Any],
) -> dict[str, Any]:
    core = _contract_core(d6_registration)
    core["protocol_version"] = PROTOCOL_VERSION
    core["candidate"] = {
        **core["candidate"],
        "id": POLICY_ID,
        "name": (
            "Protocol Specification Intent-Maturity source support, "
            "frozen D6 transport plus Bitcoin grammar overlay"
        ),
        "selection_commit": DECISION_COMMIT,
    }
    core["decision_binding"] = {
        "path": DECISION_PATH.as_posix(),
        "commit": DECISION_COMMIT,
        "sha256": DECISION_SHA256,
    }

    core["parser_contract"]["metadata_parse_failure_action"] = (
        "RETAIN_D5_KNOWN_INVALID_WITHOUT_REPAIR_OR_EMIT_TYPED_D7_"
        "EVENT_ERROR"
    )
    core["event_contract"]["d7_bitcoin_grammar"] = copy.deepcopy(
        D7_GRAMMAR_CONTRACT
    )

    representation = core["representation_contract"]
    representation["model_text_transport_contract"] = copy.deepcopy(
        MODEL_TEXT_TRANSPORT_CONTRACT
    )
    daily = core["daily_relation_contract"]
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
    memorization["model_text_chunk_aggregation_policy"] = (
        MODEL_TEXT_TRANSPORT_CONTRACT["model_aggregation_policy"]
    )

    support = core["source_support_contract"]
    support["first_failure_action"] = FAILURE_ACTION
    support["control_sensitivity_metric"]["first_failure_action"] = (
        FAILURE_ACTION
    )
    support["gate_four_semantics"] = (
        "D6_EXACT_FROZEN_MIGRATION_AND_CHUNK_SEMANTICS_PLUS_EXACT_D7_"
        "BITCOIN_UNIQUE_LATER_HEADER_AND_PREFIXED_DEPENDENCY_GRAMMAR_"
        "OTHERWISE_COMPLETE_TYPED_ERROR_ROSTER_AND_REJECT"
    )
    support["gate_four_totality_contract"] = copy.deepcopy(
        GATE_FOUR_TOTALITY_CONTRACT
    )

    core["execution_authorization_contract"] = copy.deepcopy(
        EXECUTION_AUTHORIZATION_CONTRACT
    )
    core["next_authorized_step"] = (
        "implement, test, review, and seal a synthetic-only PSIM-D7 "
        "source-support evaluator; this preregistration does not authorize "
        "official source execution"
    )
    return core


def _transport_contract_rebased_to_d6(
    contract: dict[str, Any],
) -> dict[str, Any]:
    rebased = copy.deepcopy(contract)
    rebased["trace_child_argv_ambiguity_action"] = d6.FAILURE_ACTION
    rebased["post_hydration_read"][
        "missing_object_action"
    ] = d6.FAILURE_ACTION
    rebased["first_failure_action"] = d6.FAILURE_ACTION
    rebased["forbidden_transports"] = [
        (
            "D1, D2, D3, D4, or D5 source-object reuse"
            if value
            == "D1, D2, D3, D4, D5, or D6 source-object reuse"
            else value
        )
        for value in rebased["forbidden_transports"]
    ]
    return rebased


def _model_transport_rebased_to_d6(
    contract: dict[str, Any],
) -> dict[str, Any]:
    rebased = copy.deepcopy(contract)
    rebased["model_aggregation_policy"] = (
        "UNDECIDED_NOT_AUTHORIZED_BY_D6_PREREGISTRATION"
    )
    return rebased


def build_preregistration() -> dict[str, Any]:
    d6_registration, seal, terminal, probe = _validate_authority()
    d6_core = _contract_core(d6_registration)
    successor = _successor_core(d6_registration)
    delta = _diff_values(d6_core, successor)
    if tuple(sorted(delta)) != tuple(sorted(AUTHORIZED_DELTA_PATHS)):
        raise RuntimeError(
            "PSIM-D7 inherited-contract delta changed: "
            + ",".join(sorted(delta))
        )
    if (
        canonical_hash(delta) != AUTHORIZED_DELTA_HASH
        or canonical_hash(D7_GRAMMAR_CONTRACT)
        != D7_GRAMMAR_CONTRACT_HASH
        or canonical_hash(BATCH_HYDRATION_CONTRACT)
        != BATCH_HYDRATION_CONTRACT_HASH
        or canonical_hash(EXECUTION_AUTHORIZATION_CONTRACT)
        != EXECUTION_AUTHORIZATION_CONTRACT_HASH
    ):
        raise RuntimeError("PSIM-D7 authorized source delta hash changed")
    if _transport_contract_rebased_to_d6(
        BATCH_HYDRATION_CONTRACT
    ) != d6.BATCH_HYDRATION_CONTRACT:
        raise RuntimeError("PSIM-D7 changed D6 hydration mechanics")
    if _model_transport_rebased_to_d6(
        MODEL_TEXT_TRANSPORT_CONTRACT
    ) != d6.MODEL_TEXT_TRANSPORT_CONTRACT:
        raise RuntimeError("PSIM-D7 changed D6 model-text transport")
    if (
        successor["event_contract"]["d6_source_mechanisms"]
        != d6_core["event_contract"]["d6_source_mechanisms"]
    ):
        raise RuntimeError("PSIM-D7 changed frozen D6 source mechanisms")
    if (
        successor["source_support_contract"]["gates_in_order"]
        != d6_core["source_support_contract"]["gates_in_order"]
        or successor["source_support_contract"]["relation_controls"]
        != d6_core["source_support_contract"]["relation_controls"]
    ):
        raise RuntimeError("PSIM-D7 source gate/control roster changed")
    for key in (
        "availability_contract",
        "boundary_reset_contract",
        "bucket_contract",
        "excluded_feasibility_probe",
        "forbidden_access_contract",
        "official_sources",
        "split_contract",
    ):
        if successor[key] != d6_core[key]:
            raise RuntimeError(f"PSIM-D7 inherited {key} changed")
    d6_source = d6_core["source_contract"]
    d7_source = successor["source_contract"]
    for key in (
        "card_end_exclusive",
        "clone_arguments",
        "end_exclusive",
        "repositories",
        "start",
        "traversal",
    ):
        if key == "repositories":
            d6_repositories = copy.deepcopy(d6_source[key])
            for repository in d6_repositories:
                repository["sealed_ref"] = SEALED_REF
            if d7_source[key] != d6_repositories:
                raise RuntimeError("PSIM-D7 repository authority changed")
        elif d7_source[key] != d6_source[key]:
            raise RuntimeError(f"PSIM-D7 source {key} changed")
    if (
        successor["execution_authorization_contract"][
            "official_source_execution_authorized_by_this_preregistration"
        ]
        is not False
        or successor["execution_authorization_contract"][
            "d6_forensic_or_source_root_reuse_allowed"
        ]
        is not False
        or successor["event_contract"]["d7_bitcoin_grammar"][
            "identity_conditioned_allowlist"
        ]
        is not False
        or successor["daily_relation_contract"][
            "model_text_transport_contract"
        ]["model_aggregation_policy"]
        != "UNDECIDED_NOT_AUTHORIZED_BY_D7_PREREGISTRATION"
    ):
        raise RuntimeError("PSIM-D7 execution or semantics scope expanded")

    inheritance = {
        "d6_preregistration": {
            "path": D6_PREREGISTRATION_PATH.as_posix(),
            "commit": D6_PREREGISTRATION_COMMIT,
            "sha256": D6_PREREGISTRATION_SHA256,
            "manifest_hash": D6_PREREGISTRATION_MANIFEST_HASH,
            "contract_core_hash": D6_PREREGISTRATION_CORE_HASH,
            "producer": {
                "path": D6_PREREGISTRATION_SCRIPT_PATH.as_posix(),
                "sha256": D6_PREREGISTRATION_SCRIPT_SHA256,
            },
            "test": {
                "path": D6_PREREGISTRATION_TEST_PATH.as_posix(),
                "sha256": D6_PREREGISTRATION_TEST_SHA256,
            },
            "document": {
                "path": D6_PREREGISTRATION_DOCUMENT_PATH.as_posix(),
                "sha256": D6_PREREGISTRATION_DOCUMENT_SHA256,
            },
        },
        "d6_execution_seal": {
            "path": D6_SEAL_PATH.as_posix(),
            "commit": D6_SEAL_COMMIT,
            "sha256": D6_SEAL_SHA256,
            "seal_hash": D6_SEAL_HASH,
            "implementation_commit": D6_IMPLEMENTATION_COMMIT,
            "runner_sha256": D6_RUNNER_SHA256,
            "test_sha256": D6_TEST_SHA256,
            "payload_replay_hash": canonical_hash(seal),
        },
        "d6_terminal_rejection": {
            "path": D6_TERMINAL_PATH.as_posix(),
            "commit": D6_TERMINAL_COMMIT,
            "parent_seal_commit": D6_SEAL_COMMIT,
            "terminal_is_direct_child_of_seal_commit": True,
            "sha256": D6_TERMINAL_SHA256,
            "result_hash": D6_TERMINAL_RESULT_HASH,
            "first_failure_gate_id": 4,
            "proposal_blobs_opened": terminal["access_ledger"][
                "proposal_blobs_opened"
            ],
            "proposal_text_rows_opened": terminal["access_ledger"][
                "proposal_text_rows_opened"
            ],
            "outcomes_opened": terminal["outcomes_opened"],
        },
        "d6_post_terminal_bitcoin_grammar_census": copy.deepcopy(
            probe["d6_census_binding"]
        ),
        "d6_frozen_source_mechanism": copy.deepcopy(
            probe["d6_mechanism_binding"]
        ),
        "d7_mechanism_probe": {
            **copy.deepcopy(MECHANISM_PROBE_BINDING),
            "mechanism_contract_hash": canonical_hash(
                probe["mechanism_contract"]
            ),
            "scenario_roster_hash": probe["synthetic_battery"][
                "scenario_roster_hash"
            ],
        },
        "authorized_delta_paths": list(AUTHORIZED_DELTA_PATHS),
        "authorized_delta": delta,
        "authorized_delta_hash": AUTHORIZED_DELTA_HASH,
        "d7_grammar_contract_hash": D7_GRAMMAR_CONTRACT_HASH,
        "batch_hydration_contract_hash": (
            BATCH_HYDRATION_CONTRACT_HASH
        ),
        "execution_authorization_contract_hash": (
            EXECUTION_AUTHORIZATION_CONTRACT_HASH
        ),
        "d6_hydration_mechanics_byte_equal_after_namespace_rebase": True,
        "d6_model_text_transport_byte_equal_after_aggregation_rebase": True,
        "d6_source_mechanisms_byte_equal": True,
        "all_other_contract_paths_byte_equal": True,
        "preregistration_access": {
            "git_commands": 0,
            "network_calls": 0,
            "d6_preregistration_artifact_read": True,
            "d6_execution_seal_artifact_read": True,
            "d6_terminal_artifact_read": True,
            "d6_census_artifact_read": True,
            "d6_mechanism_probe_artifact_read": True,
            "d7_mechanism_probe_artifact_read": True,
            "d6_forensic_or_source_root_opened": False,
            "d6_source_runner_invoked": False,
            "d7_official_source_execution_invoked": False,
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
                f"existing PSIM-D7 preregistration differs: {destination}"
            )
        return destination
    temporary = destination.with_name(destination.name + ".tmp")
    if os.path.lexists(temporary):
        raise RuntimeError(
            f"unsafe PSIM-D7 preregistration temporary: {temporary}"
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
