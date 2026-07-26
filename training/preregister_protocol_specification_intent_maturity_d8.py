"""Preregister the final source-only PSIM-D8 relation-subcard successor.

This module reads only committed canonical authority artifacts. It never opens
the PSIM-D7 forensic/source root, creates a PSIM-D8 source root, executes a
source runner, loads a model, accesses market data, or inspects outcomes. The
preregistration permits a later reviewed implementation and direct-child
execution seal; it does not authorize an official PSIM-D8 source run.
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
    preregister_protocol_specification_intent_maturity_d7 as d7,
)
from training import (
    probe_protocol_specification_intent_maturity_d8_relation_subcard_mechanism
    as mechanism,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(
    "training/preregister_protocol_specification_intent_maturity_d8.py"
)
DEFAULT_OUTPUT = Path(
    "results/protocol_specification_intent_maturity_d8_preregistration_"
    "2026-07-27.json"
)

D7_PREREGISTRATION_PATH = d7.DEFAULT_OUTPUT
D7_PREREGISTRATION_COMMIT = (
    "107c6fe172c2dcba604b06ca67f23f136507b6e9"
)
D7_PREREGISTRATION_SHA256 = (
    "e9402b984232a9c30a5bc427ee8b828b4e61b7f355746e36ee5fe986be3ae79d"
)
D7_PREREGISTRATION_MANIFEST_HASH = (
    "7b6ac7c514bd3c0c8fad54a69707bb682a8a97bae020a603940c3410ddea378d"
)
D7_PREREGISTRATION_CORE_HASH = (
    "b8ed324c1b1da5f0137d5b6b33bcc09ad7720aef2e7c332a85f4f9da59ce07cb"
)
D7_PREREGISTRATION_SCRIPT_PATH = d7.SCRIPT_PATH
D7_PREREGISTRATION_SCRIPT_SHA256 = (
    "669494125becd1e1ed82a3a3048eaad23a063ef49c24a1bf613ba828695203fa"
)
D7_PREREGISTRATION_TEST_PATH = Path(
    "tests/test_preregister_protocol_specification_intent_maturity_d7.py"
)
D7_PREREGISTRATION_TEST_SHA256 = (
    "720795099c82dc7dc04da41c5749527e4680a4c8ee16a4366e47e13b17d9270f"
)
D7_PREREGISTRATION_DOCUMENT_PATH = Path(
    "docs/psim-d7-source-support-preregistration-2026-07-26.md"
)
D7_PREREGISTRATION_DOCUMENT_SHA256 = (
    "a8594400fe583d8efa00c09735b2110904eadc0f4c712a0974fe908db7fc6171"
)

D7_IMPLEMENTATION_COMMIT = (
    "0e8f22f2680a9edb2cf8497343444c16e4946df0"
)
D7_RUNNER_PATH = Path(
    "training/build_protocol_specification_intent_maturity_d7_source_support.py"
)
D7_RUNNER_SHA256 = (
    "75d4345a1d2e311a49bc7bec837f2345a6f630b5d5382485e2afb04cadb92a47"
)
D7_TEST_PATH = Path(
    "tests/test_build_protocol_specification_intent_maturity_d7_source_support.py"
)
D7_TEST_SHA256 = (
    "a9e00d86bb48811f95cfb417daef70baa33e203150c6202753c01f8d3921e887"
)

D7_SEAL_PATH = Path(
    "results/psim_d7_source_support_execution_seal_2026-07-27.json"
)
D7_SEAL_COMMIT = "3cb95185bad64b6e82fdd89f8e6f7f3eaa6fda72"
D7_SEAL_SHA256 = (
    "ea94ec6566b5925fb0be16bc30aae0e47f7215d42a202943e4d5213f144573d6"
)
D7_SEAL_HASH = (
    "8088c0902479612bb7cc64f0c729c7375640fcb095bdd9c3d0fe62dcd35fa308"
)

D7_TERMINAL_PATH = Path(
    "results/protocol_specification_intent_maturity_d7_source_rejection_"
    "2026-07-26.json"
)
D7_TERMINAL_COMMIT = "6d286943566dd13f591aeea41cbc767233822adf"
D7_TERMINAL_SHA256 = (
    "36702b4737f1bb37e901241a96e04f30e77132bb6a18ade1fab277a83f15557e"
)
D7_TERMINAL_RESULT_HASH = (
    "45846070617398860a03f5a401047c95a37c7ba3526c37fbcea5a11687e8658b"
)
D7_TERMINAL_ZERO_LEDGER_FIELDS = (
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

DECISION_PATH = Path(
    "docs/psim-d8-relation-subcard-mechanism-selection-2026-07-27.md"
)
DECISION_COMMIT = "211454a96695de44af3e009b751eff7df9e3ae5f"
DECISION_SHA256 = (
    "a46b2960586a2e80f78600a4069481de03ca80048c436e67fe09afe513d36385"
)
MECHANISM_PROBE_PATH = mechanism.DEFAULT_OUTPUT
MECHANISM_PROBE_COMMIT = DECISION_COMMIT
MECHANISM_PROBE_SHA256 = (
    "9c926f1fc44e60e4fcf92679dfd36db8d410220dcbbecec8c71e05bba0076d76"
)
MECHANISM_PROBE_RESULT_HASH = (
    "3b690e6e11399a12aca41a2ba79f74f5d8642f029dc5241d72d342a6f3706672"
)
MECHANISM_PROBE_SCENARIO_ROSTER_HASH = (
    "9a718845c1af15904a9d263511c601432d1ae3e2ddd17bad9e9bfb2fbefcc00c"
)
MECHANISM_PROBE_PROTOCOL_VERSION = mechanism.PROTOCOL_VERSION
MECHANISM_VERSION = mechanism.MECHANISM_VERSION
MECHANISM_PROBE_SCRIPT_PATH = Path(
    "training/"
    "probe_protocol_specification_intent_maturity_d8_relation_subcard_"
    "mechanism.py"
)
MECHANISM_PROBE_SCRIPT_SHA256 = (
    "b869c73b4ce1de783ff67888825ba3d0c41d8075050afe1c9b8ee6579f76fb4d"
)
MECHANISM_PROBE_TEST_PATH = Path(
    "tests/"
    "test_probe_protocol_specification_intent_maturity_d8_relation_subcard_"
    "mechanism.py"
)
MECHANISM_PROBE_TEST_SHA256 = (
    "9a2a99ab3ad1e4e4bf7ff98515831532142d39761f007ecc26e2d3b280e8fbc9"
)
MECHANISM_PROBE_DOCUMENT_PATH = DECISION_PATH
MECHANISM_PROBE_DOCUMENT_SHA256 = DECISION_SHA256

POLICY_ID = "PSIM-D8"
PROTOCOL_VERSION = "psim_d8_source_preregistration_v1"
SOURCE_ROOT = "/tmp/psim-d8-source"
FAILURE_ACTION = (
    "REJECT_AND_RETIRE_PSIM_PERMANENTLY_NO_D9_AFTER_D8_SOURCE_FAILURE_"
    "BEFORE_MARKET_MODEL_OR_OUTCOMES"
)
MEMORIZATION_FAILURE_ACTION = (
    "REJECT_AND_RETIRE_PSIM_AFTER_D8_MEMORIZATION_FAILURE_BEFORE_MARKET_"
    "OR_OUTCOMES"
)
SEALED_REF = "refs/psim-d8/sealed-tip"

ARTIFACT_PATHS = {
    "result": (
        "results/protocol_specification_intent_maturity_d8_source_support_"
        "2026-07-27.json"
    ),
    "rejection": (
        "results/protocol_specification_intent_maturity_d8_source_rejection_"
        "2026-07-27.json"
    ),
    "events": (
        "data/protocol_specification_intent_maturity_d8_events_"
        "2020_2023.jsonl.gz"
    ),
    "cards": (
        "data/protocol_specification_intent_maturity_d8_cards_"
        "2020_2024q1.jsonl.gz"
    ),
    "controls": (
        "results/protocol_specification_intent_maturity_d8_source_controls_"
        "2026-07-27.json"
    ),
}

MECHANISM_PROBE_BINDING = {
    "path": MECHANISM_PROBE_PATH.as_posix(),
    "commit": MECHANISM_PROBE_COMMIT,
    "sha256": MECHANISM_PROBE_SHA256,
    "result_hash": MECHANISM_PROBE_RESULT_HASH,
    "protocol_version": MECHANISM_PROBE_PROTOCOL_VERSION,
    "mechanism_version": MECHANISM_VERSION,
    "scenario_roster_hash": MECHANISM_PROBE_SCENARIO_ROSTER_HASH,
    "script_path": MECHANISM_PROBE_SCRIPT_PATH.as_posix(),
    "script_sha256": MECHANISM_PROBE_SCRIPT_SHA256,
    "test_path": MECHANISM_PROBE_TEST_PATH.as_posix(),
    "test_sha256": MECHANISM_PROBE_TEST_SHA256,
    "document_path": MECHANISM_PROBE_DOCUMENT_PATH.as_posix(),
    "document_sha256": MECHANISM_PROBE_DOCUMENT_SHA256,
    "synthetic_only": True,
    "selection_scope": (
        "AUTHORIZE_D8_PREREGISTRATION_ONLY_NOT_OFFICIAL_SOURCE_EXECUTION"
    ),
    "d7_forensic_source_root_accessed": False,
    "market_model_outcomes_accessed": False,
}

RELATION_SUBCARD_CONTRACT = {
    "logical_daily_card_count": (
        "EXACTLY_ONE_PER_SCHEDULE_AND_DECISION_DAY"
    ),
    "logical_daily_relation_roster": (
        "EXACT_D7_ORDERED_COMPLETE_RELATION_UNITS"
    ),
    "maximum_model_relation_units_per_subcard": 64,
    "subcard_partition": "CONTIGUOUS_GREEDY_SLICES_IN_ORIGINAL_ORDER",
    "subcard_coverage": "COMPLETE_NONOVERLAPPING_NO_GAP_NO_DUPLICATION",
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
}

MODEL_TEXT_TRANSPORT_CONTRACT = copy.deepcopy(
    d7.MODEL_TEXT_TRANSPORT_CONTRACT
)
MODEL_TEXT_TRANSPORT_CONTRACT["model_aggregation_policy"] = (
    "UNDECIDED_NOT_AUTHORIZED_BY_D8_SOURCE_PREREGISTRATION"
)

GATE_FOUR_TOTALITY_CONTRACT = copy.deepcopy(
    d7.GATE_FOUR_TOTALITY_CONTRACT
)
GATE_FOUR_TOTALITY_CONTRACT["semantic_error_terminal_action"] = (
    FAILURE_ACTION
)

BATCH_HYDRATION_CONTRACT = copy.deepcopy(d7.BATCH_HYDRATION_CONTRACT)
BATCH_HYDRATION_CONTRACT["trace_child_argv_ambiguity_action"] = (
    FAILURE_ACTION
)
BATCH_HYDRATION_CONTRACT["post_hydration_read"][
    "missing_object_action"
] = FAILURE_ACTION
BATCH_HYDRATION_CONTRACT["first_failure_action"] = FAILURE_ACTION
BATCH_HYDRATION_CONTRACT["forbidden_transports"] = [
    (
        "D1, D2, D3, D4, D5, D6, or D7 source-object reuse"
        if value == "D1, D2, D3, D4, D5, or D6 source-object reuse"
        else value
    )
    for value in BATCH_HYDRATION_CONTRACT["forbidden_transports"]
]

EXECUTION_AUTHORIZATION_CONTRACT = copy.deepcopy(
    d7.EXECUTION_AUTHORIZATION_CONTRACT
)
EXECUTION_AUTHORIZATION_CONTRACT.update(
    {
        "d7_forensic_or_source_root_reuse_allowed": False,
        "d8_is_last_source_representation_successor": True,
        "d9_source_successor_allowed_after_d8_failure": False,
        "official_source_execution_authorized_by_this_preregistration": (
            False
        ),
        "required_before_official_source_execution": [
            "REVIEWED_D8_IMPLEMENTATION_COMMIT",
            "REVIEWED_D8_TEST_COMMIT",
            (
                "CANONICAL_D8_DIRECT_CHILD_EXECUTION_SEAL_BINDING_"
                "PREREGISTRATION_AND_CODE"
            ),
        ],
        "synthetic_mechanism_probe_authorizes_official_execution": False,
    }
)

# Frozen after the recursive D7 -> D8 delta was independently inspected.
AUTHORIZED_DELTA_PATHS = (
    "candidate.d9_allowed_after_source_failure",
    "candidate.id",
    "candidate.last_source_representation_successor",
    "candidate.name",
    "candidate.selection_commit",
    "daily_relation_contract.maximum_model_events_per_card",
    "daily_relation_contract.maximum_model_relation_units_per_subcard",
    (
        "daily_relation_contract.model_text_transport_contract."
        "model_aggregation_policy"
    ),
    "daily_relation_contract.over_limit_card_action",
    "daily_relation_contract.relation_subcard_contract",
    "decision_binding.commit",
    "decision_binding.path",
    "decision_binding.sha256",
    (
        "execution_authorization_contract."
        "d7_forensic_or_source_root_reuse_allowed"
    ),
    (
        "execution_authorization_contract."
        "d8_is_last_source_representation_successor"
    ),
    (
        "execution_authorization_contract."
        "d9_source_successor_allowed_after_d8_failure"
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
    "protocol_version",
    "representation_contract.later_model_input",
    "representation_contract.logical_daily_card_payload_model_visible",
    "representation_contract.model_call_granularity",
    (
        "representation_contract.model_text_transport_contract."
        "model_aggregation_policy"
    ),
    "representation_contract.relation_subcard_manifest_model_visible",
    "representation_contract.single_model_single_call_per_card",
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
    (
        "source_support_contract.gate_four_totality_contract."
        "semantic_error_terminal_action"
    ),
)
AUTHORIZED_DELTA_HASH = (
    "33db9ba0fea552e24d62d16cd4bda84973fdae351977eb35f376df08599c543f"
)
RELATION_SUBCARD_CONTRACT_HASH = (
    "c86aaf1e9975d62c88c45f89dc6943fef7e2ed8902ecc840ea9f569e09e1e0fb"
)
BATCH_HYDRATION_CONTRACT_HASH = (
    "7eab28547cabb3aacf0c2cfa0498cc26e3de6b36d7e2f8d7b1a80fd6823d048d"
)
EXECUTION_AUTHORIZATION_CONTRACT_HASH = (
    "06f1012e5fe9246286a9d0b28da53877f846ef3c917a3023158893a601c0456f"
)
D7_AUTHORITY_BINDING_HASH = (
    "662a7fba3c6d5c86590c472e208f6479aaab6f593e04405652793bdec747a80f"
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


def _d7_authority_binding() -> dict[str, Any]:
    return {
        "preregistration": {
            "path": D7_PREREGISTRATION_PATH.as_posix(),
            "commit": D7_PREREGISTRATION_COMMIT,
            "sha256": D7_PREREGISTRATION_SHA256,
            "manifest_hash": D7_PREREGISTRATION_MANIFEST_HASH,
            "contract_core_hash": D7_PREREGISTRATION_CORE_HASH,
            "producer": {
                "path": D7_PREREGISTRATION_SCRIPT_PATH.as_posix(),
                "sha256": D7_PREREGISTRATION_SCRIPT_SHA256,
            },
            "test": {
                "path": D7_PREREGISTRATION_TEST_PATH.as_posix(),
                "sha256": D7_PREREGISTRATION_TEST_SHA256,
            },
            "document": {
                "path": D7_PREREGISTRATION_DOCUMENT_PATH.as_posix(),
                "sha256": D7_PREREGISTRATION_DOCUMENT_SHA256,
            },
        },
        "execution_seal": {
            "path": D7_SEAL_PATH.as_posix(),
            "commit": D7_SEAL_COMMIT,
            "sha256": D7_SEAL_SHA256,
            "seal_hash": D7_SEAL_HASH,
            "implementation_commit": D7_IMPLEMENTATION_COMMIT,
            "runner": {
                "path": D7_RUNNER_PATH.as_posix(),
                "sha256": D7_RUNNER_SHA256,
            },
            "test": {
                "path": D7_TEST_PATH.as_posix(),
                "sha256": D7_TEST_SHA256,
            },
        },
        "terminal_rejection": {
            "path": D7_TERMINAL_PATH.as_posix(),
            "commit": D7_TERMINAL_COMMIT,
            "parent_seal_commit": D7_SEAL_COMMIT,
            "terminal_is_direct_child_of_seal_commit": True,
            "sha256": D7_TERMINAL_SHA256,
            "result_hash": D7_TERMINAL_RESULT_HASH,
            "first_failure": {
                "gate_id": 5,
                "name": "split_annual_quarterly_unique_day_support",
            },
        },
    }


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
            "PSIM-D8 preregistration output must be a safe repo-local result"
        )
    return target


def _read_canonical_json(path: str | Path) -> dict[str, Any]:
    target = repository_path(path)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"PSIM-D8 authority is absent or unsafe: {path}")
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


def _validate_authority() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    d7_registration = _read_canonical_json(D7_PREREGISTRATION_PATH)
    d7_core = d7._contract_core(d7_registration)
    if (
        re.fullmatch(r"[0-9a-f]{40}", D7_PREREGISTRATION_COMMIT) is None
        or sha256_file(D7_PREREGISTRATION_PATH)
        != D7_PREREGISTRATION_SHA256
        or d7_registration.get("manifest_hash")
        != D7_PREREGISTRATION_MANIFEST_HASH
        or canonical_hash(d7_core) != D7_PREREGISTRATION_CORE_HASH
        or d7_registration != d7.build_preregistration()
        or sha256_file(D7_PREREGISTRATION_SCRIPT_PATH)
        != D7_PREREGISTRATION_SCRIPT_SHA256
        or sha256_file(D7_PREREGISTRATION_TEST_PATH)
        != D7_PREREGISTRATION_TEST_SHA256
        or sha256_file(D7_PREREGISTRATION_DOCUMENT_PATH)
        != D7_PREREGISTRATION_DOCUMENT_SHA256
    ):
        raise RuntimeError("PSIM-D7 preregistration authority changed")

    seal = _read_canonical_json(D7_SEAL_PATH)
    seal_core = {
        key: value for key, value in seal.items() if key != "seal_hash"
    }
    seal_authority = seal.get("authority")
    if (
        re.fullmatch(r"[0-9a-f]{40}", D7_SEAL_COMMIT) is None
        or sha256_file(D7_SEAL_PATH) != D7_SEAL_SHA256
        or seal.get("protocol_version")
        != "psim_d7_source_support_execution_seal_v1"
        or seal.get("policy_id") != "PSIM-D7"
        or seal.get("seal_hash") != D7_SEAL_HASH
        or seal.get("seal_hash") != canonical_hash(seal_core)
        or seal.get("shared_commit") != D7_IMPLEMENTATION_COMMIT
        or seal.get("runner")
        != {
            "commit": D7_IMPLEMENTATION_COMMIT,
            "path": D7_RUNNER_PATH.as_posix(),
            "sha256": D7_RUNNER_SHA256,
        }
        or seal.get("tests")
        != {
            "commit": D7_IMPLEMENTATION_COMMIT,
            "path": D7_TEST_PATH.as_posix(),
            "sha256": D7_TEST_SHA256,
        }
        or not isinstance(seal_authority, dict)
        or seal_authority.get("preregistration", {}).get("sha256")
        != D7_PREREGISTRATION_SHA256
        or seal_authority.get("preregistration_manifest_hash")
        != D7_PREREGISTRATION_MANIFEST_HASH
    ):
        raise RuntimeError("PSIM-D7 execution seal authority changed")

    terminal = _read_canonical_json(D7_TERMINAL_PATH)
    terminal_core = {
        key: value
        for key, value in terminal.items()
        if key != "result_hash"
    }
    ledger = terminal.get("access_ledger")
    source_audit = terminal.get("source_audit")
    gates = terminal.get("gates")
    terminal_seal = terminal.get("authority", {}).get("execution_seal")
    if (
        re.fullmatch(r"[0-9a-f]{40}", D7_TERMINAL_COMMIT) is None
        or sha256_file(D7_TERMINAL_PATH) != D7_TERMINAL_SHA256
        or terminal.get("protocol_version")
        != "psim_d7_source_support_result_v1"
        or terminal.get("policy_id") != "PSIM-D7"
        or terminal.get("result_hash") != D7_TERMINAL_RESULT_HASH
        or terminal.get("result_hash") != canonical_hash(terminal_core)
        or terminal.get("decision") != "reject"
        or terminal.get("first_failure")
        != {
            "gate_id": 5,
            "name": "split_annual_quarterly_unique_day_support",
        }
        or terminal.get("terminal_action") != d7.FAILURE_ACTION
        or terminal.get("error") != {"type": "ValueError"}
        or terminal.get("outcomes_opened") is not False
        or terminal.get("profitability_result") is not False
        or terminal.get("counts")
        != {"events": 5_356, "daily_cards": 0}
        or not isinstance(ledger, dict)
        or any(
            ledger.get(name) != 0
            for name in D7_TERMINAL_ZERO_LEDGER_FIELDS
        )
        or ledger.get("proposal_blobs_opened") != 11_280
        or ledger.get("proposal_text_rows_opened") != 11_280
        or not isinstance(source_audit, dict)
        or source_audit.get("source_root") != d7.SOURCE_ROOT
        or source_audit.get("source_run_attempt") != 1
        or source_audit.get("repair_or_provider_swap_used") is not False
        or not isinstance(gates, list)
        or len(gates) != 5
        or [row.get("passed") for row in gates]
        != [True, True, True, True, False]
        or not isinstance(terminal_seal, dict)
        or terminal_seal.get("path") != D7_SEAL_PATH.as_posix()
        or terminal_seal.get("seal_hash") != D7_SEAL_HASH
        or terminal_seal.get("sha256") != D7_SEAL_SHA256
        or terminal_seal.get("shared_commit") != D7_IMPLEMENTATION_COMMIT
    ):
        raise RuntimeError("PSIM-D7 terminal authority changed")

    probe = _read_canonical_json(MECHANISM_PROBE_PATH)
    battery = probe.get("synthetic_battery")
    access = probe.get("access_boundary")
    d7_authority = probe.get("d7_authority")
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
        or probe.get("mechanism_contract") != RELATION_SUBCARD_CONTRACT
        or not isinstance(battery, dict)
        or battery.get("all_passed") is not True
        or battery.get("scenario_count") != 12
        or battery.get("scenario_roster_hash")
        != MECHANISM_PROBE_SCENARIO_ROSTER_HASH
        or access
        != {
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
        }
        or not isinstance(d7_authority, dict)
        or d7_authority.get("terminal", {}).get("sha256")
        != D7_TERMINAL_SHA256
        or d7_authority.get("terminal", {}).get("result_hash")
        != D7_TERMINAL_RESULT_HASH
        or probe != mechanism.build_probe()
    ):
        raise RuntimeError("PSIM-D8 mechanism probe authority changed")
    return d7_registration, seal, terminal, probe


def _diff_values(
    left: Any,
    right: Any,
    *,
    path: str = "",
) -> dict[str, dict[str, Any]]:
    return d7._diff_values(left, right, path=path)


def _contract_core(registration: dict[str, Any]) -> dict[str, Any]:
    return d7._contract_core(registration)


def _successor_core(
    d7_registration: dict[str, Any],
) -> dict[str, Any]:
    core = _contract_core(d7_registration)
    core["protocol_version"] = PROTOCOL_VERSION
    core["candidate"] = {
        **core["candidate"],
        "id": POLICY_ID,
        "d9_allowed_after_source_failure": False,
        "last_source_representation_successor": True,
        "name": (
            "Protocol Specification Intent-Maturity source support, "
            "frozen D7 semantics plus lossless ordered relation subcards"
        ),
        "selection_commit": DECISION_COMMIT,
    }
    core["decision_binding"] = {
        "path": DECISION_PATH.as_posix(),
        "commit": DECISION_COMMIT,
        "sha256": DECISION_SHA256,
    }

    representation = core["representation_contract"]
    representation["model_text_transport_contract"] = copy.deepcopy(
        MODEL_TEXT_TRANSPORT_CONTRACT
    )
    representation["single_model_single_call_per_card"] = False
    representation["model_call_granularity"] = (
        "UNDECIDED_NOT_AUTHORIZED_BY_D8_SOURCE_PREREGISTRATION"
    )
    representation["logical_daily_card_payload_model_visible"] = False
    representation["relation_subcard_manifest_model_visible"] = False
    representation["later_model_input"] = (
        RELATION_SUBCARD_CONTRACT["later_model_input"]
    )

    daily = core["daily_relation_contract"]
    daily["model_text_transport_contract"] = copy.deepcopy(
        MODEL_TEXT_TRANSPORT_CONTRACT
    )
    daily["maximum_model_events_per_card"] = None
    daily["maximum_model_relation_units_per_subcard"] = (
        mechanism.MAX_RELATION_UNITS_PER_SUBCARD
    )
    daily["over_limit_card_action"] = (
        "BUILD_LOSSLESS_ORDERED_RELATION_SUBCARDS_NO_TRUNCATION_OR_"
        "SUMMARIZATION"
    )
    daily["relation_subcard_contract"] = copy.deepcopy(
        RELATION_SUBCARD_CONTRACT
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
    support["gate_four_totality_contract"] = copy.deepcopy(
        GATE_FOUR_TOTALITY_CONTRACT
    )

    core["execution_authorization_contract"] = copy.deepcopy(
        EXECUTION_AUTHORIZATION_CONTRACT
    )
    core["next_authorized_step"] = (
        "implement, test, review, and seal the final synthetic-only PSIM-D8 "
        "source-support evaluator; this preregistration does not authorize "
        "official source execution, and any D8 source failure retires PSIM "
        "without a D9 successor"
    )
    return core


def _transport_contract_rebased_to_d7(
    contract: dict[str, Any],
) -> dict[str, Any]:
    rebased = copy.deepcopy(contract)
    rebased["trace_child_argv_ambiguity_action"] = d7.FAILURE_ACTION
    rebased["post_hydration_read"][
        "missing_object_action"
    ] = d7.FAILURE_ACTION
    rebased["first_failure_action"] = d7.FAILURE_ACTION
    rebased["forbidden_transports"] = [
        (
            "D1, D2, D3, D4, D5, or D6 source-object reuse"
            if value
            == "D1, D2, D3, D4, D5, D6, or D7 source-object reuse"
            else value
        )
        for value in rebased["forbidden_transports"]
    ]
    return rebased


def _model_transport_rebased_to_d7(
    contract: dict[str, Any],
) -> dict[str, Any]:
    rebased = copy.deepcopy(contract)
    rebased["model_aggregation_policy"] = (
        "UNDECIDED_NOT_AUTHORIZED_BY_D7_PREREGISTRATION"
    )
    return rebased


def build_preregistration() -> dict[str, Any]:
    d7_registration, seal, terminal, probe = _validate_authority()
    d7_core = _contract_core(d7_registration)
    successor = _successor_core(d7_registration)
    delta = _diff_values(d7_core, successor)
    if tuple(sorted(delta)) != tuple(sorted(AUTHORIZED_DELTA_PATHS)):
        raise RuntimeError(
            "PSIM-D8 inherited-contract delta changed: "
            + ",".join(sorted(delta))
        )
    if (
        canonical_hash(delta) != AUTHORIZED_DELTA_HASH
        or canonical_hash(RELATION_SUBCARD_CONTRACT)
        != RELATION_SUBCARD_CONTRACT_HASH
        or canonical_hash(BATCH_HYDRATION_CONTRACT)
        != BATCH_HYDRATION_CONTRACT_HASH
        or canonical_hash(EXECUTION_AUTHORIZATION_CONTRACT)
        != EXECUTION_AUTHORIZATION_CONTRACT_HASH
        or canonical_hash(_d7_authority_binding())
        != D7_AUTHORITY_BINDING_HASH
    ):
        raise RuntimeError("PSIM-D8 authorized source delta hash changed")
    if _transport_contract_rebased_to_d7(
        BATCH_HYDRATION_CONTRACT
    ) != d7.BATCH_HYDRATION_CONTRACT:
        raise RuntimeError("PSIM-D8 changed D7 hydration mechanics")
    if _model_transport_rebased_to_d7(
        MODEL_TEXT_TRANSPORT_CONTRACT
    ) != d7.MODEL_TEXT_TRANSPORT_CONTRACT:
        raise RuntimeError("PSIM-D8 changed D7 model-text transport")
    for key in (
        "availability_contract",
        "boundary_reset_contract",
        "bucket_contract",
        "event_contract",
        "excluded_feasibility_probe",
        "forbidden_access_contract",
        "official_sources",
        "parser_contract",
        "split_contract",
    ):
        if successor[key] != d7_core[key]:
            raise RuntimeError(f"PSIM-D8 inherited {key} changed")
    if (
        successor["source_support_contract"]["gates_in_order"]
        != d7_core["source_support_contract"]["gates_in_order"]
        or successor["source_support_contract"]["relation_controls"]
        != d7_core["source_support_contract"]["relation_controls"]
        or successor["source_support_contract"][
            "control_sensitivity_metric"
        ]["denominator"]
        != d7_core["source_support_contract"][
            "control_sensitivity_metric"
        ]["denominator"]
    ):
        raise RuntimeError("PSIM-D8 source gate/control roster changed")
    if (
        successor["daily_relation_contract"][
            "relation_subcard_contract"
        ]
        != probe["mechanism_contract"]
        or successor["daily_relation_contract"][
            "maximum_model_events_per_card"
        ]
        is not None
        or successor["daily_relation_contract"][
            "maximum_model_relation_units_per_subcard"
        ]
        != 64
        or successor["representation_contract"][
            "logical_daily_card_payload_model_visible"
        ]
        is not False
        or successor["representation_contract"][
            "relation_subcard_manifest_model_visible"
        ]
        is not False
        or successor["representation_contract"][
            "single_model_single_call_per_card"
        ]
        is not False
        or successor["daily_relation_contract"][
            "model_text_transport_contract"
        ]["model_aggregation_policy"]
        != "UNDECIDED_NOT_AUTHORIZED_BY_D8_SOURCE_PREREGISTRATION"
    ):
        raise RuntimeError("PSIM-D8 relation-subcard scope expanded")
    execution = successor["execution_authorization_contract"]
    if (
        execution[
            "official_source_execution_authorized_by_this_preregistration"
        ]
        is not False
        or execution["d7_forensic_or_source_root_reuse_allowed"] is not False
        or execution["d8_is_last_source_representation_successor"]
        is not True
        or execution["d9_source_successor_allowed_after_d8_failure"]
        is not False
    ):
        raise RuntimeError("PSIM-D8 execution scope expanded")

    inheritance = {
        "d7_preregistration": {
            "path": D7_PREREGISTRATION_PATH.as_posix(),
            "commit": D7_PREREGISTRATION_COMMIT,
            "sha256": D7_PREREGISTRATION_SHA256,
            "manifest_hash": D7_PREREGISTRATION_MANIFEST_HASH,
            "contract_core_hash": D7_PREREGISTRATION_CORE_HASH,
            "producer": {
                "path": D7_PREREGISTRATION_SCRIPT_PATH.as_posix(),
                "sha256": D7_PREREGISTRATION_SCRIPT_SHA256,
            },
            "test": {
                "path": D7_PREREGISTRATION_TEST_PATH.as_posix(),
                "sha256": D7_PREREGISTRATION_TEST_SHA256,
            },
            "document": {
                "path": D7_PREREGISTRATION_DOCUMENT_PATH.as_posix(),
                "sha256": D7_PREREGISTRATION_DOCUMENT_SHA256,
            },
        },
        "d7_execution_seal": {
            "path": D7_SEAL_PATH.as_posix(),
            "commit": D7_SEAL_COMMIT,
            "sha256": D7_SEAL_SHA256,
            "seal_hash": D7_SEAL_HASH,
            "implementation_commit": D7_IMPLEMENTATION_COMMIT,
            "runner_sha256": D7_RUNNER_SHA256,
            "test_sha256": D7_TEST_SHA256,
            "payload_replay_hash": canonical_hash(seal),
        },
        "d7_terminal_rejection": {
            "path": D7_TERMINAL_PATH.as_posix(),
            "commit": D7_TERMINAL_COMMIT,
            "parent_seal_commit": D7_SEAL_COMMIT,
            "terminal_is_direct_child_of_seal_commit": True,
            "sha256": D7_TERMINAL_SHA256,
            "result_hash": D7_TERMINAL_RESULT_HASH,
            "first_failure_gate_id": 5,
            "events": terminal["counts"]["events"],
            "daily_cards": terminal["counts"]["daily_cards"],
            "proposal_blobs_opened": terminal["access_ledger"][
                "proposal_blobs_opened"
            ],
            "outcomes_opened": terminal["outcomes_opened"],
        },
        "d7_post_terminal_forensic": copy.deepcopy(
            probe["d7_authority"]["forensic"]
        ),
        "d7_observed_source_cardinality": copy.deepcopy(
            probe["d7_authority"]["observed_source_cardinality"]
        ),
        "d8_mechanism_probe": {
            **copy.deepcopy(MECHANISM_PROBE_BINDING),
            "mechanism_contract_hash": canonical_hash(
                probe["mechanism_contract"]
            ),
        },
        "authorized_delta_paths": list(AUTHORIZED_DELTA_PATHS),
        "authorized_delta": delta,
        "authorized_delta_hash": AUTHORIZED_DELTA_HASH,
        "relation_subcard_contract_hash": RELATION_SUBCARD_CONTRACT_HASH,
        "batch_hydration_contract_hash": (
            BATCH_HYDRATION_CONTRACT_HASH
        ),
        "execution_authorization_contract_hash": (
            EXECUTION_AUTHORIZATION_CONTRACT_HASH
        ),
        "d7_authority_binding_hash": D7_AUTHORITY_BINDING_HASH,
        "d7_hydration_mechanics_byte_equal_after_namespace_rebase": True,
        "d7_model_text_transport_byte_equal_after_aggregation_rebase": True,
        "d7_event_parser_source_split_schedule_gate_control_contracts_equal": (
            True
        ),
        "all_other_contract_paths_byte_equal": True,
        "preregistration_access": {
            "git_commands": 0,
            "network_calls": 0,
            "d7_preregistration_artifact_read": True,
            "d7_execution_seal_artifact_read": True,
            "d7_terminal_artifact_read": True,
            "d7_forensic_artifact_read_via_mechanism_probe": True,
            "d8_mechanism_probe_artifact_read": True,
            "d7_forensic_or_source_root_opened": False,
            "d7_source_runner_invoked": False,
            "d8_source_root_created_or_opened": False,
            "d8_official_source_execution_invoked": False,
            "official_historical_proposal_source_opened": False,
            "market_model_outcomes_opened": False,
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
                f"existing PSIM-D8 preregistration differs: {destination}"
            )
        return destination
    temporary = destination.with_name(destination.name + ".tmp")
    if os.path.lexists(temporary):
        raise RuntimeError(
            f"unsafe PSIM-D8 preregistration temporary: {temporary}"
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
                "d9_source_successor_allowed_after_d8_failure": payload[
                    "execution_authorization_contract"
                ]["d9_source_successor_allowed_after_d8_failure"],
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
