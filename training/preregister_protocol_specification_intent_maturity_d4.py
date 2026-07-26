"""Preregister the outcome-blind PSIM-D4 historical EIP parser candidate."""

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
    preregister_protocol_specification_intent_maturity_d3 as d3,
)
from training import (
    probe_protocol_specification_intent_maturity_d4_parser as parser_probe,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(
    "training/preregister_protocol_specification_intent_maturity_d4.py"
)
DEFAULT_OUTPUT = Path(
    "results/protocol_specification_intent_maturity_d4_preregistration_"
    "2026-07-26.json"
)

DECISION_PATH = Path(
    "docs/post-psim-d3-alpha-mechanism-audit-2026-07-26.md"
)
DECISION_COMMIT = "131009359c60bc5b28b76d22a63abf698011fbcb"
DECISION_SHA256 = (
    "2615736ba063c2b8e35811d3d01ab3517b345d74a2f7b70d248899aa393d7b99"
)

D3_PREREGISTRATION_PATH = d3.DEFAULT_OUTPUT
D3_PREREGISTRATION_COMMIT = (
    "1760d5945f0c8adc90ea667a21cbf6201eb5567e"
)
D3_PREREGISTRATION_SHA256 = (
    "332743f25d5be45ce4d022c67758051c01297f4cc18ccdf2138be75b5ef159ab"
)
D3_PREREGISTRATION_MANIFEST_HASH = (
    "d87358780df573bde11a317bf2e56f0ce044b3fc2fad3a28ef6e154d64023d86"
)

D3_TERMINAL_PATH = parser_probe.D3_TERMINAL_PATH
D3_TERMINAL_COMMIT = parser_probe.D3_TERMINAL_COMMIT
D3_TERMINAL_SHA256 = parser_probe.D3_TERMINAL_SHA256
D3_TERMINAL_RESULT_HASH = parser_probe.D3_TERMINAL_RESULT_HASH
D3_TERMINAL_ACTION = (
    "REJECT_PSIM_D3_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES"
)

PARSER_PROBE_PATH = parser_probe.DEFAULT_OUTPUT
PARSER_PROBE_COMMIT = DECISION_COMMIT
PARSER_PROBE_SHA256 = (
    "fbb97d65ef93b307c47055ed1883d6416e510a70b38083ae17ced2c78e4745ee"
)
PARSER_PROBE_RESULT_HASH = (
    "4a3cca52755716dbf6e9b4cd801e46b72bab841cea5609f6bb42519487e5f5e6"
)
PARSER_PROBE_PROTOCOL_VERSION = parser_probe.PROTOCOL_VERSION
PARSER_VERSION = parser_probe.PARSER_VERSION

POLICY_ID = "PSIM-D4"
PROTOCOL_VERSION = "psim_d4_source_preregistration_v1"
SOURCE_ROOT = "/tmp/psim-d4-source"
FAILURE_ACTION = (
    "REJECT_PSIM_D4_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES"
)
MEMORIZATION_FAILURE_ACTION = (
    "REJECT_PSIM_D4_UNCHANGED_BEFORE_MARKET_OR_OUTCOMES"
)
SEALED_REF = "refs/psim-d4/sealed-tip"

ARTIFACT_PATHS = {
    "result": (
        "results/protocol_specification_intent_maturity_d4_source_support_"
        "2026-07-26.json"
    ),
    "rejection": (
        "results/protocol_specification_intent_maturity_d4_source_rejection_"
        "2026-07-26.json"
    ),
    "events": (
        "data/protocol_specification_intent_maturity_d4_events_"
        "2020_2023.jsonl.gz"
    ),
    "cards": (
        "data/protocol_specification_intent_maturity_d4_cards_"
        "2020_2024q1.jsonl.gz"
    ),
    "controls": (
        "results/protocol_specification_intent_maturity_d4_source_controls_"
        "2026-07-26.json"
    ),
}

PARSER_DELTA_CONTRACT = {
    "scope": "EIP_FRONT_MATTER_ONLY_AFTER_D1_NORMALIZATION",
    "empty_line_semantics": "IGNORE_AS_NONSEMANTIC_SEPARATOR",
    "physical_line_classes": (
        "PHYSICAL_EMPTY_OR_ASCII_HORIZONTAL_WHITESPACE_ONLY_BOTH_"
        "NORMALIZE_TO_EMPTY_UNDER_FROZEN_D1_NORMALIZER"
    ),
    "header_bounds_position": "BEFORE_NORMALIZED_EMPTY_LINE_FILTER",
    "nonempty_line_parser": "UNCHANGED_PSIM_PREAMBLE_STATE_MACHINE_V1",
    "bip_parser": "IDENTICAL_D1_FUNCTION_OBJECT",
    "general_yaml_parser_adopted": False,
    "current_eipw_compatibility_claim": False,
}

PARSER_PROBE_BINDING = {
    "path": PARSER_PROBE_PATH.as_posix(),
    "commit": PARSER_PROBE_COMMIT,
    "sha256": PARSER_PROBE_SHA256,
    "result_hash": PARSER_PROBE_RESULT_HASH,
    "protocol_version": PARSER_PROBE_PROTOCOL_VERSION,
    "parser_version": PARSER_VERSION,
    "synthetic_only": True,
    "official_historical_proposal_source_opened": False,
    "d3_forensic_root_opened": False,
    "market_model_outcomes_opened": False,
    "probe_may_change_only_eip_normalized_empty_line_grammar": True,
}

BATCH_HYDRATION_CONTRACT = copy.deepcopy(d3.BATCH_HYDRATION_CONTRACT)
BATCH_HYDRATION_CONTRACT["trace_child_argv_ambiguity_action"] = FAILURE_ACTION
BATCH_HYDRATION_CONTRACT["post_hydration_read"][
    "missing_object_action"
] = FAILURE_ACTION
BATCH_HYDRATION_CONTRACT["first_failure_action"] = FAILURE_ACTION
BATCH_HYDRATION_CONTRACT["forbidden_transports"] = [
    (
        "D1, D2, or D3 source-object reuse"
        if value == "D1 or D2 source-object reuse"
        else value
    )
    for value in BATCH_HYDRATION_CONTRACT["forbidden_transports"]
]

AUTHORIZED_DELTA_PATHS = (
    "candidate.id",
    "candidate.name",
    "candidate.selection_commit",
    "decision_binding.commit",
    "decision_binding.path",
    "decision_binding.sha256",
    "memorization_contract.first_failure_action",
    "next_authorized_step",
    "parser_contract.eip_frontmatter.normalized_empty_line_contract",
    "parser_contract.reference_parser.eip_function",
    "parser_contract.reference_parser.synthetic_probe_binding",
    "parser_contract.reference_parser.version",
    "protocol_version",
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
    "source_support_contract.control_sensitivity_metric.first_failure_action",
    "source_support_contract.first_failure_action",
)

# Frozen after the implementation's recursive before/after delta was reviewed.
AUTHORIZED_DELTA_HASH = (
    "dd27b354bbe4c44052af2fab7b576198930487053947a93ef89a2977887b4eb1"
)
PARSER_DELTA_CONTRACT_HASH = (
    "6cc28c808e36b15470423bf6d728bb8033bff65d3dcf7dc50987f6ae2e779b3c"
)
BATCH_HYDRATION_CONTRACT_HASH = (
    "e07466131aba3aa0f5e39f73fbd95a070d39aa956e5b76c1778db8da8c78d3d2"
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
        raise RuntimeError(f"PSIM-D4 authority is absent or unsafe: {path}")
    raw = target.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"PSIM-D4 authority is unreadable: {path}") from error
    if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
        raise RuntimeError(f"PSIM-D4 authority is noncanonical: {path}")
    return payload


def _validate_authority() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    if re.fullmatch(r"[0-9a-f]{40}", DECISION_COMMIT) is None:
        raise RuntimeError("PSIM-D4 decision commit is malformed")
    if sha256_file(DECISION_PATH) != DECISION_SHA256:
        raise RuntimeError("PSIM-D4 decision document hash changed")

    d3_registration = _read_canonical_json(D3_PREREGISTRATION_PATH)
    if (
        sha256_file(D3_PREREGISTRATION_PATH)
        != D3_PREREGISTRATION_SHA256
        or d3_registration.get("manifest_hash")
        != D3_PREREGISTRATION_MANIFEST_HASH
        or d3_registration != d3.build_preregistration()
    ):
        raise RuntimeError("PSIM-D3 preregistration authority changed")

    d3_terminal = _read_canonical_json(D3_TERMINAL_PATH)
    ledger = d3_terminal.get("access_ledger", {})
    forbidden_terminal_counters = (
        "btc_market_rows_read",
        "cagr_values_built",
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
    gates = d3_terminal.get("gates")
    if (
        sha256_file(D3_TERMINAL_PATH) != D3_TERMINAL_SHA256
        or d3_terminal.get("result_hash") != D3_TERMINAL_RESULT_HASH
        or d3_terminal.get("decision") != "reject"
        or d3_terminal.get("first_failure")
        != {
            "gate_id": 4,
            "name": "historical_blob_preamble_dependency_integrity",
        }
        or d3_terminal.get("terminal_action") != D3_TERMINAL_ACTION
        or d3_terminal.get("source_incidence_opened") is not True
        or d3_terminal.get("outcomes_opened") is not False
        or d3_terminal.get("profitability_result") is not False
        or d3_terminal.get("error") != {"type": "ValueError"}
        or not isinstance(ledger, dict)
        or any(ledger.get(key) != 0 for key in forbidden_terminal_counters)
        or ledger.get("proposal_text_rows_opened") != 17
        or ledger.get("proposal_blobs_opened") != 5206
        or ledger.get("daily_cards_built") != 0
        or not isinstance(gates, list)
        or len(gates) != 4
        or [row.get("passed") for row in gates] != [True, True, True, False]
    ):
        raise RuntimeError("PSIM-D3 terminal authority changed")

    probe = _read_canonical_json(PARSER_PROBE_PATH)
    expected_battery = {
        "bip_acceptance_outputs_unchanged": 2,
        "bip_parser_identity_alias": True,
        "bip_rejections_preserved": 6,
        "d1_accepted_eip_outputs_unchanged": 5,
        "d1_nonblank_eip_rejections_preserved": 13,
        "header_byte_limit_counts_normalized_empty_lines": True,
        "header_line_limit_counts_normalized_empty_lines": True,
        "normalized_empty_acceptance_pairs_equal": 6,
        "normalized_empty_separator_does_not_rescue_empty_value": True,
        "synthetic_d3_failure_shape_accepted": True,
    }
    if (
        sha256_file(PARSER_PROBE_PATH) != PARSER_PROBE_SHA256
        or probe.get("result_hash") != PARSER_PROBE_RESULT_HASH
        or probe.get("protocol_version") != PARSER_PROBE_PROTOCOL_VERSION
        or probe.get("parser_version") != PARSER_VERSION
        or probe.get("synthetic_only") is not True
        or probe.get("selection_scope")
        != "AUTHORIZE_D4_PREREGISTRATION_ONLY_NOT_OFFICIAL_SOURCE_EXECUTION"
        or probe.get("access_boundary")
        != {
            "d3_forensic_root_accessed": False,
            "d3_terminal_artifact_read": True,
            "market_data_accessed": False,
            "model_accessed": False,
            "official_historical_proposal_source_accessed": False,
            "outcomes_accessed": False,
        }
        or probe.get("parser_battery") != expected_battery
        or probe != parser_probe.build_probe()
    ):
        raise RuntimeError("PSIM-D4 parser probe authority changed")
    return d3_registration, d3_terminal, probe


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


def _successor_core(d3_registration: dict[str, Any]) -> dict[str, Any]:
    core = _contract_core(d3_registration)
    core["protocol_version"] = PROTOCOL_VERSION
    core["candidate"] = {
        **core["candidate"],
        "id": POLICY_ID,
        "name": (
            "Protocol Specification Intent-Maturity relation RLLM, "
            "historical EIP normalized-empty separator grammar"
        ),
        "selection_commit": DECISION_COMMIT,
    }
    core["decision_binding"] = {
        "path": DECISION_PATH.as_posix(),
        "commit": DECISION_COMMIT,
        "sha256": DECISION_SHA256,
    }

    parser = core["parser_contract"]
    parser["reference_parser"]["version"] = PARSER_VERSION
    parser["reference_parser"]["eip_function"] = "parse_eip_preamble_d4"
    parser["reference_parser"]["synthetic_probe_binding"] = copy.deepcopy(
        PARSER_PROBE_BINDING
    )
    parser["eip_frontmatter"][
        "normalized_empty_line_contract"
    ] = copy.deepcopy(PARSER_DELTA_CONTRACT)

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
    support = core["source_support_contract"]
    support["first_failure_action"] = FAILURE_ACTION
    support["control_sensitivity_metric"]["first_failure_action"] = (
        FAILURE_ACTION
    )
    core["next_authorized_step"] = (
        "implement and seal synthetic-only PSIM-D4 historical EIP parser "
        "source-support evaluator"
    )
    return core


def _transport_contract_rebased_to_d3(
    contract: dict[str, Any],
) -> dict[str, Any]:
    rebased = copy.deepcopy(contract)
    rebased["trace_child_argv_ambiguity_action"] = d3.FAILURE_ACTION
    rebased["post_hydration_read"][
        "missing_object_action"
    ] = d3.FAILURE_ACTION
    rebased["first_failure_action"] = d3.FAILURE_ACTION
    rebased["forbidden_transports"] = [
        (
            "D1 or D2 source-object reuse"
            if value == "D1, D2, or D3 source-object reuse"
            else value
        )
        for value in rebased["forbidden_transports"]
    ]
    return rebased


def build_preregistration() -> dict[str, Any]:
    d3_registration, d3_terminal, probe = _validate_authority()
    d3_core = _contract_core(d3_registration)
    successor = _successor_core(d3_registration)
    delta = _diff_values(d3_core, successor)
    if tuple(sorted(delta)) != tuple(sorted(AUTHORIZED_DELTA_PATHS)):
        raise RuntimeError(
            "PSIM-D4 inherited-contract delta changed: "
            + ",".join(sorted(delta))
        )
    if (
        canonical_hash(delta) != AUTHORIZED_DELTA_HASH
        or canonical_hash(PARSER_DELTA_CONTRACT)
        != PARSER_DELTA_CONTRACT_HASH
        or canonical_hash(BATCH_HYDRATION_CONTRACT)
        != BATCH_HYDRATION_CONTRACT_HASH
    ):
        raise RuntimeError("PSIM-D4 authorized parser delta hash changed")
    if _transport_contract_rebased_to_d3(
        BATCH_HYDRATION_CONTRACT
    ) != d3.BATCH_HYDRATION_CONTRACT:
        raise RuntimeError("PSIM-D4 changed D3 hydration mechanics")
    if successor["source_support_contract"]["gates_in_order"] != list(
        d1.SOURCE_ONLY_GATES
    ):
        raise RuntimeError("PSIM-D4 source gate roster changed")
    if successor["source_support_contract"]["relation_controls"] != list(
        d1.RELATION_CONTROLS
    ):
        raise RuntimeError("PSIM-D4 control roster changed")
    for key in (
        "availability_contract",
        "boundary_reset_contract",
        "bucket_contract",
        "daily_relation_contract",
        "event_contract",
        "excluded_feasibility_probe",
        "forbidden_access_contract",
        "official_sources",
        "representation_contract",
        "split_contract",
    ):
        if successor[key] != d3_core[key]:
            raise RuntimeError(f"PSIM-D4 inherited {key} changed")

    inherited_parser = copy.deepcopy(successor["parser_contract"])
    inherited_parser["reference_parser"]["version"] = d3_core[
        "parser_contract"
    ]["reference_parser"]["version"]
    inherited_parser["reference_parser"]["eip_function"] = d3_core[
        "parser_contract"
    ]["reference_parser"]["eip_function"]
    inherited_parser["reference_parser"].pop("synthetic_probe_binding")
    inherited_parser["eip_frontmatter"].pop(
        "normalized_empty_line_contract"
    )
    if inherited_parser != d3_core["parser_contract"]:
        raise RuntimeError("PSIM-D4 changed an unauthorized parser rule")

    inheritance = {
        "d3_preregistration": {
            "path": D3_PREREGISTRATION_PATH.as_posix(),
            "commit": D3_PREREGISTRATION_COMMIT,
            "sha256": D3_PREREGISTRATION_SHA256,
            "manifest_hash": D3_PREREGISTRATION_MANIFEST_HASH,
            "contract_core_hash": canonical_hash(d3_core),
        },
        "d3_terminal_rejection": {
            "path": D3_TERMINAL_PATH.as_posix(),
            "commit": D3_TERMINAL_COMMIT,
            "sha256": D3_TERMINAL_SHA256,
            "result_hash": D3_TERMINAL_RESULT_HASH,
            "first_failure_gate_id": 4,
            "proposal_blobs_opened": d3_terminal["access_ledger"][
                "proposal_blobs_opened"
            ],
            "proposal_text_rows_opened": d3_terminal["access_ledger"][
                "proposal_text_rows_opened"
            ],
            "outcomes_opened": d3_terminal["outcomes_opened"],
        },
        "parser_probe": {
            "path": PARSER_PROBE_PATH.as_posix(),
            "commit": PARSER_PROBE_COMMIT,
            "sha256": PARSER_PROBE_SHA256,
            "result_hash": PARSER_PROBE_RESULT_HASH,
            "protocol_version": PARSER_PROBE_PROTOCOL_VERSION,
            "parser_version": PARSER_VERSION,
            "synthetic_only": probe["synthetic_only"],
        },
        "authorized_delta_paths": list(AUTHORIZED_DELTA_PATHS),
        "authorized_delta": delta,
        "authorized_delta_hash": AUTHORIZED_DELTA_HASH,
        "parser_delta_contract_hash": PARSER_DELTA_CONTRACT_HASH,
        "batch_hydration_contract_hash": BATCH_HYDRATION_CONTRACT_HASH,
        "d3_transport_mechanics_byte_equal_after_namespace_rebase": True,
        "all_other_contract_paths_byte_equal": True,
        "official_parser_documentation": [
            "https://eips.ethereum.org/EIPS/eip-1",
            "https://yaml.org/spec/1.2.2/#66-comments",
            "https://yaml.org/spec/1.2.2/#67-separation-lines",
            (
                "https://github.com/ethereum/eipw/blob/"
                "5d3cfc2585aadd5f3c8c2c223582e2f889c82bfa/"
                "eipw-preamble/src/lib.rs#L103-L155"
            ),
        ],
        "preregistration_access": {
            "git_commands": 0,
            "network_calls": 0,
            "d3_forensic_root_opened": False,
            "official_historical_proposal_source_opened": False,
            "market_model_outcomes_opened": False,
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
                f"existing PSIM-D4 preregistration differs: {destination}"
            )
        return destination
    temporary = destination.with_name(destination.name + ".tmp")
    if os.path.lexists(temporary):
        raise RuntimeError(
            f"unsafe PSIM-D4 preregistration temporary: {temporary}"
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
