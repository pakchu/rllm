"""Preregister the outcome-blind PSIM-D3 batch-hydration candidate."""

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
    preregister_protocol_specification_intent_maturity_d2 as d2,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(
    "training/preregister_protocol_specification_intent_maturity_d3.py"
)
DEFAULT_OUTPUT = Path(
    "results/protocol_specification_intent_maturity_d3_preregistration_"
    "2026-07-25.json"
)

DECISION_PATH = Path(
    "docs/post-psim-d2-alpha-mechanism-audit-2026-07-25.md"
)
DECISION_COMMIT = "126f7f1354eff90f30d5a6b3d60bd6641268b03b"
DECISION_SHA256 = (
    "7fecb77f93bdf0f78cbdb45afbf866d3c726944627ed49bdf56ef69f0535ba4a"
)

D2_PREREGISTRATION_PATH = d2.DEFAULT_OUTPUT
D2_PREREGISTRATION_COMMIT = (
    "e853f7688a484b323c024115e3ef4af07e6a5896"
)
D2_PREREGISTRATION_SHA256 = (
    "3b405de2bcdc1979855e8505148f7de3fbee366cb126e78b1b23e10f84cf470a"
)
D2_PREREGISTRATION_MANIFEST_HASH = (
    "917d2f318b268b01621c9e969237d76fc82d7e6aff408269842e660cc155d915"
)

D2_TERMINAL_PATH = Path(
    "results/protocol_specification_intent_maturity_d2_source_rejection_"
    "2026-07-25.json"
)
D2_TERMINAL_COMMIT = "0e98ba563fb38012f7cd5c65cc1f4ca3800f0483"
D2_TERMINAL_SHA256 = (
    "461ea699ada0d6873422c537e63f5fcff3bca56a436caae9aeff4bb74761ca24"
)
D2_TERMINAL_RESULT_HASH = (
    "b8134ab47a1c69916593d1092b9125e0a8a78da11cf3080660064b12a2e6387c"
)
D2_TERMINAL_ACTION = (
    "REJECT_PSIM_D2_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES"
)

TRANSPORT_PROBE_PATH = Path(
    "results/protocol_specification_intent_maturity_d3_transport_probe_"
    "2026-07-25.json"
)
TRANSPORT_PROBE_COMMIT = DECISION_COMMIT
TRANSPORT_PROBE_SHA256 = (
    "4a815145a1f2ab9c6c61d599cf0aaf2218172e9f71251e95ce7178c1f3be13b7"
)
TRANSPORT_PROBE_RESULT_HASH = (
    "0df158cddd9b663b2daca14e01bcaa5c2e64b7f5d976720282120585bc41c63a"
)
TRANSPORT_PROBE_PROTOCOL_VERSION = (
    "psim_d3_batch_hydration_transport_probe_v1"
)

POLICY_ID = "PSIM-D3"
PROTOCOL_VERSION = "psim_d3_source_preregistration_v1"
SOURCE_ROOT = "/tmp/psim-d3-source"
FAILURE_ACTION = (
    "REJECT_PSIM_D3_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES"
)
MEMORIZATION_FAILURE_ACTION = (
    "REJECT_PSIM_D3_UNCHANGED_BEFORE_MARKET_OR_OUTCOMES"
)
SEALED_REF = "refs/psim-d3/sealed-tip"
GIT_VERSION = "git version 2.43.0"
GIT_BINARY_PATH = "/usr/bin/git"
GIT_BINARY_SHA256 = (
    "2a8c18fbf43da9f692d75474c72bea9dfd796c260b0f3dfe456376abc3bbd668"
)

ARTIFACT_PATHS = {
    "result": (
        "results/protocol_specification_intent_maturity_d3_source_support_"
        "2026-07-25.json"
    ),
    "rejection": (
        "results/protocol_specification_intent_maturity_d3_source_rejection_"
        "2026-07-25.json"
    ),
    "events": (
        "data/protocol_specification_intent_maturity_d3_events_"
        "2020_2023.jsonl.gz"
    ),
    "cards": (
        "data/protocol_specification_intent_maturity_d3_cards_"
        "2020_2024q1.jsonl.gz"
    ),
    "controls": (
        "results/protocol_specification_intent_maturity_d3_source_controls_"
        "2026-07-25.json"
    ),
}

FETCH_ARGUMENTS = (
    "-c",
    "fetch.negotiationAlgorithm=noop",
    "fetch",
    "origin",
    "--no-tags",
    "--no-write-fetch-head",
    "--recurse-submodules=no",
    "--filter=blob:none",
    "--no-auto-maintenance",
    "--stdin",
)

GIT_BINARY_BINDING = {
    "path": GIT_BINARY_PATH,
    "sha256": GIT_BINARY_SHA256,
    "version": GIT_VERSION,
    "exact_binary_required": True,
    "path_lookup_allowed": False,
    "synthetic_no_lazy_fetch_semantic_probe_required": True,
}

BATCH_HYDRATION_CONTRACT = {
    "gate_id": 4,
    "gate_name": "historical_blob_preamble_dependency_integrity",
    "oid_derivation_after_gate_id": 3,
    "oid_derivation": (
        "sorted unique non-null old_blob_oid/new_blob_oid union from the "
        "replica-local retained 2020-2023 ProposalGroup rows"
    ),
    "oid_manifest_encoding": (
        "lowercase full 40-hex SHA-1 one per LF-terminated line with final LF"
    ),
    "one_fetch_invocation_per_replica": True,
    "replica_count": 4,
    "command": [
        GIT_BINARY_PATH,
        "-C",
        "<fresh-bare-root>",
        *FETCH_ARGUMENTS,
    ],
    "stdin_is_complete_oid_manifest": True,
    "stdout_stderr_consumption": "subprocess_run_communicate",
    "timeout_seconds": 1800,
    "physical_pack_count_fixed": False,
    "multiple_new_packfiles_allowed": True,
    "at_least_one_new_promisor_pack_required": True,
    "initial_requested_oids_must_be_absent": True,
    "pre_and_post_inventory": [
        "complete local object roster and types",
        "pack roster",
        "promisor marker roster",
        "loose-object roster",
        "ref roster",
        "FETCH_HEAD presence",
    ],
    "complete_new_object_set_must_equal_requested_oids": True,
    "all_new_object_types_must_be_blob": True,
    "every_new_pack_requires_matching_promisor_marker": True,
    "new_loose_objects_allowed": False,
    "ref_roster_must_be_unchanged": True,
    "fetch_head_must_remain_absent": True,
    "maintenance_child_processes_allowed": 0,
    "trace_child_argv_ambiguity_action": FAILURE_ACTION,
    "post_hydration_read": {
        "environment": "GIT_NO_LAZY_FETCH=1",
        "cat_file_transport_role": "local_decode_only",
        "fetch_child_processes_allowed": 0,
        "object_store_ref_and_fetch_head_snapshot_must_be_unchanged": True,
        "missing_object_action": FAILURE_ACTION,
    },
    "forbidden_transports": [
        "interactive or buffered cat-file lazy hydration",
        "per-object fetch",
        "retry",
        "fallback lazy fetch",
        "full clone",
        "git fetch --refetch",
        "checkout",
        "D1 or D2 source-object reuse",
    ],
    "first_failure_action": FAILURE_ACTION,
    "synthetic_probe_binding": {
        "path": TRANSPORT_PROBE_PATH.as_posix(),
        "commit": TRANSPORT_PROBE_COMMIT,
        "sha256": TRANSPORT_PROBE_SHA256,
        "result_hash": TRANSPORT_PROBE_RESULT_HASH,
        "protocol_version": TRANSPORT_PROBE_PROTOCOL_VERSION,
        "official_source_opened": False,
        "market_model_outcomes_opened": False,
        "single_fetch_observed_pack_count": 1,
        "buffered_cat_file_control_pack_count": 6,
        "buffered_cat_file_control_requested_oids": 6,
        "probe_may_change_only_transport_contract": True,
    },
}

AUTHORIZED_DELTA_PATHS = (
    "candidate.id",
    "candidate.name",
    "candidate.selection_commit",
    "decision_binding.commit",
    "decision_binding.path",
    "decision_binding.sha256",
    "memorization_contract.first_failure_action",
    "next_authorized_step",
    "protocol_version",
    "source_contract.artifact_paths.cards",
    "source_contract.artifact_paths.controls",
    "source_contract.artifact_paths.events",
    "source_contract.artifact_paths.rejection",
    "source_contract.artifact_paths.result",
    "source_contract.bare_repository_contract.ref_roster[1]",
    "source_contract.bare_repository_contract.sealed_ref",
    "source_contract.bare_repository_contract.source_traversal_ref",
    "source_contract.batch_hydration_contract",
    "source_contract.git_binary_binding",
    "source_contract.repositories[0].sealed_ref",
    "source_contract.repositories[1].sealed_ref",
    "source_contract.source_root",
    "source_support_contract.control_sensitivity_metric.first_failure_action",
    "source_support_contract.first_failure_action",
)
AUTHORIZED_DELTA_HASH = (
    "a092091bc5f9316a90c828b2701526697a5ff29a3ca1ac82580acc30eada3b9e"
)
BATCH_HYDRATION_CONTRACT_HASH = (
    "6701b544f055c5eaa5e1c22dc4963f975514b9e5833845ee92c8384bdec9cf39"
)
GIT_BINARY_BINDING_HASH = (
    "70aa4a393c76b2d310f4cc91367533a47a93537fa06ccaa2dcb5dc6100397ebf"
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
    return sha256_bytes(canonical_json_bytes(payload, pretty=False).rstrip(b"\n"))


def _read_canonical_json(path: str | Path) -> dict[str, Any]:
    target = repository_path(path)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"PSIM-D3 authority is absent or unsafe: {path}")
    raw = target.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"PSIM-D3 authority is unreadable: {path}") from error
    if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
        raise RuntimeError(f"PSIM-D3 authority is noncanonical: {path}")
    return payload


def _validate_authority() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    if re.fullmatch(r"[0-9a-f]{40}", DECISION_COMMIT) is None:
        raise RuntimeError("PSIM-D3 decision commit is malformed")
    if sha256_file(DECISION_PATH) != DECISION_SHA256:
        raise RuntimeError("PSIM-D3 decision document hash changed")

    d2_registration = _read_canonical_json(D2_PREREGISTRATION_PATH)
    if (
        sha256_file(D2_PREREGISTRATION_PATH)
        != D2_PREREGISTRATION_SHA256
        or d2_registration.get("manifest_hash")
        != D2_PREREGISTRATION_MANIFEST_HASH
        or d2_registration != d2.build_preregistration()
    ):
        raise RuntimeError("PSIM-D2 preregistration authority changed")

    d2_terminal = _read_canonical_json(D2_TERMINAL_PATH)
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
    ledger = d2_terminal.get("access_ledger", {})
    if (
        sha256_file(D2_TERMINAL_PATH) != D2_TERMINAL_SHA256
        or d2_terminal.get("result_hash") != D2_TERMINAL_RESULT_HASH
        or d2_terminal.get("decision") != "reject"
        or d2_terminal.get("first_failure", {}).get("gate_id") != 4
        or d2_terminal.get("first_failure", {}).get("name")
        != "historical_blob_preamble_dependency_integrity"
        or d2_terminal.get("source_incidence_opened") is not True
        or d2_terminal.get("outcomes_opened") is not False
        or d2_terminal.get("profitability_result") is not False
        or d2_terminal.get("terminal_action") != D2_TERMINAL_ACTION
        or not isinstance(ledger, dict)
        or any(ledger.get(key) != 0 for key in forbidden_terminal_counters)
        or ledger.get("proposal_text_rows_opened") != 0
    ):
        raise RuntimeError("PSIM-D2 terminal authority changed")

    probe = _read_canonical_json(TRANSPORT_PROBE_PATH)
    access = probe.get("access_boundary", {})
    bulk = probe.get("bulk_fetch_probe", {})
    buffered = probe.get("buffered_cat_file_control", {})
    if (
        sha256_file(TRANSPORT_PROBE_PATH) != TRANSPORT_PROBE_SHA256
        or probe.get("result_hash") != TRANSPORT_PROBE_RESULT_HASH
        or probe.get("protocol_version")
        != TRANSPORT_PROBE_PROTOCOL_VERSION
        or probe.get("synthetic_only") is not True
        or set(access.values()) != {False}
        or probe.get("git_binding")
        != {
            "binary_path": GIT_BINARY_PATH,
            "binary_sha256": GIT_BINARY_SHA256,
            "version": GIT_VERSION,
        }
        or bulk.get("fetch_invocations") != 1
        or bulk.get("requested_blob_count") != 6
        or bulk.get("new_total_object_store_exact_requested_blobs")
        is not True
        or bulk.get("nonrequested_blob_present") is not False
        or bulk.get("maintenance_child_processes") != 0
        or bulk.get("post_hydration_fetch_child_processes") != 0
        or bulk.get("post_hydration_object_store_unchanged") is not True
        or buffered.get("requested_blob_count") != 6
        or buffered.get("promisor_pack_delta") != 6
    ):
        raise RuntimeError("PSIM-D3 transport probe authority changed")
    return d2_registration, d2_terminal, probe


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


def _successor_core(d2_registration: dict[str, Any]) -> dict[str, Any]:
    core = _contract_core(d2_registration)
    core["protocol_version"] = PROTOCOL_VERSION
    core["candidate"] = {
        **core["candidate"],
        "id": POLICY_ID,
        "name": (
            "Protocol Specification Intent-Maturity relation RLLM, "
            "targeted batch-hydration bare replay"
        ),
        "selection_commit": DECISION_COMMIT,
    }
    core["decision_binding"] = {
        "path": DECISION_PATH.as_posix(),
        "commit": DECISION_COMMIT,
        "sha256": DECISION_SHA256,
    }

    source = core["source_contract"]
    source["source_root"] = SOURCE_ROOT
    source["artifact_paths"] = dict(ARTIFACT_PATHS)
    source["git_binary_binding"] = copy.deepcopy(GIT_BINARY_BINDING)
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
        "implement and seal synthetic-only PSIM-D3 targeted batch-hydration "
        "source-support evaluator"
    )
    return core


def build_preregistration() -> dict[str, Any]:
    d2_registration, d2_terminal, probe = _validate_authority()
    d2_core = _contract_core(d2_registration)
    successor = _successor_core(d2_registration)
    delta = _diff_values(d2_core, successor)
    if tuple(sorted(delta)) != tuple(sorted(AUTHORIZED_DELTA_PATHS)):
        raise RuntimeError(
            "PSIM-D3 inherited-contract delta changed: "
            + ",".join(sorted(delta))
        )
    if (
        canonical_hash(delta) != AUTHORIZED_DELTA_HASH
        or canonical_hash(BATCH_HYDRATION_CONTRACT)
        != BATCH_HYDRATION_CONTRACT_HASH
        or canonical_hash(GIT_BINARY_BINDING)
        != GIT_BINARY_BINDING_HASH
    ):
        raise RuntimeError("PSIM-D3 authorized transport delta hash changed")
    if successor["source_support_contract"]["gates_in_order"] != list(
        d1.SOURCE_ONLY_GATES
    ):
        raise RuntimeError("PSIM-D3 source gate roster changed")
    if successor["source_support_contract"]["relation_controls"] != list(
        d1.RELATION_CONTROLS
    ):
        raise RuntimeError("PSIM-D3 control roster changed")
    for key in (
        "availability_contract",
        "boundary_reset_contract",
        "bucket_contract",
        "daily_relation_contract",
        "event_contract",
        "excluded_feasibility_probe",
        "forbidden_access_contract",
        "official_sources",
        "parser_contract",
        "representation_contract",
        "split_contract",
    ):
        if successor[key] != d2_core[key]:
            raise RuntimeError(f"PSIM-D3 inherited {key} changed")

    inheritance = {
        "d2_preregistration": {
            "path": D2_PREREGISTRATION_PATH.as_posix(),
            "commit": D2_PREREGISTRATION_COMMIT,
            "sha256": D2_PREREGISTRATION_SHA256,
            "manifest_hash": D2_PREREGISTRATION_MANIFEST_HASH,
            "contract_core_hash": canonical_hash(d2_core),
        },
        "d2_terminal_rejection": {
            "path": D2_TERMINAL_PATH.as_posix(),
            "commit": D2_TERMINAL_COMMIT,
            "sha256": D2_TERMINAL_SHA256,
            "result_hash": D2_TERMINAL_RESULT_HASH,
            "first_failure_gate_id": 4,
            "source_incidence_opened": d2_terminal[
                "source_incidence_opened"
            ],
            "outcomes_opened": d2_terminal["outcomes_opened"],
        },
        "transport_probe": {
            "path": TRANSPORT_PROBE_PATH.as_posix(),
            "commit": TRANSPORT_PROBE_COMMIT,
            "sha256": TRANSPORT_PROBE_SHA256,
            "result_hash": TRANSPORT_PROBE_RESULT_HASH,
            "protocol_version": TRANSPORT_PROBE_PROTOCOL_VERSION,
            "synthetic_only": probe["synthetic_only"],
        },
        "authorized_delta_paths": list(AUTHORIZED_DELTA_PATHS),
        "authorized_delta": delta,
        "authorized_delta_hash": AUTHORIZED_DELTA_HASH,
        "batch_hydration_contract_hash": (
            BATCH_HYDRATION_CONTRACT_HASH
        ),
        "git_binary_binding_hash": GIT_BINARY_BINDING_HASH,
        "all_other_contract_paths_byte_equal": True,
        "official_git_documentation": [
            "https://git-scm.com/docs/partial-clone/2.43.0.html",
            "https://git-scm.com/docs/git-fetch/2.43.0.html",
            "https://git-scm.com/docs/git-cat-file/2.43.0.html",
            (
                "https://github.com/git/git/blob/v2.43.0/"
                "promisor-remote.c#L17-L45"
            ),
        ],
        "preregistration_access": {
            "git_commands": 0,
            "network_calls": 0,
            "official_source_opened": False,
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
                f"existing PSIM-D3 preregistration differs: {destination}"
            )
        return destination
    temporary = destination.with_name(destination.name + ".tmp")
    if os.path.lexists(temporary):
        raise RuntimeError(
            f"unsafe PSIM-D3 preregistration temporary: {temporary}"
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
