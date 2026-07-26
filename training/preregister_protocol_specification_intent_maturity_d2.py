"""Preregister the outcome-blind PSIM-D2 bare source-support candidate."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from training import preregister_protocol_specification_intent_maturity as d1


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(
    "training/preregister_protocol_specification_intent_maturity_d2.py"
)
DEFAULT_OUTPUT = Path(
    "results/protocol_specification_intent_maturity_d2_preregistration_"
    "2026-07-25.json"
)
DECISION_PATH = Path(
    "docs/post-psim-d1-alpha-mechanism-audit-2026-07-25.md"
)
DECISION_COMMIT = "73de336a1d24399927d43e08c8394450b1cd1cb0"
DECISION_SHA256 = (
    "e68c0217a6aa3927c88c1f48d9c45ed0b2be3cee4bc3c86d3cb4c6a88e1f8598"
)

D1_PREREGISTRATION_PATH = d1.DEFAULT_OUTPUT
D1_PREREGISTRATION_COMMIT = (
    "2125584b732e81ba34d3f81534b8b1279a379e74"
)
D1_PREREGISTRATION_SHA256 = (
    "bd4053574fe6285c34356baaa080e215f08bbf8142e9c0c968bffbdccb2dc736"
)
D1_PREREGISTRATION_MANIFEST_HASH = (
    "bdf49fb396779599eb329a407685435c05217f132ea856f9bb743914b5afbe81"
)
D1_TERMINAL_PATH = Path(
    "results/protocol_specification_intent_maturity_source_rejection_"
    "2026-07-25.json"
)
D1_TERMINAL_COMMIT = "2a7e4d72d56ff29e90075b3fb872c58c8dd5e310"
D1_TERMINAL_SHA256 = (
    "9b0b2354c6edbcfe627527bf4370a4eb0c1e6c1bcb76843f843d9028b16e6494"
)
D1_TERMINAL_RESULT_HASH = (
    "5815f7473410c7d75aabea8b6a97cfb7f963b1c6d29f8efa22f0a0a64d33655d"
)

POLICY_ID = "PSIM-D2"
PROTOCOL_VERSION = "psim_d2_source_preregistration_v1"
SOURCE_ROOT = "/tmp/psim-d2-source"
FAILURE_ACTION = (
    "REJECT_PSIM_D2_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES"
)
MEMORIZATION_FAILURE_ACTION = (
    "REJECT_PSIM_D2_UNCHANGED_BEFORE_MARKET_OR_OUTCOMES"
)
SEALED_REF = "refs/psim-d2/sealed-tip"
GIT_VERSION = "git version 2.43.0"

ARTIFACT_PATHS = {
    "result": (
        "results/protocol_specification_intent_maturity_d2_source_support_"
        "2026-07-25.json"
    ),
    "rejection": (
        "results/protocol_specification_intent_maturity_d2_source_rejection_"
        "2026-07-25.json"
    ),
    "events": (
        "data/protocol_specification_intent_maturity_d2_events_"
        "2020_2023.jsonl.gz"
    ),
    "cards": (
        "data/protocol_specification_intent_maturity_d2_cards_"
        "2020_2024q1.jsonl.gz"
    ),
    "controls": (
        "results/protocol_specification_intent_maturity_d2_source_controls_"
        "2026-07-25.json"
    ),
}

CLONE_ARGUMENTS = (
    "--bare",
    "--filter=blob:none",
    "--single-branch",
    "--branch",
    "master",
    "--no-tags",
)

BARE_REPOSITORY_CONTRACT = {
    "fresh_independent_roots_required": True,
    "root_names": {
        "ethereum_a": "ethereum-a.git",
        "ethereum_b": "ethereum-b.git",
        "bitcoin_a": "bitcoin-a.git",
        "bitcoin_b": "bitcoin-b.git",
    },
    "is_bare_repository": True,
    "is_inside_work_tree": False,
    "absolute_git_dir_must_equal_configured_root": True,
    "git_common_dir": ".",
    "symbolic_head": "refs/heads/master",
    "sealed_ref": SEALED_REF,
    "ref_roster": ["refs/heads/master", SEALED_REF],
    "forbidden_paths": [
        ".git",
        "index",
        "worktrees",
        "commondir",
        "gitdir",
        "objects/info/alternates",
        "shallow",
    ],
    "git_status_allowed": False,
    "checkout_allowed": False,
    "shared_objects_or_cache_allowed": False,
    "git_fsck_no_dangling_required": True,
    "source_traversal_ref": SEALED_REF,
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
    "source_contract.artifact_paths",
    "source_contract.bare_repository_contract",
    "source_contract.clone_arguments",
    "source_contract.git_status_allowed",
    "source_contract.git_version",
    "source_contract.repositories[0].local_branch_ref",
    "source_contract.repositories[0].remote_head_symref",
    "source_contract.repositories[0].sealed_ref",
    "source_contract.repositories[1].local_branch_ref",
    "source_contract.repositories[1].remote_head_symref",
    "source_contract.repositories[1].sealed_ref",
    "source_contract.repository_representation",
    "source_contract.shared_git_environment_scrubbed",
    "source_contract.source_root",
    "source_support_contract.control_sensitivity_metric.first_failure_action",
    "source_support_contract.first_failure_action",
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
        raise RuntimeError(f"PSIM-D2 authority is absent or unsafe: {path}")
    raw = target.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"PSIM-D2 authority is unreadable: {path}") from error
    if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
        raise RuntimeError(f"PSIM-D2 authority is noncanonical: {path}")
    return payload


def _validate_authority() -> tuple[dict[str, Any], dict[str, Any]]:
    if re.fullmatch(r"[0-9a-f]{40}", DECISION_COMMIT) is None:
        raise RuntimeError("PSIM-D2 decision commit is malformed")
    if sha256_file(DECISION_PATH) != DECISION_SHA256:
        raise RuntimeError("PSIM-D2 decision document hash changed")

    d1_registration = _read_canonical_json(D1_PREREGISTRATION_PATH)
    if (
        sha256_file(D1_PREREGISTRATION_PATH) != D1_PREREGISTRATION_SHA256
        or d1_registration.get("manifest_hash")
        != D1_PREREGISTRATION_MANIFEST_HASH
        or d1_registration != d1.build_preregistration()
    ):
        raise RuntimeError("PSIM-D1 preregistration authority changed")

    d1_terminal = _read_canonical_json(D1_TERMINAL_PATH)
    if (
        sha256_file(D1_TERMINAL_PATH) != D1_TERMINAL_SHA256
        or d1_terminal.get("result_hash") != D1_TERMINAL_RESULT_HASH
        or d1_terminal.get("decision") != "reject"
        or d1_terminal.get("first_failure", {}).get("gate_id") != 1
        or d1_terminal.get("source_incidence_opened") is not False
    ):
        raise RuntimeError("PSIM-D1 terminal authority changed")
    return d1_registration, d1_terminal


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


def _successor_core(d1_registration: dict[str, Any]) -> dict[str, Any]:
    core = copy.deepcopy(d1_registration)
    core.pop("manifest_hash")
    core["protocol_version"] = PROTOCOL_VERSION
    core["candidate"] = {
        **core["candidate"],
        "id": POLICY_ID,
        "name": (
            "Protocol Specification Intent-Maturity relation RLLM, "
            "bare object-database replay"
        ),
        "selection_commit": DECISION_COMMIT,
    }
    core["decision_binding"] = {
        "path": DECISION_PATH.as_posix(),
        "commit": DECISION_COMMIT,
        "sha256": DECISION_SHA256,
    }

    source = core["source_contract"]
    source["clone_arguments"] = list(CLONE_ARGUMENTS)
    source["source_root"] = SOURCE_ROOT
    source["artifact_paths"] = dict(ARTIFACT_PATHS)
    source["repository_representation"] = (
        "BARE_OBJECT_DATABASE_NO_WORKTREE_NO_INDEX"
    )
    source["git_status_allowed"] = False
    source["shared_git_environment_scrubbed"] = True
    source["git_version"] = GIT_VERSION
    source["bare_repository_contract"] = copy.deepcopy(
        BARE_REPOSITORY_CONTRACT
    )
    for repository in source["repositories"]:
        repository["remote_head_symref"] = "refs/heads/master"
        repository["local_branch_ref"] = "refs/heads/master"
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
        "implement and seal synthetic-only PSIM-D2 bare source-support "
        "evaluator"
    )
    return core


def build_preregistration() -> dict[str, Any]:
    d1_registration, d1_terminal = _validate_authority()
    d1_core = copy.deepcopy(d1_registration)
    d1_core.pop("manifest_hash")
    successor = _successor_core(d1_registration)
    delta = _diff_values(d1_core, successor)
    if tuple(sorted(delta)) != tuple(sorted(AUTHORIZED_DELTA_PATHS)):
        raise RuntimeError(
            "PSIM-D2 inherited-contract delta changed: "
            + ",".join(sorted(delta))
        )
    if successor["source_support_contract"]["gates_in_order"] != list(
        d1.SOURCE_ONLY_GATES
    ):
        raise RuntimeError("PSIM-D2 source gate roster changed")
    if successor["source_support_contract"]["relation_controls"] != list(
        d1.RELATION_CONTROLS
    ):
        raise RuntimeError("PSIM-D2 control roster changed")
    if successor["split_contract"] != d1_core["split_contract"]:
        raise RuntimeError("PSIM-D2 split contract changed")
    if successor["parser_contract"] != d1_core["parser_contract"]:
        raise RuntimeError("PSIM-D2 parser contract changed")

    inheritance = {
        "d1_preregistration": {
            "path": D1_PREREGISTRATION_PATH.as_posix(),
            "commit": D1_PREREGISTRATION_COMMIT,
            "sha256": D1_PREREGISTRATION_SHA256,
            "manifest_hash": D1_PREREGISTRATION_MANIFEST_HASH,
            "core_hash": canonical_hash(d1_core),
        },
        "d1_terminal_rejection": {
            "path": D1_TERMINAL_PATH.as_posix(),
            "commit": D1_TERMINAL_COMMIT,
            "sha256": D1_TERMINAL_SHA256,
            "result_hash": D1_TERMINAL_RESULT_HASH,
            "source_incidence_opened": d1_terminal[
                "source_incidence_opened"
            ],
        },
        "authorized_delta_paths": list(AUTHORIZED_DELTA_PATHS),
        "authorized_delta": delta,
        "authorized_delta_hash": canonical_hash(delta),
        "all_other_paths_byte_equal": True,
        "official_git_documentation": [
            "https://git-scm.com/docs/git-clone",
            "https://git-scm.com/docs/git-rev-parse",
            "https://git-scm.com/docs/git-fsck",
        ],
        "local_bare_probe": {
            "git_version": GIT_VERSION,
            "official_source_opened": False,
            "is_bare_repository": True,
            "is_inside_work_tree": False,
            "git_common_dir": ".",
            "symbolic_head": "refs/heads/master",
            "index_present": False,
            "dot_git_present": False,
            "alternates_present": False,
            "linked_worktrees_present": False,
            "fsck_no_dangling_passed": True,
            "probe_may_change_inherited_contract": False,
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
                f"existing PSIM-D2 preregistration differs: {destination}"
            )
        return destination
    temporary = destination.with_name(destination.name + ".tmp")
    if os.path.lexists(temporary):
        raise RuntimeError(
            f"unsafe PSIM-D2 preregistration temporary: {temporary}"
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
