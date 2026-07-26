"""Build, gate, seal, and publish the outcome-blind PSIM-D2 source.

PSIM-D2 reuses the exact sealed PSIM-D1 causal parser and relation-card core.
Its only source-semantic delta is Gate 1: official repositories are acquired
as fresh independent bare object databases with no worktree or index.

``self-check`` is synthetic-only. ``create-seal`` binds committed code and
tests. ``run`` validates the exact direct-child seal before the one permitted
official source attempt. Market, model, funding, reward, trade, and portfolio
data are forbidden throughout this module.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

from training import (
    build_protocol_specification_intent_maturity_source_support as core,
)
from training import (
    preregister_protocol_specification_intent_maturity_d2 as prereg,
)


POLICY_ID = "PSIM-D2"
RUNNER_PROTOCOL = "psim_d2_source_support_runner_v1"
SEAL_PROTOCOL = "psim_d2_source_support_execution_seal_v1"
RESULT_PROTOCOL = "psim_d2_source_support_result_v1"
SELF_CHECK_PROTOCOL = "psim_d2_synthetic_self_check_v1"
CONTROL_REPORT_PROTOCOL = "psim_d2_source_controls_v1"
PASS_ACTION = "ACCEPT_PSIM_D2_SOURCE_SUPPORT_ONLY_NO_PROFITABILITY_CLAIM"
FAILURE_ACTION = prereg.FAILURE_ACTION
GATE_NAMES = tuple(core.GATE_NAMES)
SEALED_REF = prereg.SEALED_REF

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(
    "training/build_protocol_specification_intent_maturity_d2_source_support.py"
)
TEST_PATH = Path(
    "tests/test_build_protocol_specification_intent_maturity_d2_source_support.py"
)
SEAL_TEST_PATH = Path(
    "tests/test_psim_d2_source_support_execution_seal.py"
)
IMPLEMENTATION_CONTRACT_PATH = Path(
    "docs/psim-d2-source-support-implementation-contract-2026-07-25.md"
)
PREREGISTRATION_SCRIPT_PATH = prereg.SCRIPT_PATH
PREREGISTRATION_PATH = prereg.DEFAULT_OUTPUT
PREREGISTRATION_DOC_PATH = Path(
    "docs/psim-d2-source-support-preregistration-2026-07-25.md"
)
DECISION_PATH = prereg.DECISION_PATH
D1_TERMINAL_PATH = prereg.D1_TERMINAL_PATH

PREREGISTRATION_COMMIT = "e853f7688a484b323c024115e3ef4af07e6a5896"
PREREGISTRATION_SHA256 = (
    "3b405de2bcdc1979855e8505148f7de3fbee366cb126e78b1b23e10f84cf470a"
)
PREREGISTRATION_MANIFEST_HASH = (
    "917d2f318b268b01621c9e969237d76fc82d7e6aff408269842e660cc155d915"
)
PREREGISTRATION_SCRIPT_SHA256 = (
    "e69b9d0a44f4ffe39657fb46eb9b92b0fe40b1e3041f289fa15bf40e83c2e679"
)
PREREGISTRATION_DOC_SHA256 = (
    "b231f4fee51eeb7958a656ca36e7f08360ad950fab6e33e1fd0c652ae4d20b2c"
)
DECISION_COMMIT = prereg.DECISION_COMMIT
DECISION_SHA256 = prereg.DECISION_SHA256

D1_CORE_RUNNER_PATH = core.RUNNER_PATH
D1_CORE_TEST_PATH = core.TEST_PATH
D1_CORE_COMMIT = "80b656994f17548a7a599a548e23e9f1cd01302d"
D1_CORE_RUNNER_SHA256 = (
    "414e83256b3ea489a9e1cd0995f6061e5fab550cd12c795ef7e88eff8998d9fb"
)
D1_CORE_TEST_SHA256 = (
    "343aa1a72cfbca23d9756988ced042b5c61a6e8fc5a21a0b6d18e45870e906e9"
)
D1_CORE_SELF_CHECK_MANIFEST_HASH = (
    "24ad04222852e97ffbd37067102cb52b2e38d5d992fd4641ab416b0670168a61"
)
D1_CORE_SELF_CHECK_STDOUT_SHA256 = (
    "4acc071bee5de333c804da59273d5d0ad1fcfc4e735e6f0ac78b5c1539e65a88"
)

EXECUTION_SEAL_PATH = Path(
    "results/psim_d2_source_support_execution_seal_2026-07-25.json"
)
RUN_LOCK_PATH = Path("results/.psim_d2_source_support_run.lock")
DEFAULT_RESULT_PATH = Path(prereg.ARTIFACT_PATHS["result"])
DEFAULT_REJECTION_PATH = Path(prereg.ARTIFACT_PATHS["rejection"])
DEFAULT_EVENTS_PATH = Path(prereg.ARTIFACT_PATHS["events"])
DEFAULT_CARDS_PATH = Path(prereg.ARTIFACT_PATHS["cards"])
DEFAULT_CONTROLS_PATH = Path(prereg.ARTIFACT_PATHS["controls"])
DEFAULT_SOURCE_ROOT = Path(prereg.SOURCE_ROOT)

HEX40 = core.HEX40
HEX64 = core.HEX64
ZERO_OID = core.ZERO_OID
SOURCE_START = core.SOURCE_START
SOURCE_END_EXCLUSIVE = core.SOURCE_END_EXCLUSIVE
CARD_END_EXCLUSIVE = core.CARD_END_EXCLUSIVE
DISK_LIMIT_GIB = core.prereg.DISK_LIMIT_GIB
FORBIDDEN_ACCESS_FIELDS = core.FORBIDDEN_ACCESS_FIELDS

AccessLedger = core.AccessLedger
BlobFeatures = core.BlobFeatures
CommitRecord = core.CommitRecord
DailyCard = core.DailyCard
GateResult = core.GateResult
PathChange = core.PathChange
ProposalEvent = core.ProposalEvent
ProposalGroup = core.ProposalGroup

canonical_hash = core.canonical_hash
canonical_json_bytes = core.canonical_json_bytes
card_row = core.card_row
deterministic_gzip = core.deterministic_gzip
event_row = core.event_row
format_time = core.format_time
gate_commit_chains = core.gate_commit_chains
gate_control_sensitivity = core.gate_control_sensitivity
gate_daily_cards = core.gate_daily_cards
gate_event_parser_replay = core.gate_event_parser_replay
gate_forbidden_access = core.gate_forbidden_access
gate_future_append = core.gate_future_append
gate_independent_replay = core.gate_independent_replay
gate_pairing_reset_quarantine = core.gate_pairing_reset_quarantine
gate_proposal_groups = core.gate_proposal_groups
gate_split_support = core.gate_split_support
gate_vocabulary = core.gate_vocabulary
jsonl_bytes = core.jsonl_bytes
parse_bip_preamble = core.prereg.parse_bip_preamble
parse_blob_features = core.parse_blob_features
parse_commit_object = core.parse_commit_object
parse_dependency_ids = core.prereg.parse_dependency_ids
parse_eip_preamble = core.prereg.parse_eip_preamble
parse_raw_path_delta = core.parse_raw_path_delta
proposal_groups_for_commit = core.proposal_groups_for_commit
rows_fingerprint = core.rows_fingerprint
sha256_bytes = core.sha256_bytes
split_support_metrics = core.split_support_metrics
synthetic_events = core.synthetic_events
transform_events = core.transform_events


@dataclass(frozen=True)
class Config:
    source_root: Path = DEFAULT_SOURCE_ROOT
    result_path: Path = DEFAULT_RESULT_PATH
    rejection_path: Path = DEFAULT_REJECTION_PATH
    events_path: Path = DEFAULT_EVENTS_PATH
    cards_path: Path = DEFAULT_CARDS_PATH
    controls_path: Path = DEFAULT_CONTROLS_PATH
    network_timeout_seconds: int = 900


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(repository_path(path).read_bytes())


def _safe_destination(path: str | Path) -> Path:
    destination = Path(os.path.abspath(repository_path(path)))
    root = REPO_ROOT.resolve()
    try:
        relative = destination.relative_to(root)
    except ValueError as error:
        raise RuntimeError(
            f"PSIM-D2 output escapes repository: {destination}"
        ) from error
    if not relative.parts:
        raise RuntimeError(f"PSIM-D2 output is repository root: {destination}")
    cursor = root
    for component in relative.parts[:-1]:
        cursor /= component
        if os.path.lexists(cursor) and cursor.is_symlink():
            raise RuntimeError(
                f"PSIM-D2 output parent is symlinked: {cursor}"
            )
    cursor.mkdir(parents=True, exist_ok=True)
    return destination


def _write_once_bytes(path: str | Path, raw: bytes) -> Path:
    destination = _safe_destination(path)
    if os.path.lexists(destination):
        if (
            destination.is_symlink()
            or not destination.is_file()
            or destination.read_bytes() != raw
        ):
            raise RuntimeError(
                f"existing PSIM-D2 artifact differs: {destination}"
            )
        return destination
    temporary = destination.with_name(destination.name + ".tmp")
    if os.path.lexists(temporary):
        if temporary.is_dir() or temporary.is_symlink():
            raise RuntimeError(
                f"unsafe PSIM-D2 temporary path: {temporary}"
            )
        temporary.unlink()
    temporary.write_bytes(raw)
    os.replace(temporary, destination)
    return destination


def _git_environment() -> dict[str, str]:
    return core._git_environment()


def _run_git(
    arguments: Sequence[str],
    *,
    ledger: AccessLedger,
    cwd: Path | None = None,
    network: bool = False,
    timeout: int = 900,
    check: bool = True,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    ledger.git_commands += 1
    if network:
        ledger.network_commands += 1
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        env=_git_environment(),
        input=input_bytes,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"PSIM-D2 git command failed: {detail}")
    return completed


def _git_text(
    arguments: Sequence[str],
    *,
    ledger: AccessLedger,
    cwd: Path | None = None,
    network: bool = False,
    timeout: int = 900,
) -> str:
    return _run_git(
        arguments,
        ledger=ledger,
        cwd=cwd,
        network=network,
        timeout=timeout,
    ).stdout.decode("utf-8", errors="strict").strip()


def _disk_used_gib(path: Path) -> int:
    usage = shutil.disk_usage(path if path.exists() else path.parent)
    return usage.used // (1024**3)


def enforce_disk_guard(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    used = _disk_used_gib(path)
    if used > DISK_LIMIT_GIB:
        raise RuntimeError(
            f"PSIM-D2 disk guard exceeded: {used} GiB > "
            f"{DISK_LIMIT_GIB} GiB"
        )
    return used


def _repository_spec(protocol: str) -> Any:
    matches = [
        row for row in core.prereg.REPOSITORIES if row.protocol == protocol
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"PSIM-D2 repository spec is not unique: {protocol}"
        )
    return matches[0]


def clone_path(config: Config, protocol: str, replica: str) -> Path:
    if protocol not in {"ethereum", "bitcoin"}:
        raise ValueError("PSIM-D2 protocol must be ethereum or bitcoin")
    if replica not in {"a", "b"}:
        raise ValueError("PSIM-D2 replica must be a or b")
    return config.source_root / f"{protocol}-{replica}.git"


def _validate_object_store(repo: Path) -> dict[str, int]:
    objects = repo / "objects"
    if objects.is_symlink() or not objects.is_dir():
        raise RuntimeError("PSIM-D2 bare object directory is unsafe")
    regular_files = 0
    for root, directories, files in os.walk(objects, followlinks=False):
        root_path = Path(root)
        for name in [*directories, *files]:
            candidate = root_path / name
            if candidate.is_symlink():
                raise RuntimeError(
                    "PSIM-D2 object store contains a symlink"
                )
        for name in files:
            candidate = root_path / name
            if not candidate.is_file():
                raise RuntimeError(
                    "PSIM-D2 object store contains a non-regular file"
                )
            stat = candidate.stat(follow_symlinks=False)
            regular_files += 1
            if stat.st_nlink != 1:
                raise RuntimeError(
                    "PSIM-D2 object store contains a shared hard link"
                )
    if regular_files <= 0:
        raise RuntimeError("PSIM-D2 object store has no regular files")
    return {
        "regular_files": regular_files,
        "symlinks": 0,
        "multiple_link_files": 0,
    }


def _assert_forbidden_bare_paths_absent(repo: Path) -> None:
    for relative in prereg.BARE_REPOSITORY_CONTRACT["forbidden_paths"]:
        candidate = repo / relative
        if os.path.lexists(candidate):
            raise RuntimeError(
                f"PSIM-D2 forbidden bare path exists: {relative}"
            )


def prepare_source_repository(
    config: Config,
    protocol: str,
    replica: str,
    ledger: AccessLedger,
) -> dict[str, Any]:
    spec = _repository_spec(protocol)
    destination = clone_path(config, protocol, replica)
    enforce_disk_guard(destination)
    if os.path.lexists(destination):
        raise RuntimeError(
            f"PSIM-D2 fresh clone root already exists: {destination}"
        )
    _validate_no_symlink_ancestors(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _validate_no_symlink_ancestors(destination.parent)
    _run_git(
        [
            "clone",
            *prereg.CLONE_ARGUMENTS,
            spec.remote,
            str(destination),
        ],
        ledger=ledger,
        network=True,
        timeout=config.network_timeout_seconds,
    )
    _validate_no_symlink_ancestors(destination)
    remote = _git_text(
        ["-C", str(destination), "remote", "get-url", "origin"],
        ledger=ledger,
    )
    if remote != spec.remote:
        raise RuntimeError(f"PSIM-D2 source remote changed for {protocol}")

    remote_head = _git_text(
        ["ls-remote", "--symref", spec.remote, "HEAD"],
        ledger=ledger,
        network=True,
        timeout=config.network_timeout_seconds,
    )
    remote_head_lines = remote_head.splitlines()
    expected_head = f"ref: refs/heads/{spec.branch}\tHEAD"
    if (
        len(remote_head_lines) != 2
        or remote_head_lines[0] != expected_head
        or "\t" not in remote_head_lines[1]
    ):
        raise RuntimeError(
            f"PSIM-D2 remote HEAD symref changed for {protocol}"
        )
    remote_head_oid, remote_head_name = remote_head_lines[1].split("\t", 1)
    if (
        HEX40.fullmatch(remote_head_oid) is None
        or remote_head_name != "HEAD"
    ):
        raise RuntimeError(
            f"PSIM-D2 remote HEAD object changed for {protocol}"
        )

    _run_git(
        [
            "-C",
            str(destination),
            "fetch",
            "--no-tags",
            "--filter=blob:none",
            "origin",
            spec.sealed_tip,
        ],
        ledger=ledger,
        network=True,
        timeout=config.network_timeout_seconds,
    )
    _run_git(
        [
            "-C",
            str(destination),
            "update-ref",
            SEALED_REF,
            spec.sealed_tip,
            ZERO_OID,
        ],
        ledger=ledger,
    )

    is_bare = _git_text(
        ["-C", str(destination), "rev-parse", "--is-bare-repository"],
        ledger=ledger,
    )
    inside_worktree = _git_text(
        ["-C", str(destination), "rev-parse", "--is-inside-work-tree"],
        ledger=ledger,
    )
    absolute_git_dir = _git_text(
        ["-C", str(destination), "rev-parse", "--absolute-git-dir"],
        ledger=ledger,
    )
    git_common_dir = _git_text(
        ["-C", str(destination), "rev-parse", "--git-common-dir"],
        ledger=ledger,
    )
    symbolic_head = _git_text(
        ["-C", str(destination), "symbolic-ref", "HEAD"],
        ledger=ledger,
    )
    refs = sorted(
        line
        for line in _git_text(
            [
                "-C",
                str(destination),
                "for-each-ref",
                "--format=%(refname)",
            ],
            ledger=ledger,
        ).splitlines()
        if line
    )
    local_branch_oid = _git_text(
        [
            "-C",
            str(destination),
            "rev-parse",
            f"refs/heads/{spec.branch}^{{commit}}",
        ],
        ledger=ledger,
    )
    sealed_ref_oid = _git_text(
        [
            "-C",
            str(destination),
            "rev-parse",
            f"{SEALED_REF}^{{commit}}",
        ],
        ledger=ledger,
    )
    object_format = _git_text(
        ["-C", str(destination), "rev-parse", "--show-object-format"],
        ledger=ledger,
    )
    object_type = _git_text(
        ["-C", str(destination), "cat-file", "-t", SEALED_REF],
        ledger=ledger,
    )
    is_shallow = _git_text(
        ["-C", str(destination), "rev-parse", "--is-shallow-repository"],
        ledger=ledger,
    )
    _assert_forbidden_bare_paths_absent(destination)
    object_store = _validate_object_store(destination)
    _run_git(
        ["-C", str(destination), "fsck", "--no-dangling"],
        ledger=ledger,
        timeout=config.network_timeout_seconds,
    )
    used = enforce_disk_guard(destination)

    expected_root = Path(os.path.abspath(destination))
    if (
        is_bare != "true"
        or inside_worktree != "false"
        or Path(absolute_git_dir) != expected_root
        or git_common_dir != "."
        or symbolic_head != f"refs/heads/{spec.branch}"
        or refs
        != sorted([f"refs/heads/{spec.branch}", SEALED_REF])
        or HEX40.fullmatch(local_branch_oid) is None
        or sealed_ref_oid != spec.sealed_tip
        or object_format != spec.object_format
        or object_type != "commit"
        or is_shallow != "false"
    ):
        raise RuntimeError(
            f"PSIM-D2 bare repository shape changed for {protocol}"
        )

    return {
        "protocol": protocol,
        "replica": replica,
        "root_name": destination.name,
        "remote": remote,
        "remote_head_symref": f"refs/heads/{spec.branch}",
        "remote_head_oid": remote_head_oid,
        "local_branch_ref": f"refs/heads/{spec.branch}",
        "local_branch_oid": local_branch_oid,
        "sealed_ref": SEALED_REF,
        "sealed_tip": sealed_ref_oid,
        "object_format": object_format,
        "object_type": object_type,
        "is_bare_repository": True,
        "is_inside_work_tree": False,
        "absolute_git_dir_matches_root": True,
        "git_common_dir": git_common_dir,
        "symbolic_head": symbolic_head,
        "ref_roster": refs,
        "git_fsck_no_dangling": True,
        "forbidden_paths_absent": True,
        "shared_object_alternates": False,
        "checkout_created": False,
        "git_status_invoked": False,
        "is_shallow_repository": False,
        "object_store": object_store,
        "disk_used_gib": used,
    }


def gate_git_identity(receipts: Sequence[Mapping[str, Any]]) -> GateResult:
    expected_roots = {
        ("ethereum", "a"): "ethereum-a.git",
        ("ethereum", "b"): "ethereum-b.git",
        ("bitcoin", "a"): "bitcoin-a.git",
        ("bitcoin", "b"): "bitcoin-b.git",
    }
    observed = {
        (str(row.get("protocol")), str(row.get("replica"))): row
        for row in receipts
    }
    exact_identity = len(receipts) == 4 and set(observed) == set(
        expected_roots
    )
    for key, expected_name in expected_roots.items():
        row = observed.get(key, {})
        spec = _repository_spec(key[0])
        exact_identity = exact_identity and (
            row.get("root_name") == expected_name
            and row.get("remote") == spec.remote
            and row.get("remote_head_symref")
            == f"refs/heads/{spec.branch}"
            and isinstance(row.get("remote_head_oid"), str)
            and HEX40.fullmatch(str(row.get("remote_head_oid"))) is not None
            and row.get("local_branch_ref")
            == f"refs/heads/{spec.branch}"
            and isinstance(row.get("local_branch_oid"), str)
            and HEX40.fullmatch(str(row.get("local_branch_oid"))) is not None
            and row.get("sealed_ref") == SEALED_REF
            and row.get("sealed_tip") == spec.sealed_tip
            and row.get("object_format") == spec.object_format
            and row.get("object_type") == "commit"
        )
    checks = {
        "four_fresh_independent_bare_roots": exact_identity,
        "all_bare_no_worktree": all(
            row.get("is_bare_repository") is True
            and row.get("is_inside_work_tree") is False
            and row.get("absolute_git_dir_matches_root") is True
            and row.get("git_common_dir") == "."
            for row in receipts
        ),
        "exact_head_sealed_ref_and_ref_roster": all(
            row.get("symbolic_head") == "refs/heads/master"
            and row.get("local_branch_ref") == "refs/heads/master"
            and row.get("ref_roster")
            == ["refs/heads/master", SEALED_REF]
            for row in receipts
        ),
        "all_fsck": all(
            row.get("git_fsck_no_dangling") is True for row in receipts
        ),
        "no_forbidden_paths_or_shared_objects": all(
            row.get("forbidden_paths_absent") is True
            and row.get("shared_object_alternates") is False
            and type(row.get("object_store", {}).get("regular_files")) is int
            and row.get("object_store", {}).get("regular_files") > 0
            and row.get("object_store", {}).get("symlinks") == 0
            and row.get("object_store", {}).get("multiple_link_files") == 0
            for row in receipts
        ),
        "no_checkout_index_status_or_shallow_state": all(
            row.get("checkout_created") is False
            and row.get("git_status_invoked") is False
            and row.get("is_shallow_repository") is False
            for row in receipts
        ),
        "disk_guard": all(
            type(row.get("disk_used_gib")) is int
            and 0 <= row["disk_used_gib"] <= DISK_LIMIT_GIB
            for row in receipts
        ),
    }
    return GateResult(
        name="sealed_git_identity_and_object_integrity",
        passed=all(checks.values()),
        metrics={"checks": checks, "receipts": list(receipts)},
        failure=";".join(key for key, value in checks.items() if not value),
    )


def collect_commit_chain(
    repo: Path,
    protocol: str,
    ledger: AccessLedger,
) -> list[CommitRecord]:
    spec = _repository_spec(protocol)
    sealed_oid = _git_text(
        ["-C", str(repo), "rev-parse", f"{SEALED_REF}^{{commit}}"],
        ledger=ledger,
    )
    if sealed_oid != spec.sealed_tip:
        raise RuntimeError(
            f"PSIM-D2 sealed traversal ref changed: {protocol}"
        )
    output = _git_text(
        [
            "-C",
            str(repo),
            "rev-list",
            "--first-parent",
            "--reverse",
            SEALED_REF,
        ],
        ledger=ledger,
    )
    object_ids = [line for line in output.splitlines() if line]
    if not object_ids or object_ids[-1] != spec.sealed_tip:
        raise RuntimeError(
            f"PSIM-D2 first-parent chain does not end at tip: {protocol}"
        )
    records: list[CommitRecord] = []
    prior_effective: date | None = None
    prior_oid: str | None = None
    for index, (oid, raw) in enumerate(
        core._cat_file_batch(
            repo,
            object_ids,
            expected_type="commit",
            ledger=ledger,
        )
    ):
        record = parse_commit_object(
            protocol,
            oid,
            raw,
            index,
            prior_effective,
        )
        if prior_oid is not None and record.parent_oid != prior_oid:
            raise RuntimeError(
                f"PSIM-D2 first-parent chain is discontinuous: {protocol}"
            )
        if prior_oid is None and record.parent_oid is not None:
            raise RuntimeError(
                f"PSIM-D2 first record is not repository root: {protocol}"
            )
        records.append(record)
        prior_oid = oid
        prior_effective = record.effective_day
    return records


collect_proposal_groups = core.collect_proposal_groups
materialize_events = core.materialize_events
build_daily_cards = core.build_daily_cards
build_control_metrics = core.build_control_metrics


def _commit_rows(
    records: Sequence[CommitRecord],
) -> list[dict[str, Any]]:
    return core._commit_rows(records)


def _group_rows(
    groups: Sequence[ProposalGroup],
) -> list[dict[str, Any]]:
    return core._group_rows(groups)


def build_self_check_manifest() -> dict[str, Any]:
    inherited = core.build_self_check_manifest()
    inherited_core = {
        key: value
        for key, value in inherited.items()
        if key != "manifest_hash"
    }
    inherited_bytes = canonical_json_bytes(inherited)
    if (
        inherited.get("policy_id") != "PSIM-D1"
        or inherited.get("manifest_hash")
        != D1_CORE_SELF_CHECK_MANIFEST_HASH
        or inherited.get("manifest_hash") != canonical_hash(inherited_core)
        or sha256_bytes(inherited_bytes)
        != D1_CORE_SELF_CHECK_STDOUT_SHA256
        or inherited.get("failed") != []
        or inherited.get("forbidden_access")
        != AccessLedger.zero().snapshot()
        or inherited.get("network_calls") != 0
        or inherited.get("source_event_rows_opened") != 0
        or inherited.get("outcomes_opened") is not False
    ):
        raise RuntimeError("PSIM-D2 inherited core self-check changed")
    payload = {
        "protocol_version": SELF_CHECK_PROTOCOL,
        "policy_id": POLICY_ID,
        "inherited_core": {
            "runner_path": D1_CORE_RUNNER_PATH.as_posix(),
            "runner_commit": D1_CORE_COMMIT,
            "runner_sha256": D1_CORE_RUNNER_SHA256,
            "manifest_hash": inherited["manifest_hash"],
            "stdout_sha256": sha256_bytes(inherited_bytes),
        },
        "checks": inherited["checks"],
        "failed": inherited["failed"],
        "synthetic": inherited["synthetic"],
        "forbidden_access": inherited["forbidden_access"],
        "network_calls": 0,
        "git_commands": 0,
        "source_event_rows_opened": 0,
        "official_source_opened": False,
        "outcomes_opened": False,
    }
    return {**payload, "manifest_hash": canonical_hash(payload)}


def self_check_bytes() -> bytes:
    payload = build_self_check_manifest()
    if payload["failed"]:
        raise RuntimeError(
            "PSIM-D2 synthetic self-check failed: "
            + ",".join(payload["failed"])
        )
    return canonical_json_bytes(payload)


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        env=_git_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _assert_committed(
    path: str | Path,
    *,
    expected_commit: str | None = None,
) -> str:
    relative = Path(path).as_posix()
    commit = _git_output("log", "-1", "--format=%H", "--", relative)
    if HEX40.fullmatch(commit) is None:
        raise RuntimeError(f"PSIM-D2 path is not committed: {relative}")
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError(f"PSIM-D2 path commit changed: {relative}")
    return commit


def _git_blob_sha256(commit: str, path: str | Path) -> str:
    relative = Path(path).as_posix()
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=REPO_ROOT,
        env=_git_environment(),
        check=True,
        capture_output=True,
    )
    return sha256_bytes(completed.stdout)


def _worktree_clean() -> bool:
    return not _git_output(
        "status",
        "--porcelain",
        "--untracked-files=all",
    )


def _binding(
    path: str | Path,
    commit: str,
    digest: str,
) -> dict[str, str]:
    return {
        "path": Path(path).as_posix(),
        "commit": commit,
        "sha256": digest,
    }


def _implementation_binding(path: str | Path) -> dict[str, str]:
    commit = _assert_committed(path)
    digest = sha256_file(path)
    if _git_blob_sha256(commit, path) != digest:
        raise RuntimeError(
            f"PSIM-D2 committed implementation differs: {path}"
        )
    return _binding(path, commit, digest)


def _validate_binding(
    binding: Mapping[str, Any],
    *,
    path: str | Path,
    expected_commit: str | None = None,
    expected_sha256: str | None = None,
) -> None:
    relative = Path(path).as_posix()
    commit = binding.get("commit")
    digest = binding.get("sha256")
    if (
        set(binding) != {"path", "commit", "sha256"}
        or binding.get("path") != relative
        or not isinstance(commit, str)
        or HEX40.fullmatch(commit) is None
        or not isinstance(digest, str)
        or HEX64.fullmatch(digest) is None
        or (expected_commit is not None and commit != expected_commit)
        or (expected_sha256 is not None and digest != expected_sha256)
        or _assert_committed(path, expected_commit=commit) != commit
        or sha256_file(path) != digest
        or _git_blob_sha256(commit, path) != digest
    ):
        raise RuntimeError(f"PSIM-D2 binding mismatch: {relative}")


def _load_preregistration() -> dict[str, Any]:
    path = repository_path(PREREGISTRATION_PATH)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("PSIM-D2 preregistration is absent or unsafe")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "PSIM-D2 preregistration is unreadable"
        ) from error
    core_payload = {
        key: value
        for key, value in payload.items()
        if key != "manifest_hash"
    }
    if (
        not isinstance(payload, dict)
        or raw != canonical_json_bytes(payload)
        or sha256_bytes(raw) != PREREGISTRATION_SHA256
        or payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH
        or payload.get("manifest_hash") != canonical_hash(core_payload)
        or payload != prereg.build_preregistration()
        or payload.get("candidate", {}).get("id") != POLICY_ID
        or payload.get("inheritance_proof", {}).get(
            "all_other_paths_byte_equal"
        )
        is not True
    ):
        raise RuntimeError("PSIM-D2 preregistration manifest changed")
    return payload


def python_runtime() -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    git_version = _git_output("--version")
    if git_version != prereg.GIT_VERSION:
        raise RuntimeError(
            f"PSIM-D2 Git version changed: {git_version}"
        )
    return {
        "python": {
            "path": str(executable),
            "sha256": sha256_file(executable),
            "version": sys.version,
        },
        "git_version": git_version,
    }


def static_authority() -> dict[str, Any]:
    registration = _load_preregistration()
    bindings = {
        "decision": _binding(
            DECISION_PATH,
            DECISION_COMMIT,
            DECISION_SHA256,
        ),
        "preregistration": _binding(
            PREREGISTRATION_PATH,
            PREREGISTRATION_COMMIT,
            PREREGISTRATION_SHA256,
        ),
        "preregistration_producer": _binding(
            PREREGISTRATION_SCRIPT_PATH,
            PREREGISTRATION_COMMIT,
            PREREGISTRATION_SCRIPT_SHA256,
        ),
        "preregistration_document": _binding(
            PREREGISTRATION_DOC_PATH,
            PREREGISTRATION_COMMIT,
            PREREGISTRATION_DOC_SHA256,
        ),
        "d1_terminal_rejection": _binding(
            D1_TERMINAL_PATH,
            prereg.D1_TERMINAL_COMMIT,
            prereg.D1_TERMINAL_SHA256,
        ),
        "core_runner": _binding(
            D1_CORE_RUNNER_PATH,
            D1_CORE_COMMIT,
            D1_CORE_RUNNER_SHA256,
        ),
        "core_tests": _binding(
            D1_CORE_TEST_PATH,
            D1_CORE_COMMIT,
            D1_CORE_TEST_SHA256,
        ),
        "implementation_contract": _implementation_binding(
            IMPLEMENTATION_CONTRACT_PATH
        ),
    }
    for name, path, commit, digest in (
        (
            "decision",
            DECISION_PATH,
            DECISION_COMMIT,
            DECISION_SHA256,
        ),
        (
            "preregistration",
            PREREGISTRATION_PATH,
            PREREGISTRATION_COMMIT,
            PREREGISTRATION_SHA256,
        ),
        (
            "preregistration_producer",
            PREREGISTRATION_SCRIPT_PATH,
            PREREGISTRATION_COMMIT,
            PREREGISTRATION_SCRIPT_SHA256,
        ),
        (
            "preregistration_document",
            PREREGISTRATION_DOC_PATH,
            PREREGISTRATION_COMMIT,
            PREREGISTRATION_DOC_SHA256,
        ),
        (
            "d1_terminal_rejection",
            D1_TERMINAL_PATH,
            prereg.D1_TERMINAL_COMMIT,
            prereg.D1_TERMINAL_SHA256,
        ),
        (
            "core_runner",
            D1_CORE_RUNNER_PATH,
            D1_CORE_COMMIT,
            D1_CORE_RUNNER_SHA256,
        ),
        (
            "core_tests",
            D1_CORE_TEST_PATH,
            D1_CORE_COMMIT,
            D1_CORE_TEST_SHA256,
        ),
    ):
        _validate_binding(
            bindings[name],
            path=path,
            expected_commit=commit,
            expected_sha256=digest,
        )

    source = registration["source_contract"]
    support = registration["source_support_contract"]
    source_core = {
        "source_start": format_time(SOURCE_START),
        "source_end_exclusive": format_time(SOURCE_END_EXCLUSIVE),
        "card_end_exclusive": format_time(CARD_END_EXCLUSIVE),
        "repositories": source["repositories"],
        "repository_representation": source[
            "repository_representation"
        ],
        "clone_arguments": source["clone_arguments"],
        "bare_repository_contract": source[
            "bare_repository_contract"
        ],
        "schedules": registration["availability_contract"]["schedules"],
        "splits": registration["split_contract"]["splits"],
        "gates": support["gates_in_order"],
        "controls": support["relation_controls"],
        "parser_reference": registration["parser_contract"][
            "reference_parser"
        ],
        "d1_core_runner": bindings["core_runner"],
    }
    return {
        "runtime": python_runtime(),
        **bindings,
        "preregistration_manifest_hash": registration["manifest_hash"],
        "authorized_delta_hash": registration["inheritance_proof"][
            "authorized_delta_hash"
        ],
        "source_authority_hash": canonical_hash(source_core),
    }


def _run_self_check_subprocess() -> dict[str, Any]:
    argv = [sys.executable, RUNNER_PATH.as_posix(), "self-check"]
    completed = subprocess.run(
        argv,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0 or completed.stderr:
        raise RuntimeError(
            "PSIM-D2 self-check subprocess failed: "
            + completed.stderr.decode("utf-8", errors="replace")
        )
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "PSIM-D2 self-check output is not JSON"
        ) from error
    core_payload = {
        key: value
        for key, value in payload.items()
        if key != "manifest_hash"
    }
    if (
        completed.stdout != canonical_json_bytes(payload)
        or payload.get("protocol_version") != SELF_CHECK_PROTOCOL
        or payload.get("policy_id") != POLICY_ID
        or payload.get("manifest_hash") != canonical_hash(core_payload)
        or payload.get("failed") != []
        or payload.get("network_calls") != 0
        or payload.get("git_commands") != 0
        or payload.get("source_event_rows_opened") != 0
        or payload.get("official_source_opened") is not False
        or payload.get("outcomes_opened") is not False
        or payload.get("forbidden_access")
        != AccessLedger.zero().snapshot()
    ):
        raise RuntimeError("PSIM-D2 self-check manifest changed")
    return {
        "argv": argv,
        "exit_code": completed.returncode,
        "stdout_sha256": sha256_bytes(completed.stdout),
        "manifest_hash": payload["manifest_hash"],
        "inherited_core": payload["inherited_core"],
        "network_calls": 0,
        "git_commands": 0,
        "source_event_rows_opened": 0,
        "official_source_opened": False,
        "outcomes_opened": False,
        "forbidden_access": AccessLedger.zero().snapshot(),
    }


def _pytest_counts(stdout: str, stderr: str) -> dict[str, int]:
    return core._pytest_counts(stdout, stderr)


VERIFICATION_TEST_PATHS = (
    D1_CORE_TEST_PATH,
    Path(
        "tests/test_preregister_protocol_specification_intent_maturity_d2.py"
    ),
    TEST_PATH,
)


def _run_pytest_verification() -> dict[str, Any]:
    argv = [
        ".venv/bin/pytest",
        "-q",
        *(path.as_posix() for path in VERIFICATION_TEST_PATHS),
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "."
    completed = subprocess.run(
        argv,
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    counts = _pytest_counts(completed.stdout, completed.stderr)
    if (
        completed.returncode != 0
        or counts["failed"]
        or counts["skipped"]
        or counts["errors"]
        or counts["xfailed"]
        or counts["xpassed"]
    ):
        raise RuntimeError(
            "PSIM-D2 exact pytest verification failed\n"
            + completed.stdout
            + completed.stderr
        )
    return {
        "argv": argv,
        "environment_override": {"PYTHONPATH": "."},
        "exit_code": completed.returncode,
        **counts,
    }


def build_execution_seal() -> dict[str, Any]:
    if not _worktree_clean():
        raise RuntimeError(
            "PSIM-D2 seal creation requires a clean worktree"
        )
    authority = static_authority()
    runner = _implementation_binding(RUNNER_PATH)
    tests = _implementation_binding(TEST_PATH)
    implementation_contract = authority["implementation_contract"]
    head = _git_output("rev-parse", "HEAD")
    if (
        runner["commit"] != head
        or tests["commit"] != head
        or implementation_contract["commit"] != head
    ):
        raise RuntimeError(
            "PSIM-D2 runner, tests, and implementation contract "
            "must share current HEAD"
        )
    core_payload = {
        "protocol_version": SEAL_PROTOCOL,
        "policy_id": POLICY_ID,
        "authority": authority,
        "runner": runner,
        "tests": tests,
        "shared_commit": head,
        "synthetic_verification": {
            "self_check": _run_self_check_subprocess(),
            "pytest": _run_pytest_verification(),
        },
        "forbidden_access": AccessLedger.zero().snapshot(),
    }
    return {
        **core_payload,
        "seal_hash": canonical_hash(core_payload),
    }


def create_execution_seal() -> dict[str, Any]:
    payload = build_execution_seal()
    _write_once_bytes(
        EXECUTION_SEAL_PATH,
        canonical_json_bytes(payload),
    )
    return payload


def validate_execution_seal() -> dict[str, Any]:
    path = repository_path(EXECUTION_SEAL_PATH)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("PSIM-D2 execution seal is absent or unsafe")
    seal_commit = _assert_committed(EXECUTION_SEAL_PATH)
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("PSIM-D2 execution seal is unreadable") from error
    if raw != canonical_json_bytes(payload):
        raise RuntimeError(
            "PSIM-D2 execution seal bytes are noncanonical"
        )
    expected = {
        "protocol_version",
        "policy_id",
        "authority",
        "runner",
        "tests",
        "shared_commit",
        "synthetic_verification",
        "forbidden_access",
        "seal_hash",
    }
    core_payload = {
        key: value
        for key, value in payload.items()
        if key != "seal_hash"
    }
    if (
        set(payload) != expected
        or payload.get("protocol_version") != SEAL_PROTOCOL
        or payload.get("policy_id") != POLICY_ID
        or payload.get("seal_hash") != canonical_hash(core_payload)
        or payload.get("authority") != static_authority()
        or payload.get("forbidden_access")
        != AccessLedger.zero().snapshot()
    ):
        raise RuntimeError("PSIM-D2 execution seal core changed")
    shared_commit = payload.get("shared_commit")
    if (
        not isinstance(shared_commit, str)
        or HEX40.fullmatch(shared_commit) is None
    ):
        raise RuntimeError("PSIM-D2 sealed commit is malformed")
    _validate_binding(
        payload["runner"],
        path=RUNNER_PATH,
        expected_commit=shared_commit,
    )
    _validate_binding(
        payload["tests"],
        path=TEST_PATH,
        expected_commit=shared_commit,
    )
    head = _git_output("rev-parse", "HEAD")
    parent_row = _git_output(
        "rev-list",
        "--parents",
        "-n",
        "1",
        seal_commit,
    ).split()
    changed_paths = set(
        _git_output(
            "diff",
            "--name-only",
            shared_commit,
            seal_commit,
        ).splitlines()
    )
    if (
        head != seal_commit
        or parent_row != [seal_commit, shared_commit]
        or changed_paths
        != {
            EXECUTION_SEAL_PATH.as_posix(),
            SEAL_TEST_PATH.as_posix(),
        }
        or _assert_committed(
            SEAL_TEST_PATH,
            expected_commit=seal_commit,
        )
        != seal_commit
    ):
        raise RuntimeError(
            "PSIM-D2 seal commit is not the exact current execution HEAD"
        )
    expected_verification = {
        "self_check": _run_self_check_subprocess(),
        "pytest": _run_pytest_verification(),
    }
    if payload.get("synthetic_verification") != expected_verification:
        raise RuntimeError(
            "PSIM-D2 sealed synthetic verification changed"
        )
    return payload


def _artifact_entry(
    path: str | Path,
    raw: bytes,
    *,
    rows: int,
    row_hash: str,
) -> dict[str, Any]:
    return {
        "path": Path(path).as_posix(),
        "sha256": sha256_bytes(raw),
        "rows": rows,
        "row_hash": row_hash,
    }


def build_control_report(
    cards: Sequence[DailyCard],
    metrics: Mapping[str, Any],
    gate: GateResult,
) -> dict[str, Any]:
    if tuple(metrics) != tuple(core.prereg.RELATION_CONTROLS):
        raise RuntimeError(
            "PSIM-D2 control order differs from preregistration"
        )
    if gate.name != "relation_control_sensitivity":
        raise RuntimeError(
            "PSIM-D2 control report received the wrong gate"
        )
    card_rows = [card_row(row) for row in cards]
    report_core = {
        "protocol_version": CONTROL_REPORT_PROTOCOL,
        "policy_id": POLICY_ID,
        "baseline_cards": rows_fingerprint(card_rows),
        "control_order": list(core.prereg.RELATION_CONTROLS),
        "metrics": dict(metrics),
        "gate": gate.payload(),
        "outcomes_opened": False,
        "profitability_result": False,
        "forbidden_access": {
            name: 0 for name in FORBIDDEN_ACCESS_FIELDS
        },
    }
    return {
        **report_core,
        "report_hash": canonical_hash(report_core),
    }


def _combined_event_rows(
    events: Sequence[ProposalEvent],
) -> list[dict[str, Any]]:
    ordered = sorted(
        events,
        key=lambda row: (
            row.effective_day,
            row.protocol,
            row.first_parent_index,
            row.proposal_number,
            row.event_id,
        ),
    )
    return [event_row(row) for row in ordered]


def build_pass_artifacts(
    config: Config,
    events: Sequence[ProposalEvent],
    cards: Sequence[DailyCard],
    controls: Mapping[str, Any],
    control_gate: GateResult,
) -> tuple[dict[Path, bytes], dict[str, Any]]:
    event_rows = _combined_event_rows(events)
    card_rows = [card_row(row) for row in cards]
    control_report = build_control_report(
        cards,
        controls,
        control_gate,
    )
    event_jsonl = jsonl_bytes(event_rows)
    card_jsonl = jsonl_bytes(card_rows)
    raw_by_path = {
        config.events_path: deterministic_gzip(event_jsonl),
        config.cards_path: deterministic_gzip(card_jsonl),
        config.controls_path: canonical_json_bytes(control_report),
    }
    manifest = {
        "events": _artifact_entry(
            config.events_path,
            raw_by_path[config.events_path],
            rows=len(event_rows),
            row_hash=sha256_bytes(event_jsonl),
        ),
        "daily_cards": _artifact_entry(
            config.cards_path,
            raw_by_path[config.cards_path],
            rows=len(card_rows),
            row_hash=sha256_bytes(card_jsonl),
        ),
        "controls": _artifact_entry(
            config.controls_path,
            raw_by_path[config.controls_path],
            rows=len(core.prereg.RELATION_CONTROLS),
            row_hash=canonical_hash(dict(controls)),
        ),
    }
    return raw_by_path, manifest


def _authority_report(
    seal: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "execution_seal": {
            "path": EXECUTION_SEAL_PATH.as_posix(),
            "sha256": sha256_file(EXECUTION_SEAL_PATH),
            "seal_hash": seal["seal_hash"],
            "shared_commit": seal["shared_commit"],
            "runner": seal["runner"],
            "tests": seal["tests"],
        },
        **dict(authority),
    }


def _first_failure(
    gates: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    for index, gate in enumerate(gates, start=1):
        if gate.get("passed") is not True:
            return {"gate_id": index, "name": gate.get("name")}
    return None


def build_result_report(
    *,
    decision: str,
    authority: Mapping[str, Any],
    gates: Sequence[GateResult],
    source_audit: Mapping[str, Any],
    event_count: int,
    card_count: int,
    artifacts: Mapping[str, Any] | None,
    ledger: AccessLedger,
    error: BaseException | None = None,
) -> dict[str, Any]:
    gate_rows = [gate.payload() for gate in gates]
    first_failure = _first_failure(gate_rows)
    if [row["name"] for row in gate_rows] != list(
        GATE_NAMES[: len(gate_rows)]
    ):
        raise RuntimeError("PSIM-D2 gate order changed")
    if decision == "pass":
        if (
            len(gate_rows) != len(GATE_NAMES)
            or first_failure is not None
            or artifacts is None
            or any(row["passed"] is not True for row in gate_rows)
        ):
            raise RuntimeError("PSIM-D2 pass report is incomplete")
    elif decision == "reject":
        if not gate_rows or first_failure is None or artifacts is not None:
            raise RuntimeError("PSIM-D2 rejection report is incomplete")
    else:
        raise RuntimeError("PSIM-D2 result decision changed")
    source_opened = bool(
        source_audit.get("proposal_path_incidence_opened", False)
        or ledger.source_path_rows_opened
        or ledger.proposal_blobs_opened
    )
    report_core = {
        "protocol_version": RESULT_PROTOCOL,
        "policy_id": POLICY_ID,
        "decision": decision,
        "terminal_action": (
            PASS_ACTION if decision == "pass" else FAILURE_ACTION
        ),
        "profitability_result": False,
        "outcomes_opened": False,
        "source_incidence_opened": source_opened,
        "authority": dict(authority),
        "gates": gate_rows,
        "first_failure": first_failure,
        "source_audit": dict(source_audit),
        "counts": {
            "events": event_count,
            "daily_cards": card_count,
        },
        "artifacts": None if artifacts is None else dict(artifacts),
        "access_ledger": ledger.snapshot(),
        "error": (
            None if error is None else {"type": type(error).__name__}
        ),
    }
    return {
        **report_core,
        "result_hash": canonical_hash(report_core),
    }


def _read_canonical_json(path: str | Path) -> dict[str, Any]:
    target = repository_path(path)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"PSIM-D2 artifact path is unsafe: {path}")
    raw = target.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"PSIM-D2 artifact is unreadable: {path}") from error
    if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
        raise RuntimeError(f"PSIM-D2 artifact is noncanonical: {path}")
    return payload


def _entry_exists(path: str | Path) -> bool:
    return os.path.lexists(repository_path(path))


def _validate_result_report(
    payload: Mapping[str, Any],
    *,
    config: Config,
    decision: str,
) -> None:
    report_core = {
        key: value
        for key, value in payload.items()
        if key != "result_hash"
    }
    gates = payload.get("gates")
    access = payload.get("access_ledger")
    if (
        payload.get("protocol_version") != RESULT_PROTOCOL
        or payload.get("policy_id") != POLICY_ID
        or payload.get("decision") != decision
        or payload.get("terminal_action")
        != (PASS_ACTION if decision == "pass" else FAILURE_ACTION)
        or payload.get("profitability_result") is not False
        or payload.get("outcomes_opened") is not False
        or payload.get("result_hash") != canonical_hash(report_core)
        or not isinstance(gates, list)
        or [row.get("name") for row in gates]
        != list(GATE_NAMES[: len(gates)])
        or payload.get("first_failure") != _first_failure(gates)
        or not isinstance(access, dict)
        or set(access) != set(AccessLedger.zero().snapshot())
        or any(
            type(value) is not int or value < 0
            for value in access.values()
        )
    ):
        raise RuntimeError("PSIM-D2 terminal result report changed")
    if decision == "pass":
        artifacts = payload.get("artifacts")
        expected_paths = {
            "events": config.events_path.as_posix(),
            "daily_cards": config.cards_path.as_posix(),
            "controls": config.controls_path.as_posix(),
        }
        if (
            len(gates) != len(GATE_NAMES)
            or any(row.get("passed") is not True for row in gates)
            or not isinstance(artifacts, dict)
            or set(artifacts) != set(expected_paths)
            or gates[-1].get("metrics", {}).get("prepared_artifacts")
            != artifacts
            or payload.get("source_incidence_opened") is not True
            or any(access[name] for name in FORBIDDEN_ACCESS_FIELDS)
        ):
            raise RuntimeError("PSIM-D2 pass report is invalid")
        for name, entry in artifacts.items():
            if (
                not isinstance(entry, dict)
                or set(entry) != {"path", "sha256", "rows", "row_hash"}
                or entry.get("path") != expected_paths[name]
                or not isinstance(entry.get("sha256"), str)
                or HEX64.fullmatch(entry["sha256"]) is None
                or type(entry.get("rows")) is not int
                or entry["rows"] < 0
                or not isinstance(entry.get("row_hash"), str)
                or HEX64.fullmatch(entry["row_hash"]) is None
            ):
                raise RuntimeError(
                    "PSIM-D2 pass artifact manifest is invalid"
                )
    elif (
        not gates
        or payload.get("first_failure") is None
        or payload.get("artifacts") is not None
    ):
        raise RuntimeError("PSIM-D2 rejection report is invalid")


def terminal_state(config: Config | None = None) -> dict[str, Any] | None:
    config = Config() if config is None else config
    pass_paths = (
        config.events_path,
        config.cards_path,
        config.controls_path,
        config.result_path,
    )
    rejection_exists = _entry_exists(config.rejection_path)
    pass_exists = {path: _entry_exists(path) for path in pass_paths}
    if not rejection_exists and not any(pass_exists.values()):
        return None
    if rejection_exists and not any(pass_exists.values()):
        payload = _read_canonical_json(config.rejection_path)
        _validate_result_report(
            payload,
            config=config,
            decision="reject",
        )
        return payload
    if not rejection_exists and all(pass_exists.values()):
        payload = _read_canonical_json(config.result_path)
        _validate_result_report(payload, config=config, decision="pass")
        for entry in payload["artifacts"].values():
            path = repository_path(entry["path"])
            if (
                path.is_symlink()
                or not path.is_file()
                or sha256_file(path) != entry["sha256"]
            ):
                raise RuntimeError("PSIM-D2 pass artifact hash changed")
        return payload
    raise RuntimeError("PSIM-D2 terminal state is partial or conflicting")


def _publish_rejection(
    config: Config,
    payload: Mapping[str, Any],
) -> None:
    _validate_result_report(payload, config=config, decision="reject")
    if any(
        _entry_exists(path)
        for path in (
            config.events_path,
            config.cards_path,
            config.controls_path,
            config.result_path,
        )
    ):
        raise RuntimeError(
            "PSIM-D2 pass artifact exists before rejection"
        )
    _write_once_bytes(
        config.rejection_path,
        canonical_json_bytes(payload),
    )


def _publish_pass_group(
    config: Config,
    raw_by_path: Mapping[Path, bytes],
    report: Mapping[str, Any],
) -> None:
    _validate_result_report(report, config=config, decision="pass")
    expected_raw_paths = {
        config.events_path,
        config.cards_path,
        config.controls_path,
    }
    if set(raw_by_path) != expected_raw_paths:
        raise RuntimeError("PSIM-D2 pass artifact group changed")
    expected_entries = {
        **dict(raw_by_path),
        config.result_path: canonical_json_bytes(report),
    }
    rejection_exists = _entry_exists(config.rejection_path)
    existing = {path: _entry_exists(path) for path in expected_entries}
    if not rejection_exists and all(existing.values()):
        for relative, raw in expected_entries.items():
            target = repository_path(relative)
            if (
                target.is_symlink()
                or not target.is_file()
                or target.read_bytes() != raw
            ):
                raise RuntimeError(
                    "PSIM-D2 existing pass artifact conflicts "
                    "with publication"
                )
        return
    if rejection_exists or any(existing.values()):
        raise RuntimeError("PSIM-D2 terminal target already exists")
    staged: list[tuple[Path, Path]] = []
    published: list[tuple[Path, Path]] = []
    try:
        for relative, raw in expected_entries.items():
            target = _safe_destination(relative)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.",
                suffix=".psim-d2-stage",
                dir=target.parent,
            )
            temporary = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            staged.append((temporary, target))
        for temporary, target in staged:
            os.link(temporary, target)
            published.append((temporary, target))
        for parent in {target.parent for _, target in staged}:
            descriptor = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except BaseException:
        for temporary, target in reversed(published):
            try:
                if target.exists() and os.path.samefile(temporary, target):
                    target.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def gate_terminal_publication_ready(config: Config) -> GateResult:
    paths = (
        config.result_path,
        config.rejection_path,
        config.events_path,
        config.cards_path,
        config.controls_path,
    )
    failures: list[str] = []
    for path in paths:
        if _entry_exists(path):
            failures.append(f"exists:{path}")
        try:
            _safe_destination(path)
        except RuntimeError as error:
            failures.append(str(error))
    return GateResult(
        name="terminal_publication",
        passed=not failures,
        metrics={"paths": [str(path) for path in paths]},
        failure=";".join(failures),
    )


def _gate_error(name: str, error: BaseException) -> GateResult:
    return GateResult(
        name=name,
        passed=False,
        metrics={
            "gate_evaluation_completed": False,
            "error_type": type(error).__name__,
        },
        failure=f"{name} raised {type(error).__name__}",
    )


def _evaluate_gate(
    name: str,
    builder: Any,
) -> tuple[GateResult, BaseException | None]:
    try:
        gate = builder()
    except Exception as error:
        return _gate_error(name, error), error
    if not isinstance(gate, GateResult) or gate.name != name:
        error = RuntimeError(
            "PSIM-D2 gate builder returned wrong identity"
        )
        return _gate_error(name, error), error
    return gate, None


def _combined_events(
    values: Mapping[tuple[str, str], Sequence[ProposalEvent]],
    replica: str,
) -> list[ProposalEvent]:
    return sorted(
        [
            event
            for protocol in ("ethereum", "bitcoin")
            for event in values[(protocol, replica)]
        ],
        key=lambda row: (
            row.effective_day,
            row.protocol,
            row.first_parent_index,
            row.proposal_number,
            row.event_id,
        ),
    )


def _validate_no_symlink_ancestors(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    cursor = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        cursor /= component
        if os.path.lexists(cursor) and cursor.is_symlink():
            raise ValueError(
                f"PSIM-D2 path has a symlink ancestor: {cursor}"
            )
    return absolute


def _validate_source_configuration(config: Config) -> None:
    if config.network_timeout_seconds <= 0:
        raise ValueError("PSIM-D2 network timeout must be positive")
    if not config.source_root.is_absolute():
        raise ValueError("PSIM-D2 source root must be absolute")
    if config.source_root != DEFAULT_SOURCE_ROOT:
        raise ValueError("PSIM-D2 official source root is frozen")
    source_root = _validate_no_symlink_ancestors(config.source_root)
    if source_root.exists() and not source_root.is_dir():
        raise ValueError("PSIM-D2 source root is unsafe")
    output_paths = (
        config.result_path,
        config.rejection_path,
        config.events_path,
        config.cards_path,
        config.controls_path,
    )
    expected_output_paths = (
        DEFAULT_RESULT_PATH,
        DEFAULT_REJECTION_PATH,
        DEFAULT_EVENTS_PATH,
        DEFAULT_CARDS_PATH,
        DEFAULT_CONTROLS_PATH,
    )
    if output_paths != expected_output_paths:
        raise ValueError("PSIM-D2 official output paths are frozen")
    if len({repository_path(path) for path in output_paths}) != len(
        output_paths
    ):
        raise ValueError("PSIM-D2 output paths are not unique")
    for path in output_paths:
        _safe_destination(path)


def _acquire_run_lock() -> tuple[Path, bytes]:
    target = _safe_destination(RUN_LOCK_PATH)
    raw = canonical_json_bytes(
        {
            "policy_id": POLICY_ID,
            "state": "SOURCE_RUN_IN_PROGRESS",
        }
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    return target, raw


def _release_run_lock(target: Path, expected: bytes) -> None:
    if (
        target.is_symlink()
        or not target.is_file()
        or target.read_bytes() != expected
    ):
        raise RuntimeError("PSIM-D2 run lock identity changed")
    target.unlink()
    descriptor = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_official(config: Config) -> dict[str, Any]:
    _validate_source_configuration(config)
    existing = terminal_state(config)
    if existing is not None:
        seal = validate_execution_seal()
        authority = static_authority()
        if existing.get("authority") != _authority_report(seal, authority):
            raise RuntimeError("PSIM-D2 terminal authority changed")
        return existing

    seal = validate_execution_seal()
    authority = static_authority()
    authority_record = _authority_report(seal, authority)
    if not _worktree_clean():
        raise RuntimeError(
            "PSIM-D2 official run requires a clean worktree"
        )
    run_lock_path, run_lock_bytes = _acquire_run_lock()
    try:
        concurrent_terminal = terminal_state(config)
    except BaseException:
        _release_run_lock(run_lock_path, run_lock_bytes)
        raise
    if concurrent_terminal is not None:
        _release_run_lock(run_lock_path, run_lock_bytes)
        raise RuntimeError(
            "PSIM-D2 terminal state appeared concurrently"
        )

    ledger = AccessLedger.zero()
    gates: list[GateResult] = []
    receipts: list[dict[str, Any]] = []
    chains: dict[tuple[str, str], list[CommitRecord]] = {}
    groups: dict[tuple[str, str], list[ProposalGroup]] = {}
    group_issues: dict[tuple[str, str], list[str]] = {}
    events: dict[tuple[str, str], list[ProposalEvent]] = {}
    cards_a: list[DailyCard] = []
    cards_b: list[DailyCard] = []
    control_metrics: dict[str, Any] | None = None
    control_gate: GateResult | None = None
    raw_by_path: dict[Path, bytes] | None = None
    artifact_manifest: dict[str, Any] | None = None
    source_audit: dict[str, Any] = {
        "source_root": str(config.source_root),
        "repository_representation": (
            "BARE_OBJECT_DATABASE_NO_WORKTREE_NO_INDEX"
        ),
        "source_traversal_ref": SEALED_REF,
        "source_classes_opened": [],
        "remote_identity_opened": False,
        "commit_metadata_opened": False,
        "proposal_path_incidence_opened": False,
        "proposal_blobs_opened": False,
        "clone_receipts_sha256": None,
        "commit_chains_sha256": None,
        "proposal_groups_sha256": None,
        "events_sha256": None,
        "cards_sha256": None,
        "controls_sha256": None,
        "git_status_invoked": False,
        "checkout_created": False,
        "disk_limit_gib": DISK_LIMIT_GIB,
        "disk_used_gib_at_start": None,
        "disk_below_limit_at_start": False,
        "source_run_attempt": 1,
        "repair_or_provider_swap_used": False,
    }

    def mark_source_class(name: str) -> None:
        values = source_audit["source_classes_opened"]
        if name not in values:
            values.append(name)

    def reject(error: BaseException | None = None) -> dict[str, Any]:
        combined = (
            _combined_events(events, "a")
            if all(
                (protocol, "a") in events
                for protocol in ("ethereum", "bitcoin")
            )
            else []
        )
        report = build_result_report(
            decision="reject",
            authority=authority_record,
            gates=gates,
            source_audit=source_audit,
            event_count=len(combined),
            card_count=len(cards_a),
            artifacts=None,
            ledger=ledger,
            error=error,
        )
        try:
            _publish_rejection(config, report)
            return report
        finally:
            _release_run_lock(run_lock_path, run_lock_bytes)

    try:
        _validate_no_symlink_ancestors(config.source_root)
        config.source_root.mkdir(parents=True, exist_ok=True)
        _validate_no_symlink_ancestors(config.source_root)
        used_gib = enforce_disk_guard(config.source_root)
        source_audit["disk_used_gib_at_start"] = used_gib
        source_audit["disk_below_limit_at_start"] = True
        for protocol in ("ethereum", "bitcoin"):
            for replica in ("a", "b"):
                destination = clone_path(config, protocol, replica)
                if os.path.lexists(destination):
                    raise RuntimeError(
                        "PSIM-D2 fresh clone root already exists: "
                        f"{destination}"
                    )
                mark_source_class("git_remote_identity")
                source_audit["remote_identity_opened"] = True
                receipts.append(
                    prepare_source_repository(
                        config,
                        protocol,
                        replica,
                        ledger,
                    )
                )
        source_audit["clone_receipts_sha256"] = canonical_hash(receipts)
        gate = gate_git_identity(receipts)
    except Exception as error:
        gates.append(_gate_error(GATE_NAMES[0], error))
        return reject(error)
    gates.append(gate)
    if not gate.passed:
        return reject()

    try:
        mark_source_class("git_commit_metadata")
        source_audit["commit_metadata_opened"] = True
        for protocol in ("ethereum", "bitcoin"):
            for replica in ("a", "b"):
                chains[(protocol, replica)] = collect_commit_chain(
                    clone_path(config, protocol, replica),
                    protocol,
                    ledger,
                )
        source_audit["commit_chains_sha256"] = canonical_hash(
            {
                f"{protocol}:{replica}": _commit_rows(rows)
                for (protocol, replica), rows in sorted(chains.items())
            }
        )
        gate = gate_commit_chains(chains)
    except Exception as error:
        gates.append(_gate_error(GATE_NAMES[1], error))
        return reject(error)
    gates.append(gate)
    if not gate.passed:
        return reject()

    try:
        mark_source_class("proposal_path_incidence")
        source_audit["proposal_path_incidence_opened"] = True
        for protocol in ("ethereum", "bitcoin"):
            for replica in ("a", "b"):
                key = (protocol, replica)
                rows, issues = collect_proposal_groups(
                    clone_path(config, protocol, replica),
                    chains[key],
                    ledger,
                )
                groups[key] = rows
                group_issues[key] = issues
        source_audit["proposal_groups_sha256"] = canonical_hash(
            {
                f"{protocol}:{replica}": {
                    "groups": _group_rows(
                        groups[(protocol, replica)]
                    ),
                    "issues": group_issues[(protocol, replica)],
                }
                for protocol in ("ethereum", "bitcoin")
                for replica in ("a", "b")
            }
        )
        gate = gate_proposal_groups(groups, group_issues)
    except Exception as error:
        gates.append(_gate_error(GATE_NAMES[2], error))
        return reject(error)
    gates.append(gate)
    if not gate.passed:
        return reject()

    try:
        mark_source_class("proposal_blob_content")
        source_audit["proposal_blobs_opened"] = True
        for protocol in ("ethereum", "bitcoin"):
            for replica in ("a", "b"):
                key = (protocol, replica)
                events[key] = materialize_events(
                    clone_path(config, protocol, replica),
                    groups[key],
                    ledger,
                )
        source_audit["events_sha256"] = canonical_hash(
            {
                f"{protocol}:{replica}": [
                    event_row(row)
                    for row in events[(protocol, replica)]
                ]
                for protocol in ("ethereum", "bitcoin")
                for replica in ("a", "b")
            }
        )
        gate = gate_event_parser_replay(events)
    except Exception as error:
        gates.append(_gate_error(GATE_NAMES[3], error))
        return reject(error)
    gates.append(gate)
    if not gate.passed:
        return reject()

    events_a = _combined_events(events, "a")
    events_b = _combined_events(events, "b")
    try:
        cards_a = build_daily_cards(events_a, ledger=ledger)
        cards_b = build_daily_cards(events_b, ledger=ledger)
        support = split_support_metrics(events_a, cards_a)
    except Exception as error:
        gates.append(_gate_error(GATE_NAMES[4], error))
        return reject(error)
    gate, error = _evaluate_gate(
        GATE_NAMES[4],
        lambda: gate_split_support(support),
    )
    gates.append(gate)
    if not gate.passed:
        return reject(error)

    gate, error = _evaluate_gate(
        GATE_NAMES[5],
        lambda: gate_vocabulary(events_a),
    )
    gates.append(gate)
    if not gate.passed:
        return reject(error)

    gate, error = _evaluate_gate(
        GATE_NAMES[6],
        lambda: gate_daily_cards(cards_a),
    )
    gates.append(gate)
    if not gate.passed:
        return reject(error)

    source_audit["cards_sha256"] = canonical_hash(
        [card_row(row) for row in cards_a]
    )
    gate, error = _evaluate_gate(
        GATE_NAMES[7],
        lambda: gate_independent_replay(events, cards_a, cards_b),
    )
    gates.append(gate)
    if not gate.passed:
        return reject(error)

    gate, error = _evaluate_gate(
        GATE_NAMES[8],
        lambda: gate_future_append(events_a, cards_a),
    )
    gates.append(gate)
    if not gate.passed:
        return reject(error)

    try:
        control_metrics = build_control_metrics(events_a, cards_a)
    except Exception as error:
        gates.append(_gate_error(GATE_NAMES[9], error))
        return reject(error)
    source_audit["controls_sha256"] = canonical_hash(control_metrics)
    gate, error = _evaluate_gate(
        GATE_NAMES[9],
        lambda: gate_control_sensitivity(control_metrics),
    )
    gates.append(gate)
    if not gate.passed:
        return reject(error)
    control_gate = gate

    gate, error = _evaluate_gate(
        GATE_NAMES[10],
        lambda: gate_pairing_reset_quarantine(events_a, cards_a),
    )
    gates.append(gate)
    if not gate.passed:
        return reject(error)

    gate, error = _evaluate_gate(
        GATE_NAMES[11],
        lambda: gate_forbidden_access(ledger),
    )
    gates.append(gate)
    if not gate.passed:
        return reject(error)

    try:
        if control_metrics is None or control_gate is None:
            raise RuntimeError(
                "PSIM-D2 control evidence disappeared"
            )
        raw_by_path, artifact_manifest = build_pass_artifacts(
            config,
            events_a,
            cards_a,
            control_metrics,
            control_gate,
        )
        terminal_gate = gate_terminal_publication_ready(config)
        if terminal_gate.passed:
            terminal_gate = replace(
                terminal_gate,
                metrics={
                    **dict(terminal_gate.metrics),
                    "prepared_artifacts": artifact_manifest,
                },
            )
    except Exception as error:
        gates.append(_gate_error(GATE_NAMES[12], error))
        return reject(error)
    gates.append(terminal_gate)
    if not terminal_gate.passed:
        return reject()
    if raw_by_path is None or artifact_manifest is None:
        raise RuntimeError(
            "PSIM-D2 pass artifacts were not prepared"
        )

    report = build_result_report(
        decision="pass",
        authority=authority_record,
        gates=gates,
        source_audit=source_audit,
        event_count=len(events_a),
        card_count=len(cards_a),
        artifacts=artifact_manifest,
        ledger=ledger,
    )
    try:
        _publish_pass_group(config, raw_by_path, report)
    except Exception as error:
        gates[-1] = _gate_error(GATE_NAMES[12], error)
        return reject(error)
    _release_run_lock(run_lock_path, run_lock_bytes)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-check")
    subparsers.add_parser("seal")
    subparsers.add_parser("create-seal")
    subparsers.add_parser("validate-seal")
    run = subparsers.add_parser("run")
    run.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    run.add_argument(
        "--network-timeout-seconds",
        type=int,
        default=900,
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.command == "self-check":
        sys.stdout.buffer.write(self_check_bytes())
        return
    if arguments.command in {"seal", "create-seal"}:
        payload = create_execution_seal()
        print(
            json.dumps(
                {
                    "path": EXECUTION_SEAL_PATH.as_posix(),
                    "seal_hash": payload["seal_hash"],
                    "shared_commit": payload["shared_commit"],
                },
                sort_keys=True,
                indent=2,
            )
        )
        return
    if arguments.command == "validate-seal":
        payload = validate_execution_seal()
        print(
            json.dumps(
                {
                    "path": EXECUTION_SEAL_PATH.as_posix(),
                    "seal_hash": payload["seal_hash"],
                },
                sort_keys=True,
                indent=2,
            )
        )
        return
    report = run_official(
        Config(
            source_root=Path(arguments.source_root),
            network_timeout_seconds=arguments.network_timeout_seconds,
        )
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "terminal_action": report["terminal_action"],
                "first_failure": report["first_failure"],
                "result_hash": report["result_hash"],
            },
            sort_keys=True,
            indent=2,
        )
    )
    if report["decision"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
