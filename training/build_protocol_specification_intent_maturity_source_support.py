"""Build, gate, seal, and publish the outcome-blind PSIM-D1 source.

The runner has three hard boundaries:

* ``self-check`` is synthetic-only and performs no network/source access;
* ``create-seal`` binds committed runner/tests and reruns the synthetic battery;
* ``run`` validates that seal before opening official Git source incidence.

Market, funding, model, reward, action, trade, and portfolio data are forbidden
throughout this module.
"""

from __future__ import annotations

import argparse
import difflib
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, time, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

from training import preregister_protocol_specification_intent_maturity as prereg


UTC = timezone.utc
DAY = timedelta(days=1)
POLICY_ID = "PSIM-D1"
RUNNER_PROTOCOL = "psim_d1_source_support_runner_v1"
SEAL_PROTOCOL = "psim_d1_source_support_execution_seal_v1"
RESULT_PROTOCOL = "psim_d1_source_support_result_v1"
SELF_CHECK_PROTOCOL = "psim_d1_synthetic_self_check_v1"
CONTROL_REPORT_PROTOCOL = "psim_d1_source_controls_v1"
PASS_ACTION = "ACCEPT_PSIM_D1_SOURCE_SUPPORT_ONLY_NO_PROFITABILITY_CLAIM"
FAILURE_ACTION = (
    "REJECT_PSIM_D1_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES"
)
GATE_NAMES = prereg.SOURCE_ONLY_GATES

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(
    "training/build_protocol_specification_intent_maturity_source_support.py"
)
TEST_PATH = Path(
    "tests/test_build_protocol_specification_intent_maturity_source_support.py"
)
SEAL_TEST_PATH = Path(
    "tests/test_psim_d1_source_support_execution_seal.py"
)
IMPLEMENTATION_CONTRACT_PATH = Path(
    "docs/psim-d1-source-support-implementation-contract-2026-07-25.md"
)
PREREGISTRATION_SCRIPT_PATH = prereg.SCRIPT_PATH
PREREGISTRATION_PATH = prereg.DEFAULT_OUTPUT
PREREGISTRATION_DOC_PATH = Path(
    "docs/psim-d1-source-support-preregistration-2026-07-25.md"
)
DECISION_PATH = prereg.DECISION_PATH

PREREGISTRATION_COMMIT = "2125584b732e81ba34d3f81534b8b1279a379e74"
PREREGISTRATION_SHA256 = (
    "bd4053574fe6285c34356baaa080e215f08bbf8142e9c0c968bffbdccb2dc736"
)
PREREGISTRATION_MANIFEST_HASH = (
    "bdf49fb396779599eb329a407685435c05217f132ea856f9bb743914b5afbe81"
)
PREREGISTRATION_SCRIPT_SHA256 = (
    "982f14cff8c903c9ad528018ab996f053179cc054626bfaa8af7ab63405f858b"
)
PREREGISTRATION_DOC_SHA256 = (
    "09612ec67c093edf952bf54664d1f73a6796c96e2ac3be24b60c804ae700074d"
)
DECISION_COMMIT = prereg.SELECTION_COMMIT
DECISION_SHA256 = prereg.DECISION_SHA256

EXECUTION_SEAL_PATH = Path(
    "results/psim_d1_source_support_execution_seal_2026-07-25.json"
)
RUN_LOCK_PATH = Path("results/.psim_d1_source_support_run.lock")
DEFAULT_RESULT_PATH = Path(
    "results/protocol_specification_intent_maturity_source_support_"
    "2026-07-25.json"
)
DEFAULT_REJECTION_PATH = Path(
    "results/protocol_specification_intent_maturity_source_rejection_"
    "2026-07-25.json"
)
DEFAULT_EVENTS_PATH = Path(
    "data/protocol_specification_intent_maturity_events_2020_2023.jsonl.gz"
)
DEFAULT_CARDS_PATH = Path(
    "data/protocol_specification_intent_maturity_cards_2020_2024q1.jsonl.gz"
)
DEFAULT_CONTROLS_PATH = Path(
    "results/protocol_specification_intent_maturity_source_controls_"
    "2026-07-25.json"
)
DEFAULT_SOURCE_ROOT = Path("/tmp/psim-d1-source")

SOURCE_START = datetime(2020, 1, 1, tzinfo=UTC)
SOURCE_END_EXCLUSIVE = datetime(2024, 1, 1, tzinfo=UTC)
CARD_END_EXCLUSIVE = datetime(2024, 4, 1, tzinfo=UTC)
RAW_DIFF_HEADER_PATTERN = re.compile(
    rb":(?P<old_mode>[0-7]{6}) (?P<new_mode>[0-7]{6}) "
    rb"(?P<old_oid>[0-9a-f]{40}) (?P<new_oid>[0-9a-f]{40}) "
    rb"(?P<status>[A-Z])\Z"
)
HEX40 = re.compile(r"[0-9a-f]{40}", re.ASCII)
HEX64 = re.compile(r"[0-9a-f]{64}", re.ASCII)
ZERO_OID = "0" * 40
FORBIDDEN_ACCESS_FIELDS = (
    "pre_2020_proposal_blobs_opened",
    "post_2023_proposal_blobs_opened",
    "btc_market_rows_read",
    "funding_rows_read",
    "future_return_rows_read",
    "reward_rows_built",
    "models_loaded",
    "model_outputs_built",
    "trade_rows_built",
    "pnl_rows_built",
    "cagr_values_built",
    "strict_mdd_values_built",
)

MODEL_SECTION_ORDER = (
    "ABSTRACT",
    "MOTIVATION",
    "SPECIFICATION",
    "RATIONALE",
    "BACKWARD_COMPATIBILITY",
    "SECURITY",
    "TESTS",
    "IMPLEMENTATION",
)
SECTION_ROTATION_ORDER = (
    "ABSTRACT",
    "MOTIVATION",
    "SPECIFICATION",
    "RATIONALE",
    "BACKWARD_COMPATIBILITY",
    "SECURITY",
    "TESTS",
    "IMPLEMENTATION",
    "COPYRIGHT",
)
SECTION_ROTATION = {
    value: SECTION_ROTATION_ORDER[(index + 1) % len(SECTION_ROTATION_ORDER)]
    for index, value in enumerate(SECTION_ROTATION_ORDER)
}


@dataclass(frozen=True)
class Config:
    source_root: Path = DEFAULT_SOURCE_ROOT
    result_path: Path = DEFAULT_RESULT_PATH
    rejection_path: Path = DEFAULT_REJECTION_PATH
    events_path: Path = DEFAULT_EVENTS_PATH
    cards_path: Path = DEFAULT_CARDS_PATH
    controls_path: Path = DEFAULT_CONTROLS_PATH
    network_timeout_seconds: int = 900


@dataclass(frozen=True)
class CommitRecord:
    protocol: str
    oid: str
    tree_oid: str
    parent_oid: str | None
    first_parent_index: int
    committer_epoch: int
    committer_day: date
    effective_day: date


@dataclass(frozen=True)
class PathChange:
    path: str
    old_mode: str
    new_mode: str
    old_oid: str
    new_oid: str
    status: str


@dataclass(frozen=True)
class ProposalGroup:
    protocol: str
    proposal_number: int
    commit_oid: str
    first_parent_index: int
    committer_day: date
    effective_day: date
    old_path: str | None
    new_path: str | None
    old_blob_oid: str | None
    new_blob_oid: str | None
    event_type: str
    event_id: str


@dataclass(frozen=True)
class BlobFeatures:
    blob_oid: str
    blob_sha256: str
    proposal_number: int
    header: Mapping[str, str]
    dependency_edges: Mapping[str, tuple[int, ...]]
    normalized_lines: tuple[str, ...]
    line_sections: tuple[str, ...]
    section_presence: tuple[str, ...]


@dataclass(frozen=True)
class ProposalEvent:
    protocol: str
    proposal_number: int
    commit_oid: str
    first_parent_index: int
    committer_day: date
    effective_day: date
    event_type: str
    event_id: str
    old_path: str | None
    new_path: str | None
    old_blob_oid: str | None
    new_blob_oid: str | None
    old_blob_sha256: str | None
    new_blob_sha256: str | None
    old_blob_role: str
    prior_dependency_state: str
    old_sections: tuple[str, ...]
    new_sections: tuple[str, ...]
    changed_sections: tuple[str, ...]
    dependency_delta_state: str
    dependency_edge_delta_count: int
    line_change_count: int
    changed_section_count: int
    window_revision_count: int
    window_age_days: int | None
    update_gap_days: int | None
    intent_text: str
    memorization_excluded: bool
    available_at: Mapping[str, datetime]
    control_order: int = 0


@dataclass(frozen=True)
class DailyCard:
    schedule: str
    decision_at: datetime
    split: str | None
    local_payload: Mapping[str, Any]
    local_payload_sha256: str
    prior_card_hash: str
    card_hash: str
    event_ids: tuple[str, ...]
    eligible: Mapping[str, bool]


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    metrics: Mapping[str, Any] = field(default_factory=dict)
    failure: str = ""

    def payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "metrics": dict(self.metrics),
            "failure": self.failure,
        }


@dataclass
class AccessLedger:
    network_commands: int = 0
    git_commands: int = 0
    source_path_rows_opened: int = 0
    proposal_blobs_opened: int = 0
    proposal_text_rows_opened: int = 0
    pre_2020_proposal_blobs_opened: int = 0
    post_2023_proposal_blobs_opened: int = 0
    daily_cards_built: int = 0
    btc_market_rows_read: int = 0
    funding_rows_read: int = 0
    future_return_rows_read: int = 0
    reward_rows_built: int = 0
    models_loaded: int = 0
    model_outputs_built: int = 0
    trade_rows_built: int = 0
    pnl_rows_built: int = 0
    cagr_values_built: int = 0
    strict_mdd_values_built: int = 0

    @classmethod
    def zero(cls) -> "AccessLedger":
        return cls()

    def snapshot(self) -> dict[str, int]:
        return asdict(self)


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


def deterministic_gzip(raw: bytes) -> bytes:
    output = gzip.compress(raw, compresslevel=9, mtime=0)
    return output


def jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row), pretty=False) for row in rows)


def format_time(value: datetime) -> str:
    normalized = value.astimezone(UTC).replace(microsecond=0)
    return normalized.isoformat().replace("+00:00", "Z")


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo != UTC:
        raise ValueError(f"PSIM timestamp is not UTC: {value}")
    return parsed


def _safe_destination(path: str | Path) -> Path:
    destination = Path(os.path.abspath(repository_path(path)))
    root = REPO_ROOT.resolve()
    try:
        relative = destination.relative_to(root)
    except ValueError as error:
        raise RuntimeError(
            f"PSIM output escapes repository: {destination}"
        ) from error
    if not relative.parts:
        raise RuntimeError(f"PSIM output is repository root: {destination}")
    cursor = root
    for component in relative.parts[:-1]:
        cursor /= component
        if os.path.lexists(cursor) and cursor.is_symlink():
            raise RuntimeError(f"PSIM output parent is symlinked: {cursor}")
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
            raise RuntimeError(f"existing PSIM artifact differs: {destination}")
        return destination
    temporary = destination.with_name(destination.name + ".tmp")
    if os.path.lexists(temporary):
        if temporary.is_dir() or temporary.is_symlink():
            raise RuntimeError(f"unsafe PSIM temporary path: {temporary}")
        temporary.unlink()
    temporary.write_bytes(raw)
    os.replace(temporary, destination)
    return destination


def _write_json_once(path: str | Path, payload: Mapping[str, Any]) -> Path:
    return _write_once_bytes(path, canonical_json_bytes(payload))


def git_object_sha1(kind: str, raw: bytes) -> str:
    header = f"{kind} {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw, usedforsecurity=False).hexdigest()


def bucket_token(name: str, value: int | None) -> str:
    if value is None or value < 0:
        return f"{name.upper()}_SENTINEL"
    edges = prereg.BUCKET_EDGES[name]
    for lower, upper in zip(edges, edges[1:]):
        if lower <= value < upper:
            return f"{name.upper()}_{lower}_{upper - 1}"
    return f"{name.upper()}_{edges[-1]}_PLUS"


def split_name(value: datetime) -> str | None:
    for split in prereg.SPLITS:
        start = parse_time(str(split["decision_start"]))
        end = parse_time(str(split["decision_end_exclusive"]))
        if start <= value < end:
            return str(split["name"])
    return None


def _git_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for key in tuple(environment):
        if key.startswith("GIT_"):
            environment.pop(key)
    environment.update(
        {
            "LC_ALL": "C",
            "LANG": "C",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
        }
    )
    return environment


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
        raise RuntimeError(f"PSIM git command failed: {detail}")
    return completed


def _git_text(
    arguments: Sequence[str],
    *,
    ledger: AccessLedger,
    cwd: Path | None = None,
    network: bool = False,
    timeout: int = 900,
) -> str:
    raw = _run_git(
        arguments,
        ledger=ledger,
        cwd=cwd,
        network=network,
        timeout=timeout,
    ).stdout
    return raw.decode("utf-8", errors="strict").strip()


def _disk_used_gib(path: Path) -> int:
    usage = shutil.disk_usage(path if path.exists() else path.parent)
    return usage.used // (1024**3)


def enforce_disk_guard(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    used = _disk_used_gib(path)
    if used > prereg.DISK_LIMIT_GIB:
        raise RuntimeError(
            f"PSIM disk guard exceeded: {used} GiB > {prereg.DISK_LIMIT_GIB} GiB"
        )
    return used


def _repository_spec(protocol: str) -> prereg.RepositorySpec:
    matches = [row for row in prereg.REPOSITORIES if row.protocol == protocol]
    if len(matches) != 1:
        raise RuntimeError(f"PSIM repository spec is not unique: {protocol}")
    return matches[0]


def clone_path(config: Config, protocol: str, replica: str) -> Path:
    if replica not in {"a", "b"}:
        raise ValueError("PSIM replica must be a or b")
    return config.source_root / f"{protocol}-{replica}"


def _validate_no_alternates(repo: Path) -> None:
    alternates = repo / ".git" / "objects" / "info" / "alternates"
    if alternates.exists() and alternates.read_bytes().strip():
        raise RuntimeError("PSIM shared Git object alternates are forbidden")


def prepare_source_repository(
    config: Config,
    protocol: str,
    replica: str,
    ledger: AccessLedger,
) -> dict[str, Any]:
    spec = _repository_spec(protocol)
    destination = clone_path(config, protocol, replica)
    enforce_disk_guard(destination)
    if destination.exists() and not (destination / ".git").is_dir():
        raise RuntimeError(f"PSIM source path is not a Git repository: {destination}")
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        _run_git(
            [
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "--single-branch",
                "--branch",
                spec.branch,
                spec.remote,
                str(destination),
            ],
            ledger=ledger,
            network=True,
            timeout=config.network_timeout_seconds,
        )
    _validate_no_alternates(destination)
    remote = _git_text(
        ["-C", str(destination), "remote", "get-url", "origin"],
        ledger=ledger,
    )
    if remote != spec.remote:
        raise RuntimeError(f"PSIM source remote changed for {protocol}")
    remote_head = _git_text(
        ["ls-remote", "--symref", spec.remote, "HEAD"],
        ledger=ledger,
        network=True,
        timeout=config.network_timeout_seconds,
    )
    remote_head_lines = remote_head.splitlines()
    first_line = remote_head_lines[0] if remote_head_lines else ""
    expected = f"ref: refs/heads/{spec.branch}\tHEAD"
    if (
        first_line != expected
        or len(remote_head_lines) != 2
        or "\t" not in remote_head_lines[1]
    ):
        raise RuntimeError(f"PSIM remote HEAD symref changed for {protocol}")
    remote_head_oid, remote_head_name = remote_head_lines[1].split("\t", 1)
    if (
        HEX40.fullmatch(remote_head_oid) is None
        or remote_head_name != "HEAD"
    ):
        raise RuntimeError(f"PSIM remote HEAD object changed for {protocol}")
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
    object_format = _git_text(
        ["-C", str(destination), "rev-parse", "--show-object-format"],
        ledger=ledger,
    )
    if object_format != spec.object_format:
        raise RuntimeError(f"PSIM object format changed for {protocol}")
    object_type = _git_text(
        ["-C", str(destination), "cat-file", "-t", spec.sealed_tip],
        ledger=ledger,
    )
    if object_type != "commit":
        raise RuntimeError(f"PSIM sealed tip is not a commit for {protocol}")
    _run_git(
        ["-C", str(destination), "fsck", "--no-dangling"],
        ledger=ledger,
        timeout=config.network_timeout_seconds,
    )
    porcelain = _git_text(
        ["-C", str(destination), "status", "--porcelain=v1"],
        ledger=ledger,
    )
    if porcelain:
        raise RuntimeError(f"PSIM no-checkout source repository is dirty: {protocol}")
    used = enforce_disk_guard(destination)
    return {
        "protocol": protocol,
        "replica": replica,
        "remote": remote,
        "remote_head_symref": f"refs/heads/{spec.branch}",
        "remote_head_oid": remote_head_oid,
        "local_tracking_symref": spec.remote_head_symref,
        "sealed_tip": spec.sealed_tip,
        "object_format": object_format,
        "git_fsck_no_dangling": True,
        "shared_object_alternates": False,
        "worktree_porcelain_empty": True,
        "disk_used_gib": used,
    }


def _cat_file_batch(
    repo: Path,
    object_ids: Sequence[str],
    *,
    expected_type: str,
    ledger: AccessLedger,
    network_capable: bool = False,
) -> Iterator[tuple[str, bytes]]:
    if not object_ids:
        return
    ledger.git_commands += 1
    if network_capable:
        ledger.network_commands += 1
    process = subprocess.Popen(
        ["git", "-C", str(repo), "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_git_environment(),
    )
    if process.stdin is None or process.stdout is None or process.stderr is None:
        process.kill()
        raise RuntimeError("PSIM failed to open git cat-file pipes")
    try:
        for expected_oid in object_ids:
            process.stdin.write(expected_oid.encode("ascii") + b"\n")
            process.stdin.flush()
            header = process.stdout.readline().rstrip(b"\n")
            parts = header.split(b" ")
            if (
                len(parts) != 3
                or parts[0].decode("ascii") != expected_oid
                or parts[1].decode("ascii") != expected_type
                or not parts[2].isdigit()
            ):
                raise RuntimeError("PSIM git cat-file returned unexpected header")
            size = int(parts[2])
            raw = process.stdout.read(size)
            if len(raw) != size or process.stdout.read(1) != b"\n":
                raise RuntimeError("PSIM git cat-file returned truncated object")
            yield expected_oid, raw
        process.stdin.close()
        stderr = process.stderr.read()
        return_code = process.wait()
        if return_code != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"PSIM git cat-file failed: {detail}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def parse_commit_object(
    protocol: str,
    expected_oid: str,
    raw: bytes,
    first_parent_index: int,
    prior_effective_day: date | None,
) -> CommitRecord:
    if git_object_sha1("commit", raw) != expected_oid:
        raise ValueError("PSIM commit object SHA-1 mismatch")
    if b"\n\n" not in raw:
        raise ValueError("PSIM commit object has no header terminator")
    header, _message = raw.split(b"\n\n", 1)
    tree_oid: str | None = None
    parents: list[str] = []
    committer_epoch: int | None = None
    for line in header.splitlines():
        if line.startswith(b"tree "):
            if tree_oid is not None:
                raise ValueError("PSIM commit object repeats tree")
            tree_oid = line[5:].decode("ascii")
        elif line.startswith(b"parent "):
            parents.append(line[7:].decode("ascii"))
        elif line.startswith(b"committer "):
            if committer_epoch is not None:
                raise ValueError("PSIM commit object repeats committer")
            parts = line.rsplit(b" ", 2)
            if (
                len(parts) != 3
                or not parts[1].isdigit()
                or re.fullmatch(rb"[+-][0-9]{4}", parts[2]) is None
            ):
                raise ValueError("PSIM committer identity time is malformed")
            committer_epoch = int(parts[1])
    if tree_oid is None or HEX40.fullmatch(tree_oid) is None:
        raise ValueError("PSIM commit tree OID is malformed")
    if any(HEX40.fullmatch(parent) is None for parent in parents):
        raise ValueError("PSIM commit parent OID is malformed")
    if committer_epoch is None:
        raise ValueError("PSIM commit has no committer time")
    committer_day = datetime.fromtimestamp(committer_epoch, tz=UTC).date()
    effective_day = (
        committer_day
        if prior_effective_day is None
        else max(prior_effective_day, committer_day)
    )
    return CommitRecord(
        protocol=protocol,
        oid=expected_oid,
        tree_oid=tree_oid,
        parent_oid=parents[0] if parents else None,
        first_parent_index=first_parent_index,
        committer_epoch=committer_epoch,
        committer_day=committer_day,
        effective_day=effective_day,
    )


def collect_commit_chain(
    repo: Path,
    protocol: str,
    ledger: AccessLedger,
) -> list[CommitRecord]:
    tip = _repository_spec(protocol).sealed_tip
    output = _git_text(
        ["-C", str(repo), "rev-list", "--first-parent", "--reverse", tip],
        ledger=ledger,
    )
    object_ids = [line for line in output.splitlines() if line]
    if not object_ids or object_ids[-1] != tip:
        raise RuntimeError(f"PSIM first-parent chain does not end at tip: {protocol}")
    records: list[CommitRecord] = []
    prior_effective: date | None = None
    prior_oid: str | None = None
    for index, (oid, raw) in enumerate(
        _cat_file_batch(
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
            raise RuntimeError(f"PSIM first-parent chain is discontinuous: {protocol}")
        if prior_oid is None and record.parent_oid is not None:
            raise RuntimeError(f"PSIM first record is not repository root: {protocol}")
        records.append(record)
        prior_oid = oid
        prior_effective = record.effective_day
    return records


def parse_raw_path_delta(raw: bytes) -> list[PathChange]:
    if not raw:
        return []
    tokens = raw.split(b"\x00")
    if tokens[-1] != b"":
        raise ValueError("PSIM raw path delta is not NUL terminated")
    tokens.pop()
    if len(tokens) % 2:
        raise ValueError("PSIM raw path delta lacks header/path pairs")
    rows: list[PathChange] = []
    seen: set[str] = set()
    for index in range(0, len(tokens), 2):
        header, path_raw = tokens[index], tokens[index + 1]
        match = RAW_DIFF_HEADER_PATTERN.fullmatch(header)
        if match is None:
            raise ValueError("PSIM raw path delta header is malformed")
        status = match.group("status").decode("ascii")
        if status not in {"A", "D", "M", "T"}:
            raise ValueError("PSIM raw path delta status is unsupported")
        path = path_raw.decode("utf-8", errors="strict")
        if path in seen:
            raise ValueError("PSIM raw path delta repeats path")
        seen.add(path)
        rows.append(
            PathChange(
                path=path,
                old_mode=match.group("old_mode").decode("ascii"),
                new_mode=match.group("new_mode").decode("ascii"),
                old_oid=match.group("old_oid").decode("ascii"),
                new_oid=match.group("new_oid").decode("ascii"),
                status=status,
            )
        )
    return rows


def _path_identity(protocol: str, path: str) -> tuple[int, str] | None:
    pattern = (
        prereg.EIP_PATH_PATTERN
        if protocol == "ethereum"
        else prereg.BIP_PATH_PATTERN
    )
    match = re.fullmatch(pattern, path, re.ASCII)
    if match is None:
        return None
    proposal_number = int(match.group(1), 10)
    if proposal_number <= 0:
        raise ValueError("PSIM path proposal number is not positive")
    extension = "md" if protocol == "ethereum" else str(match.group(2))
    return proposal_number, extension


def _path_delta(
    repo: Path,
    record: CommitRecord,
    ledger: AccessLedger,
) -> list[PathChange]:
    if record.parent_oid is None:
        arguments = [
            "-C",
            str(repo),
            "diff-tree",
            "--root",
            "--no-commit-id",
            "-r",
            "--raw",
            "-z",
            "--no-renames",
            record.oid,
        ]
    else:
        arguments = [
            "-C",
            str(repo),
            "diff-tree",
            "--no-commit-id",
            "-r",
            "--raw",
            "-z",
            "--no-renames",
            record.parent_oid,
            record.oid,
        ]
    raw = _run_git(arguments, ledger=ledger).stdout
    rows = parse_raw_path_delta(raw)
    ledger.source_path_rows_opened += len(rows)
    return rows


def _recognized_paths_in_tree(
    repo: Path,
    protocol: str,
    treeish: str,
    ledger: AccessLedger,
) -> dict[int, list[str]]:
    raw = _run_git(
        [
            "-C",
            str(repo),
            "ls-tree",
            "-r",
            "-z",
            "--name-only",
            treeish,
        ],
        ledger=ledger,
    ).stdout
    if raw and not raw.endswith(b"\x00"):
        raise ValueError("PSIM ls-tree output is not NUL terminated")
    values: dict[int, list[str]] = defaultdict(list)
    for token in raw.split(b"\x00"):
        if not token:
            continue
        path = token.decode("utf-8", errors="strict")
        identity = _path_identity(protocol, path)
        if identity is not None:
            values[identity[0]].append(path)
    return {key: sorted(paths) for key, paths in values.items()}


def _event_id(
    protocol: str,
    commit_oid: str,
    proposal_number: int,
    old_blob_oid: str | None,
    new_blob_oid: str | None,
) -> str:
    parts = (
        protocol,
        commit_oid,
        str(proposal_number),
        old_blob_oid or "NULL",
        new_blob_oid or "NULL",
    )
    return sha256_bytes(b"\x00".join(part.encode("ascii") for part in parts))


def proposal_groups_for_commit(
    repo: Path,
    record: CommitRecord,
    ledger: AccessLedger,
) -> tuple[list[ProposalGroup], list[str]]:
    changes = _path_delta(repo, record, ledger)
    grouped: dict[int, dict[str, list[tuple[str, str]]]] = defaultdict(
        lambda: {"old": [], "new": []}
    )
    for row in changes:
        identity = _path_identity(record.protocol, row.path)
        if identity is None:
            continue
        proposal_number, _extension = identity
        if row.old_oid != ZERO_OID:
            grouped[proposal_number]["old"].append((row.path, row.old_oid))
        if row.new_oid != ZERO_OID:
            grouped[proposal_number]["new"].append((row.path, row.new_oid))
    if not grouped:
        return [], []
    issues: list[str] = []
    tree_paths = _recognized_paths_in_tree(
        repo,
        record.protocol,
        record.oid,
        ledger,
    )
    parent_paths = (
        {}
        if record.parent_oid is None
        else _recognized_paths_in_tree(
            repo,
            record.protocol,
            record.parent_oid,
            ledger,
        )
    )
    for proposal_number, paths in sorted(tree_paths.items()):
        if len(paths) > 1:
            issues.append(
                f"{record.protocol}:{record.oid}:{proposal_number}:"
                "duplicate_new_tree_paths"
            )
    for proposal_number, paths in sorted(parent_paths.items()):
        if len(paths) > 1:
            issues.append(
                f"{record.protocol}:{record.oid}:{proposal_number}:"
                "duplicate_old_tree_paths"
            )
    rows: list[ProposalGroup] = []
    for proposal_number, sides in sorted(grouped.items()):
        old_rows = sorted(set(sides["old"]))
        new_rows = sorted(set(sides["new"]))
        if len(old_rows) > 1 or len(new_rows) > 1:
            issues.append(
                f"{record.protocol}:{record.oid}:{proposal_number}:"
                "ambiguous_old_or_new_blob"
            )
            continue
        if not old_rows and len(new_rows) == 1:
            event_type = "CREATE"
        elif len(old_rows) == 1 and len(new_rows) == 1:
            event_type = "UPDATE"
        elif len(old_rows) == 1 and not new_rows:
            event_type = "DELETE"
        else:
            issues.append(
                f"{record.protocol}:{record.oid}:{proposal_number}:"
                "invalid_event_shape"
            )
            continue
        old_path, old_oid = old_rows[0] if old_rows else (None, None)
        new_path, new_oid = new_rows[0] if new_rows else (None, None)
        rows.append(
            ProposalGroup(
                protocol=record.protocol,
                proposal_number=proposal_number,
                commit_oid=record.oid,
                first_parent_index=record.first_parent_index,
                committer_day=record.committer_day,
                effective_day=record.effective_day,
                old_path=old_path,
                new_path=new_path,
                old_blob_oid=old_oid,
                new_blob_oid=new_oid,
                event_type=event_type,
                event_id=_event_id(
                    record.protocol,
                    record.oid,
                    proposal_number,
                    old_oid,
                    new_oid,
                ),
            )
        )
    return rows, issues


def collect_proposal_groups(
    repo: Path,
    records: Sequence[CommitRecord],
    ledger: AccessLedger,
) -> tuple[list[ProposalGroup], list[str]]:
    rows: list[ProposalGroup] = []
    issues: list[str] = []
    for record in records:
        effective = datetime.combine(record.effective_day, time.min, tzinfo=UTC)
        if effective < SOURCE_START:
            continue
        if effective >= SOURCE_END_EXCLUSIVE:
            break
        commit_rows, commit_issues = proposal_groups_for_commit(
            repo,
            record,
            ledger,
        )
        rows.extend(commit_rows)
        issues.extend(commit_issues)
    identities = [row.event_id for row in rows]
    if len(identities) != len(set(identities)):
        issues.append("duplicate_event_id")
    return rows, sorted(issues)


def _canonical_heading(value: str) -> str:
    normalized = " ".join(
        value.casefold().strip().removesuffix(":").split()
    )
    for canonical, aliases in prereg.SECTION_ALIASES.items():
        if normalized in aliases:
            return canonical
    return "OTHER"


def _line_sections(protocol: str, lines: Sequence[str]) -> tuple[str, ...]:
    current = "OTHER"
    rows: list[str] = []
    markdown = re.compile(
        prereg.build_preregistration()["parser_contract"][
            "markdown_heading_pattern"
        ]
    )
    mediawiki = re.compile(
        prereg.build_preregistration()["parser_contract"][
            "mediawiki_heading_pattern"
        ]
    )
    for line in lines:
        match = markdown.fullmatch(line)
        if match is not None:
            current = _canonical_heading(match.group(1))
        elif protocol == "bitcoin":
            match = mediawiki.fullmatch(line)
            if match is not None:
                current = _canonical_heading(match.group(2))
        rows.append(current)
    return tuple(rows)


def _dependency_edges(
    protocol: str,
    header: Mapping[str, str],
    proposal_number: int,
) -> dict[str, tuple[int, ...]]:
    fields = (
        prereg.EIP_DEPENDENCY_FIELDS
        if protocol == "ethereum"
        else prereg.BIP_DEPENDENCY_FIELDS
    )
    result: dict[str, tuple[int, ...]] = {}
    for field_name in fields:
        if field_name in header:
            result[field_name] = prereg.parse_dependency_ids(
                header[field_name],
                self_id=proposal_number,
            )
        else:
            result[field_name] = ()
    return result


def parse_blob_features(
    protocol: str,
    proposal_number: int,
    blob_oid: str,
    raw: bytes,
) -> BlobFeatures:
    if git_object_sha1("blob", raw) != blob_oid:
        raise ValueError("PSIM blob object SHA-1 mismatch")
    header = (
        prereg.parse_eip_preamble(raw)
        if protocol == "ethereum"
        else prereg.parse_bip_preamble(raw)
    )
    number_field = "eip" if protocol == "ethereum" else "bip"
    parsed_number = prereg.parse_positive_proposal_number(header[number_field])
    if parsed_number != proposal_number:
        raise ValueError("PSIM path number differs from preamble number")
    lines = tuple(prereg.normalize_blob_bytes(raw))
    line_sections = _line_sections(protocol, lines)
    return BlobFeatures(
        blob_oid=blob_oid,
        blob_sha256=sha256_bytes(raw),
        proposal_number=proposal_number,
        header=dict(sorted(header.items())),
        dependency_edges=_dependency_edges(
            protocol,
            header,
            proposal_number,
        ),
        normalized_lines=lines,
        line_sections=line_sections,
        section_presence=tuple(sorted(set(line_sections))),
    )


def _flatten_dependency_edges(
    values: Mapping[str, Sequence[int]],
) -> set[tuple[str, int]]:
    return {
        (field_name, int(proposal))
        for field_name, proposals in values.items()
        for proposal in proposals
    }


def _dependency_delta(
    old: BlobFeatures | None,
    new: BlobFeatures | None,
) -> tuple[str, int]:
    if old is None and new is not None:
        return "NO_PRIOR", len(_flatten_dependency_edges(new.dependency_edges))
    if old is not None and new is None:
        return "DELETED", len(_flatten_dependency_edges(old.dependency_edges))
    if old is None or new is None:
        raise ValueError("PSIM dependency delta received empty pair")
    old_edges = _flatten_dependency_edges(old.dependency_edges)
    new_edges = _flatten_dependency_edges(new.dependency_edges)
    added = new_edges - old_edges
    removed = old_edges - new_edges
    if not added and not removed:
        return "STABLE", 0
    if added and not removed:
        return "ADDED", len(added)
    if removed and not added:
        return "REMOVED", len(removed)
    return "MIXED", len(added) + len(removed)


def _changed_lines_and_sections(
    old: BlobFeatures | None,
    new: BlobFeatures | None,
) -> tuple[int, tuple[str, ...], str]:
    old_lines = () if old is None else old.normalized_lines
    new_lines = () if new is None else new.normalized_lines
    old_sections = () if old is None else old.line_sections
    new_sections = () if new is None else new.line_sections
    matcher = difflib.SequenceMatcher(
        a=old_lines,
        b=new_lines,
        autojunk=False,
    )
    changed_sections: set[str] = set()
    selected_text: list[str] = []
    line_change_count = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        line_change_count += (i2 - i1) + (j2 - j1)
        changed_sections.update(old_sections[i1:i2])
        changed_sections.update(new_sections[j1:j2])
        for section_order in MODEL_SECTION_ORDER:
            for index in range(j1, j2):
                if new_sections[index] == section_order:
                    selected_text.append(
                        f"{section_order}|ADD|{new_lines[index]}"
                    )
            for index in range(i1, i2):
                if old_sections[index] == section_order:
                    selected_text.append(
                        f"{section_order}|REMOVE|{old_lines[index]}"
                    )
    intent_text = "\n".join(selected_text)
    if (
        len(intent_text.encode("utf-8"))
        > prereg.MAX_MODEL_TEXT_BYTES_PER_EVENT
    ):
        raise ValueError("PSIM model-visible intent text exceeds frozen bound")
    return line_change_count, tuple(sorted(changed_sections)), intent_text


def _available_at(effective_day: date) -> dict[str, datetime]:
    return {
        row.name: datetime.combine(
            effective_day + timedelta(days=row.delay_calendar_days),
            time(hour=12),
            tzinfo=UTC,
        )
        for row in prereg.ARCHIVE_SCHEDULES
    }


def materialize_events(
    repo: Path,
    groups: Sequence[ProposalGroup],
    ledger: AccessLedger,
) -> list[ProposalEvent]:
    for group in groups:
        blob_sides = sum(
            oid is not None
            for oid in (group.old_blob_oid, group.new_blob_oid)
        )
        if group.effective_day < SOURCE_START.date():
            ledger.pre_2020_proposal_blobs_opened += blob_sides
            raise RuntimeError(
                "PSIM attempted to open a pre-2020 proposal event blob"
            )
        if group.effective_day >= SOURCE_END_EXCLUSIVE.date():
            ledger.post_2023_proposal_blobs_opened += blob_sides
            raise RuntimeError(
                "PSIM attempted to open a post-2023 proposal event blob"
            )
    object_ids = sorted(
        {
            oid
            for group in groups
            for oid in (group.old_blob_oid, group.new_blob_oid)
            if oid is not None
        }
    )
    raw_by_oid: dict[str, bytes] = {}
    for oid, raw in _cat_file_batch(
        repo,
        object_ids,
        expected_type="blob",
        ledger=ledger,
        network_capable=True,
    ):
        raw_by_oid[oid] = raw
        ledger.proposal_blobs_opened += 1
    features: dict[tuple[str, int, str], BlobFeatures] = {}
    for group in groups:
        for oid in (group.old_blob_oid, group.new_blob_oid):
            if oid is None:
                continue
            key = (group.protocol, group.proposal_number, oid)
            if key not in features:
                features[key] = parse_blob_features(
                    group.protocol,
                    group.proposal_number,
                    oid,
                    raw_by_oid[oid],
                )
                ledger.proposal_text_rows_opened += 1
    prior_by_proposal: dict[tuple[str, int], tuple[int, date]] = {}
    events: list[ProposalEvent] = []
    quarantine = prereg.MEMORIZATION_QUARANTINE
    for group in sorted(
        groups,
        key=lambda row: (
            row.effective_day,
            row.first_parent_index,
            row.proposal_number,
            row.event_id,
        ),
    ):
        old = (
            None
            if group.old_blob_oid is None
            else features[
                (group.protocol, group.proposal_number, group.old_blob_oid)
            ]
        )
        new = (
            None
            if group.new_blob_oid is None
            else features[
                (group.protocol, group.proposal_number, group.new_blob_oid)
            ]
        )
        line_count, changed_sections, intent_text = _changed_lines_and_sections(
            old,
            new,
        )
        dependency_state, dependency_count = _dependency_delta(old, new)
        identity = (group.protocol, group.proposal_number)
        prior = prior_by_proposal.get(identity)
        old_blob_role = (
            "NO_OLD_BLOB"
            if old is None
            else (
                "PRE_WINDOW_BASELINE"
                if prior is None
                else "IN_WINDOW_PRIOR"
            )
        )
        prior_dependency_state = (
            "PRE_WINDOW_UNKNOWN"
            if prior is None
            else "IN_WINDOW_KNOWN"
        )
        revision_count = 0 if prior is None else prior[0] + 1
        first_day = group.effective_day if prior is None else prior[1]
        window_age_days = (group.effective_day - first_day).days
        previous_event = next(
            (
                event
                for event in reversed(events)
                if (event.protocol, event.proposal_number) == identity
            ),
            None,
        )
        update_gap_days = (
            None
            if previous_event is None
            else (group.effective_day - previous_event.effective_day).days
        )
        prior_by_proposal[identity] = (revision_count, first_day)
        events.append(
            ProposalEvent(
                protocol=group.protocol,
                proposal_number=group.proposal_number,
                commit_oid=group.commit_oid,
                first_parent_index=group.first_parent_index,
                committer_day=group.committer_day,
                effective_day=group.effective_day,
                event_type=group.event_type,
                event_id=group.event_id,
                old_path=group.old_path,
                new_path=group.new_path,
                old_blob_oid=group.old_blob_oid,
                new_blob_oid=group.new_blob_oid,
                old_blob_sha256=None if old is None else old.blob_sha256,
                new_blob_sha256=None if new is None else new.blob_sha256,
                old_blob_role=old_blob_role,
                prior_dependency_state=prior_dependency_state,
                old_sections=() if old is None else old.section_presence,
                new_sections=() if new is None else new.section_presence,
                changed_sections=changed_sections,
                dependency_delta_state=dependency_state,
                dependency_edge_delta_count=dependency_count,
                line_change_count=line_count,
                changed_section_count=len(changed_sections),
                window_revision_count=revision_count,
                window_age_days=window_age_days,
                update_gap_days=update_gap_days,
                intent_text=intent_text,
                memorization_excluded=(
                    group.proposal_number in quarantine[group.protocol]
                ),
                available_at=_available_at(group.effective_day),
            )
        )
    return events


def _event_order_key(event: ProposalEvent, schedule: str) -> tuple[Any, ...]:
    return (
        event.available_at[schedule],
        event.control_order,
        event.first_parent_index,
        event.proposal_number,
        event.event_id,
    )


def _event_payload(
    event: ProposalEvent,
    decision_at: datetime,
    schedule: str,
) -> dict[str, Any]:
    stale_days = max(
        0,
        (decision_at.date() - event.available_at[schedule].date()).days,
    )
    return {
        "protocol": event.protocol.upper(),
        "event_type": event.event_type,
        "window_revision_count_bucket": bucket_token(
            "window_revision_count",
            event.window_revision_count,
        ),
        "window_age_bucket": bucket_token(
            "window_age_days",
            event.window_age_days,
        ),
        "update_gap_bucket": bucket_token(
            "update_gap_days",
            event.update_gap_days,
        ),
        "stale_age_bucket": bucket_token("stale_age_days", stale_days),
        "old_sections": list(event.old_sections),
        "new_sections": list(event.new_sections),
        "changed_sections": list(event.changed_sections),
        "dependency_delta_state": event.dependency_delta_state,
        "dependency_edge_delta_count_bucket": bucket_token(
            "dependency_edge_delta_count",
            event.dependency_edge_delta_count,
        ),
        "line_change_count_bucket": bucket_token(
            "line_change_count",
            event.line_change_count,
        ),
        "changed_section_count_bucket": bucket_token(
            "changed_section_count",
            event.changed_section_count,
        ),
        "intent_text": event.intent_text,
        "memorization_excluded": event.memorization_excluded,
    }


def _decision_times() -> tuple[datetime, ...]:
    rows: list[datetime] = []
    current = datetime.combine(
        SOURCE_START.date(),
        time(hour=12, minute=5),
        tzinfo=UTC,
    )
    while current < CARD_END_EXCLUSIVE:
        rows.append(current)
        current += DAY
    return tuple(rows)


def _version_pair_eligible_ids(
    events: Sequence[ProposalEvent],
) -> set[str]:
    by_stratum: dict[tuple[str, int], list[ProposalEvent]] = defaultdict(list)
    for event in events:
        if event.event_type == "UPDATE":
            by_stratum[(event.protocol, event.effective_day.year)].append(event)
    return {
        event.event_id
        for values in by_stratum.values()
        if len(values) >= 2
        for event in values
    }


def build_daily_cards(
    events: Sequence[ProposalEvent],
    *,
    ledger: AccessLedger | None = None,
) -> list[DailyCard]:
    cards: list[DailyCard] = []
    version_eligible = _version_pair_eligible_ids(events)
    for schedule_row in prereg.ARCHIVE_SCHEDULES:
        schedule = schedule_row.name
        ordered = sorted(events, key=lambda row: _event_order_key(row, schedule))
        by_day: dict[date, list[ProposalEvent]] = defaultdict(list)
        for event in ordered:
            by_day[event.available_at[schedule].date()].append(event)
        prior_hash = canonical_hash(
            {"schedule": schedule, "state": "PSIM_CARD_CHAIN_START"}
        )
        latest: dict[str, ProposalEvent | None] = {
            "ethereum": None,
            "bitcoin": None,
        }
        history: dict[str, list[ProposalEvent]] = {
            "ethereum": [],
            "bitcoin": [],
        }
        for decision_at in _decision_times():
            new_events = sorted(
                by_day.get(decision_at.date(), []),
                key=lambda row: _event_order_key(row, schedule),
            )
            ethereum_new = [
                row for row in new_events if row.protocol == "ethereum"
            ]
            bitcoin_new = [
                row for row in new_events if row.protocol == "bitcoin"
            ]
            relation_units: list[dict[str, Any]] = []
            audit_pairs: list[tuple[str | None, str | None]] = []
            if ethereum_new and bitcoin_new:
                for left in ethereum_new:
                    for right in bitcoin_new:
                        relation_units.append(
                            {
                                "ethereum": _event_payload(
                                    left,
                                    decision_at,
                                    schedule,
                                ),
                                "bitcoin": _event_payload(
                                    right,
                                    decision_at,
                                    schedule,
                                ),
                                "counterpart_state": "SAME_DAY_CARTESIAN",
                                "memorization_excluded": (
                                    left.memorization_excluded
                                    or right.memorization_excluded
                                ),
                            }
                        )
                        audit_pairs.append((left.event_id, right.event_id))
            elif ethereum_new or bitcoin_new:
                anchors = ethereum_new or bitcoin_new
                anchor_protocol = anchors[0].protocol
                opposite = (
                    "bitcoin" if anchor_protocol == "ethereum" else "ethereum"
                )
                lower = decision_at - timedelta(
                    days=prereg.COUNTERPART_LOOKBACK_DAYS
                )
                candidates = [
                    row
                    for row in history[opposite]
                    if lower <= row.available_at[schedule] <= decision_at
                ]
                counterpart = (
                    max(
                        candidates,
                        key=lambda row: (
                            row.available_at[schedule],
                            row.event_id,
                        ),
                    )
                    if candidates
                    else None
                )
                for anchor in anchors:
                    left = anchor if anchor.protocol == "ethereum" else counterpart
                    right = anchor if anchor.protocol == "bitcoin" else counterpart
                    relation_units.append(
                        {
                            "ethereum": (
                                "NO_COUNTERPART"
                                if left is None
                                else _event_payload(
                                    left,
                                    decision_at,
                                    schedule,
                                )
                            ),
                            "bitcoin": (
                                "NO_COUNTERPART"
                                if right is None
                                else _event_payload(
                                    right,
                                    decision_at,
                                    schedule,
                                )
                            ),
                            "counterpart_state": (
                                "NO_COUNTERPART"
                                if counterpart is None
                                else "TRAILING_90D"
                            ),
                            "memorization_excluded": (
                                anchor.memorization_excluded
                                or (
                                    False
                                    if counterpart is None
                                    else counterpart.memorization_excluded
                                )
                            ),
                        }
                    )
                    audit_pairs.append(
                        (
                            None if left is None else left.event_id,
                            None if right is None else right.event_id,
                        )
                    )
            else:
                relation_units.append(
                    {
                        "ethereum": "NO_ANCHOR",
                        "bitcoin": "NO_ANCHOR",
                        "counterpart_state": "NO_ANCHOR",
                        "memorization_excluded": False,
                    }
                )
                audit_pairs.append((None, None))
            if len(relation_units) > prereg.MAX_MODEL_EVENTS_PER_CARD:
                raise ValueError("PSIM relation card exceeds frozen event bound")
            for event in new_events:
                latest[event.protocol] = event
                history[event.protocol].append(event)
            protocol_state: dict[str, Any] = {}
            for protocol in ("ethereum", "bitcoin"):
                latest_event = latest[protocol]
                if latest_event is None:
                    protocol_state[protocol] = {
                        "new_event_state": (
                            "NEW_EVENT"
                            if any(
                                row.protocol == protocol for row in new_events
                            )
                            else "NO_NEW_EVENT"
                        ),
                        "stale_age_bucket": "NO_EVENT_YET",
                    }
                else:
                    stale_days = (
                        decision_at.date()
                        - latest_event.available_at[schedule].date()
                    ).days
                    protocol_state[protocol] = {
                        "new_event_state": (
                            "NEW_EVENT"
                            if any(
                                row.protocol == protocol for row in new_events
                            )
                            else "NO_NEW_EVENT"
                        ),
                        "stale_age_bucket": bucket_token(
                            "stale_age_days",
                            stale_days,
                        ),
                    }
            local_payload = {
                "protocol_state": protocol_state,
                "new_events": [
                    _event_payload(row, decision_at, schedule)
                    for row in new_events
                ],
                "relation_units": relation_units,
            }
            local_sha = canonical_hash(local_payload)
            card_hash = canonical_hash(
                {
                    "schedule": schedule,
                    "decision_at": format_time(decision_at),
                    "prior_card_hash": prior_hash,
                    "local_payload_sha256": local_sha,
                }
            )
            eligible = {
                "protocol_label_swap": any(
                    left is not None or right is not None
                    for left, right in audit_pairs
                ),
                "within_day_event_order_reverse": (
                    len(ethereum_new) >= 2 or len(bitcoin_new) >= 2
                ),
                "proposal_version_pair_cyclic_permutation": any(
                    row.event_id in version_eligible for row in new_events
                ),
                "old_new_direction_reverse": bool(new_events),
                "section_label_cyclic_rotation": any(
                    any(section != "OTHER" for section in row.changed_sections)
                    for row in new_events
                ),
                "dependency_edge_direction_reverse": any(
                    row.dependency_delta_state in {"ADDED", "REMOVED"}
                    for row in new_events
                ),
                "availability_plus_seven_days": bool(new_events),
            }
            cards.append(
                DailyCard(
                    schedule=schedule,
                    decision_at=decision_at,
                    split=split_name(decision_at),
                    local_payload=local_payload,
                    local_payload_sha256=local_sha,
                    prior_card_hash=prior_hash,
                    card_hash=card_hash,
                    event_ids=tuple(row.event_id for row in new_events),
                    eligible=eligible,
                )
            )
            prior_hash = card_hash
    if ledger is not None:
        ledger.daily_cards_built += len(cards)
    return cards


def _swap_add_remove_text(value: str) -> str:
    rows: list[str] = []
    for line in value.splitlines():
        parts = line.split("|", 2)
        if len(parts) != 3:
            raise ValueError("PSIM intent text line is malformed")
        direction = (
            "REMOVE"
            if parts[1] == "ADD"
            else "ADD"
            if parts[1] == "REMOVE"
            else parts[1]
        )
        rows.append("|".join((parts[0], direction, parts[2])))
    return "\n".join(rows)


def _rotate_section_text(value: str) -> str:
    rows: list[str] = []
    for line in value.splitlines():
        parts = line.split("|", 2)
        if len(parts) != 3:
            raise ValueError("PSIM intent text line is malformed")
        section = SECTION_ROTATION.get(parts[0], parts[0])
        rows.append("|".join((section, parts[1], parts[2])))
    return "\n".join(rows)


def _intent_lines(value: str, direction: str) -> list[str]:
    return [
        line
        for line in value.splitlines()
        if len(line.split("|", 2)) == 3
        and line.split("|", 2)[1] == direction
    ]


def transform_events(
    events: Sequence[ProposalEvent],
    control: str,
) -> list[ProposalEvent]:
    if control not in prereg.RELATION_CONTROLS:
        raise ValueError(f"unknown PSIM control: {control}")
    if control == "protocol_label_swap":
        return [
            replace(
                event,
                protocol=(
                    "bitcoin" if event.protocol == "ethereum" else "ethereum"
                ),
            )
            for event in events
        ]
    if control == "within_day_event_order_reverse":
        return [
            replace(event, control_order=-event.first_parent_index)
            for event in events
        ]
    if control == "proposal_version_pair_cyclic_permutation":
        transformed = list(events)
        by_stratum: dict[tuple[str, int], list[int]] = defaultdict(list)
        for index, event in enumerate(transformed):
            if event.event_type == "UPDATE":
                by_stratum[(event.protocol, event.effective_day.year)].append(index)
        for indexes in by_stratum.values():
            indexes.sort(key=lambda index: transformed[index].event_id)
            if len(indexes) < 2:
                continue
            donors = indexes[1:] + indexes[:1]
            originals = list(transformed)
            for target_index, donor_index in zip(indexes, donors, strict=True):
                target = originals[target_index]
                donor = originals[donor_index]
                intent = "\n".join(
                    _intent_lines(target.intent_text, "ADD")
                    + _intent_lines(donor.intent_text, "REMOVE")
                )
                sections = tuple(
                    sorted(
                        {
                            line.split("|", 1)[0]
                            for line in intent.splitlines()
                            if "|" in line
                        }
                    )
                )
                transformed[target_index] = replace(
                    target,
                    old_blob_sha256=donor.old_blob_sha256,
                    old_sections=donor.old_sections,
                    changed_sections=sections,
                    changed_section_count=len(sections),
                    dependency_delta_state=donor.dependency_delta_state,
                    dependency_edge_delta_count=(
                        donor.dependency_edge_delta_count
                    ),
                    intent_text=intent,
                )
        return transformed
    if control == "old_new_direction_reverse":
        event_type_map = {
            "CREATE": "DELETE",
            "UPDATE": "UPDATE",
            "DELETE": "CREATE",
        }
        dependency_map = {
            "NO_PRIOR": "DELETED",
            "DELETED": "NO_PRIOR",
            "ADDED": "REMOVED",
            "REMOVED": "ADDED",
            "STABLE": "STABLE",
            "MIXED": "MIXED",
        }
        return [
            replace(
                event,
                event_type=event_type_map[event.event_type],
                old_path=event.new_path,
                new_path=event.old_path,
                old_blob_oid=event.new_blob_oid,
                new_blob_oid=event.old_blob_oid,
                old_blob_sha256=event.new_blob_sha256,
                new_blob_sha256=event.old_blob_sha256,
                old_sections=event.new_sections,
                new_sections=event.old_sections,
                dependency_delta_state=dependency_map[
                    event.dependency_delta_state
                ],
                intent_text=_swap_add_remove_text(event.intent_text),
            )
            for event in events
        ]
    if control == "section_label_cyclic_rotation":
        return [
            replace(
                event,
                old_sections=tuple(
                    sorted(
                        SECTION_ROTATION.get(value, value)
                        for value in event.old_sections
                    )
                ),
                new_sections=tuple(
                    sorted(
                        SECTION_ROTATION.get(value, value)
                        for value in event.new_sections
                    )
                ),
                changed_sections=tuple(
                    sorted(
                        SECTION_ROTATION.get(value, value)
                        for value in event.changed_sections
                    )
                ),
                intent_text=_rotate_section_text(event.intent_text),
            )
            for event in events
        ]
    if control == "dependency_edge_direction_reverse":
        dependency_map = {
            "ADDED": "REMOVED",
            "REMOVED": "ADDED",
        }
        return [
            replace(
                event,
                dependency_delta_state=dependency_map.get(
                    event.dependency_delta_state,
                    event.dependency_delta_state,
                ),
            )
            for event in events
        ]
    if control == "availability_plus_seven_days":
        return [
            replace(
                event,
                available_at={
                    key: value + timedelta(days=7)
                    for key, value in event.available_at.items()
                },
            )
            for event in events
        ]
    raise AssertionError("unreachable PSIM control")


def _card_map(
    cards: Sequence[DailyCard],
) -> dict[tuple[str, str], DailyCard]:
    mapping = {
        (card.schedule, format_time(card.decision_at)): card for card in cards
    }
    if len(mapping) != len(cards):
        raise RuntimeError("PSIM card schedule contains duplicate decisions")
    return mapping


def build_control_metrics(
    events: Sequence[ProposalEvent],
    baseline_cards: Sequence[DailyCard],
) -> dict[str, Any]:
    baseline = _card_map(baseline_cards)
    results: dict[str, Any] = {}
    for control in prereg.RELATION_CONTROLS:
        controlled_cards = build_daily_cards(transform_events(events, control))
        controlled = _card_map(controlled_cards)
        if set(controlled) != set(baseline):
            raise RuntimeError("PSIM control card schedule differs from baseline")
        cells: dict[str, Any] = {}
        for schedule in (row.name for row in prereg.ARCHIVE_SCHEDULES):
            for split in ("train", "test", "eval"):
                eligible = 0
                changed = 0
                for key, base_card in baseline.items():
                    if key[0] != schedule or base_card.split != split:
                        continue
                    control_card = controlled[key]
                    is_eligible = bool(base_card.eligible[control])
                    if control == "availability_plus_seven_days":
                        is_eligible = is_eligible or bool(control_card.event_ids)
                    if not is_eligible:
                        continue
                    eligible += 1
                    changed += int(
                        base_card.local_payload_sha256
                        != control_card.local_payload_sha256
                    )
                passed = (
                    eligible >= 4
                    and changed * 10 >= eligible
                )
                cells[f"{schedule}:{split}"] = {
                    "eligible": eligible,
                    "changed": changed,
                    "changed_fraction": f"{changed}/{eligible}",
                    "passed": passed,
                }
        results[control] = {
            "transform": prereg.CONTROL_TRANSFORMS[control],
            "eligibility": prereg.CONTROL_ELIGIBILITY[control],
            "cells": cells,
            "passed": all(row["passed"] for row in cells.values()),
        }
    return results


def gate_control_sensitivity(metrics: Mapping[str, Any]) -> GateResult:
    failures: list[str] = []
    expected_cells = {
        f"{schedule.name}:{split}"
        for schedule in prereg.ARCHIVE_SCHEDULES
        for split in ("train", "test", "eval")
    }
    if tuple(metrics) != tuple(prereg.RELATION_CONTROLS):
        failures.append("control_roster")
    for control in prereg.RELATION_CONTROLS:
        control_row = metrics.get(control)
        if not isinstance(control_row, Mapping):
            failures.append(f"{control}:missing")
            continue
        cells = control_row.get("cells")
        if not isinstance(cells, Mapping) or set(cells) != expected_cells:
            failures.append(f"{control}:cell_roster")
            continue
        computed_passes: list[bool] = []
        for cell in sorted(expected_cells):
            row = cells[cell]
            valid = (
                isinstance(row, Mapping)
                and type(row.get("eligible")) is int
                and type(row.get("changed")) is int
                and row["eligible"] >= 0
                and 0 <= row["changed"] <= row["eligible"]
                and row.get("changed_fraction")
                == f"{row['changed']}/{row['eligible']}"
            )
            computed = bool(
                valid
                and row["eligible"] >= 4
                and row["changed"] * 10 >= row["eligible"]
            )
            computed_passes.append(computed)
            if not valid or row.get("passed") is not computed or not computed:
                failures.append(f"{control}:{cell}")
        if control_row.get("passed") is not all(computed_passes):
            failures.append(f"{control}:aggregate")
    return GateResult(
        name="relation_control_sensitivity",
        passed=not failures,
        metrics={
            "controls": metrics,
            "failed_cells": failures,
        },
        failure=";".join(failures),
    )


def event_row(event: ProposalEvent) -> dict[str, Any]:
    return {
        "protocol": event.protocol,
        "proposal_number": event.proposal_number,
        "commit_oid": event.commit_oid,
        "first_parent_index": event.first_parent_index,
        "committer_day": event.committer_day.isoformat(),
        "effective_day": event.effective_day.isoformat(),
        "event_type": event.event_type,
        "event_id": event.event_id,
        "old_path": event.old_path,
        "new_path": event.new_path,
        "old_blob_oid": event.old_blob_oid,
        "new_blob_oid": event.new_blob_oid,
        "old_blob_sha256": event.old_blob_sha256,
        "new_blob_sha256": event.new_blob_sha256,
        "old_blob_role": event.old_blob_role,
        "prior_dependency_state": event.prior_dependency_state,
        "old_sections": list(event.old_sections),
        "new_sections": list(event.new_sections),
        "changed_sections": list(event.changed_sections),
        "dependency_delta_state": event.dependency_delta_state,
        "dependency_edge_delta_count": event.dependency_edge_delta_count,
        "line_change_count": event.line_change_count,
        "changed_section_count": event.changed_section_count,
        "window_revision_count": event.window_revision_count,
        "window_age_days": event.window_age_days,
        "update_gap_days": event.update_gap_days,
        "intent_text": event.intent_text,
        "memorization_excluded": event.memorization_excluded,
        "available_at": {
            key: format_time(value)
            for key, value in sorted(event.available_at.items())
        },
    }


def card_row(card: DailyCard) -> dict[str, Any]:
    return {
        "schedule": card.schedule,
        "decision_at": format_time(card.decision_at),
        "split": card.split,
        "local_payload": card.local_payload,
        "local_payload_sha256": card.local_payload_sha256,
        "prior_card_hash": card.prior_card_hash,
        "card_hash": card.card_hash,
        "event_ids": list(card.event_ids),
        "eligible": dict(card.eligible),
    }


def rows_fingerprint(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    raw = jsonl_bytes(rows)
    return {
        "rows": len(rows),
        "canonical_jsonl_sha256": sha256_bytes(raw),
        "ordered_identity_sha256": canonical_hash(
            [
                row.get("event_id")
                or (
                    f"{row.get('schedule')}:{row.get('decision_at')}:"
                    f"{row.get('card_hash')}"
                )
                for row in rows
            ]
        ),
    }


def _event_split(event: ProposalEvent) -> str | None:
    stamp = datetime.combine(event.effective_day, time.min, tzinfo=UTC)
    return split_name(stamp)


def split_support_metrics(
    events: Sequence[ProposalEvent],
    cards: Sequence[DailyCard],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    d90_cards = [
        card
        for card in cards
        if card.schedule == "ARCHIVE_D90" and card.split is not None
    ]
    for split in prereg.SPLITS:
        name = str(split["name"])
        split_events = [event for event in events if _event_split(event) == name]
        protocol_counts = Counter(event.protocol for event in split_events)
        proposal_counts = Counter(
            (event.protocol, event.proposal_number) for event in split_events
        )
        day_counts = Counter(event.effective_day for event in split_events)
        per_year = Counter(
            (event.protocol, event.effective_day.year)
            for event in split_events
        )
        unique_proposals = {
            (event.protocol, event.proposal_number) for event in split_events
        }
        unique_by_protocol = {
            protocol: len(
                {
                    event.proposal_number
                    for event in split_events
                    if event.protocol == protocol
                }
            )
            for protocol in ("ethereum", "bitcoin")
        }
        unique_days = {
            protocol: len(
                {
                    event.effective_day
                    for event in split_events
                    if event.protocol == protocol
                }
            )
            for protocol in ("ethereum", "bitcoin")
        }
        active_months = {
            protocol: len(
                {
                    (event.effective_day.year, event.effective_day.month)
                    for event in split_events
                    if event.protocol == protocol
                }
            )
            for protocol in ("ethereum", "bitcoin")
        }
        active_quarters = {
            protocol: len(
                {
                    (
                        event.effective_day.year,
                        (event.effective_day.month - 1) // 3 + 1,
                    )
                    for event in split_events
                    if event.protocol == protocol
                }
            )
            for protocol in ("ethereum", "bitcoin")
        }
        split_cards = [card for card in d90_cards if card.split == name]
        relation_units = 0
        nonexcluded = 0
        counterparts = 0
        for card in split_cards:
            for unit in card.local_payload["relation_units"]:
                relation_units += 1
                if (
                    not unit["memorization_excluded"]
                    and unit["counterpart_state"] != "NO_ANCHOR"
                ):
                    nonexcluded += 1
                    if unit["counterpart_state"] not in {
                        "NO_COUNTERPART",
                    }:
                        counterparts += 1
        metrics[name] = {
            "events_total": len(split_events),
            "events_per_protocol": dict(sorted(protocol_counts.items())),
            "events_per_protocol_source_year": {
                f"{protocol}:{year}": per_year[(protocol, year)]
                for protocol in ("ethereum", "bitcoin")
                for year in range(
                    parse_time(str(split["decision_start"])).year,
                    parse_time(str(split["decision_end_exclusive"])).year,
                )
            },
            "unique_proposals_total": len(unique_proposals),
            "unique_proposals_per_protocol": unique_by_protocol,
            "unique_event_days_per_protocol": unique_days,
            "active_months_per_protocol": active_months,
            "active_quarters_per_protocol": active_quarters,
            "top_proposal_event_count": max(proposal_counts.values(), default=0),
            "top_event_day_count": max(day_counts.values(), default=0),
            "relation_units": relation_units,
            "relation_units_nonexcluded": nonexcluded,
            "relation_units_with_counterpart_nonexcluded": counterparts,
            "d90_daily_cards": len(split_cards),
        }
    return metrics


def gate_split_support(metrics: Mapping[str, Any]) -> GateResult:
    failures: list[str] = []
    checks: dict[str, Any] = {}
    for split in prereg.SPLITS:
        name = str(split["name"])
        row = metrics[name]
        minimum_counterpart = Fraction(
            str(split["minimum_counterpart_fraction"])
        )
        maximum_proposal = Fraction(
            str(split["maximum_top_proposal_event_share"])
        )
        maximum_day = Fraction(
            str(split["maximum_top_event_day_share"])
        )
        per_protocol = row["events_per_protocol"]
        per_year = row["events_per_protocol_source_year"]
        checks[name] = {
            "events_total": (
                row["events_total"] >= split["minimum_events_total"]
            ),
            "events_per_protocol": all(
                int(per_protocol.get(protocol, 0))
                >= split["minimum_events_per_protocol"]
                for protocol in ("ethereum", "bitcoin")
            ),
            "events_per_protocol_per_source_year": all(
                int(value)
                >= split["minimum_events_per_protocol_per_source_year"]
                for value in per_year.values()
            ),
            "unique_proposals_total": (
                row["unique_proposals_total"]
                >= split["minimum_unique_proposals_total"]
            ),
            "unique_proposals_per_protocol": all(
                int(row["unique_proposals_per_protocol"][protocol])
                >= split["minimum_unique_proposals_per_protocol"]
                for protocol in ("ethereum", "bitcoin")
            ),
            "unique_event_days_per_protocol": all(
                int(row["unique_event_days_per_protocol"][protocol])
                >= split["minimum_unique_event_days_per_protocol"]
                for protocol in ("ethereum", "bitcoin")
            ),
            "active_months_per_protocol": all(
                int(row["active_months_per_protocol"][protocol])
                >= split["minimum_active_months_per_protocol"]
                for protocol in ("ethereum", "bitcoin")
            ),
            "active_quarters_per_protocol": all(
                int(row["active_quarters_per_protocol"][protocol])
                >= split["minimum_active_quarters_per_protocol"]
                for protocol in ("ethereum", "bitcoin")
            ),
            "relation_units_nonexcluded": (
                row["relation_units_nonexcluded"]
                >= split["minimum_relation_units_nonexcluded"]
            ),
            "counterpart_fraction": (
                row["relation_units_nonexcluded"] > 0
                and row["relation_units_with_counterpart_nonexcluded"]
                * minimum_counterpart.denominator
                >= row["relation_units_nonexcluded"]
                * minimum_counterpart.numerator
            ),
            "top_proposal_event_share": (
                row["events_total"] > 0
                and row["top_proposal_event_count"]
                * maximum_proposal.denominator
                <= row["events_total"] * maximum_proposal.numerator
            ),
            "top_event_day_share": (
                row["events_total"] > 0
                and row["top_event_day_count"] * maximum_day.denominator
                <= row["events_total"] * maximum_day.numerator
            ),
        }
        failures.extend(
            f"{name}:{key}"
            for key, passed in checks[name].items()
            if not passed
        )
    return GateResult(
        name="split_annual_quarterly_unique_day_support",
        passed=not failures,
        metrics={"support": metrics, "checks": checks},
        failure=";".join(failures),
    )


def gate_vocabulary(events: Sequence[ProposalEvent]) -> GateResult:
    event_types = Counter(event.event_type for event in events)
    sections = Counter(
        section for event in events for section in event.changed_sections
    )
    dependency = Counter(event.dependency_delta_state for event in events)
    revision_tokens = Counter(
        bucket_token("window_revision_count", event.window_revision_count)
        for event in events
    )
    nonquarantine = Counter(
        event.protocol for event in events if not event.memorization_excluded
    )
    maximum_section_count = max(sections.values(), default=0)
    total_section_count = sum(sections.values())
    checks = {
        "event_types": (
            len(event_types)
            >= prereg.build_preregistration()["source_support_contract"][
                "minimum_event_types_overall"
            ]
        ),
        "section_categories": (
            len(sections)
            >= prereg.build_preregistration()["source_support_contract"][
                "minimum_section_categories_overall"
            ]
        ),
        "dependency_categories": (
            len(dependency)
            >= prereg.build_preregistration()["source_support_contract"][
                "minimum_dependency_delta_categories_overall"
            ]
        ),
        "revision_buckets": (
            len(revision_tokens)
            >= prereg.build_preregistration()["source_support_contract"][
                "minimum_revision_buckets_overall"
            ]
        ),
        "top_changed_section_share": (
            total_section_count > 0
            and maximum_section_count * 100 <= total_section_count * 80
        ),
        "nonquarantined_per_protocol": all(
            nonquarantine[protocol]
            >= prereg.build_preregistration()["source_support_contract"][
                "minimum_non_quarantined_events_per_protocol"
            ]
            for protocol in ("ethereum", "bitcoin")
        ),
    }
    return GateResult(
        name="event_section_dependency_revision_vocabulary_diversity",
        passed=all(checks.values()),
        metrics={
            "checks": checks,
            "event_types": dict(sorted(event_types.items())),
            "changed_sections": dict(sorted(sections.items())),
            "dependency_delta_states": dict(sorted(dependency.items())),
            "revision_buckets": dict(sorted(revision_tokens.items())),
            "nonquarantined_events": dict(sorted(nonquarantine.items())),
        },
        failure=";".join(key for key, value in checks.items() if not value),
    )


def gate_daily_cards(cards: Sequence[DailyCard]) -> GateResult:
    expected_decisions = _decision_times()
    expected_days = len(expected_decisions)
    expected_decision_set = set(expected_decisions)
    by_schedule = Counter(card.schedule for card in cards)
    failures: list[str] = []
    checks: dict[str, Any] = {}
    for schedule in (row.name for row in prereg.ARCHIVE_SCHEDULES):
        rows = [card for card in cards if card.schedule == schedule]
        chain_ok = True
        prior = canonical_hash(
            {"schedule": schedule, "state": "PSIM_CARD_CHAIN_START"}
        )
        for card in rows:
            expected_local_sha = canonical_hash(card.local_payload)
            expected_card_hash = canonical_hash(
                {
                    "schedule": schedule,
                    "decision_at": format_time(card.decision_at),
                    "prior_card_hash": prior,
                    "local_payload_sha256": expected_local_sha,
                }
            )
            if (
                card.prior_card_hash != prior
                or card.local_payload_sha256 != expected_local_sha
                or card.card_hash != expected_card_hash
            ):
                chain_ok = False
                break
            prior = card.card_hash
        explicit_states = all(
            set(card.local_payload["protocol_state"]) == {
                "ethereum",
                "bitcoin",
            }
            and card.local_payload["relation_units"]
            for card in rows
        )
        checks[schedule] = {
            "decision_count": len(rows) == expected_days,
            "chain": chain_ok,
            "explicit_states": explicit_states,
            "unique_decisions": (
                len({card.decision_at for card in rows}) == len(rows)
            ),
            "exact_decision_roster": (
                {card.decision_at for card in rows}
                == expected_decision_set
            ),
            "decision_split_identity": all(
                card.split == split_name(card.decision_at)
                for card in rows
            ),
        }
        failures.extend(
            f"{schedule}:{key}"
            for key, passed in checks[schedule].items()
            if not passed
        )
    return GateResult(
        name="daily_card_coverage_and_explicit_staleness",
        passed=not failures,
        metrics={
            "expected_days_per_schedule": expected_days,
            "cards_per_schedule": dict(sorted(by_schedule.items())),
            "checks": checks,
        },
        failure=";".join(failures),
    )


def _commit_rows(records: Sequence[CommitRecord]) -> list[dict[str, Any]]:
    return [
        {
            "protocol": row.protocol,
            "oid": row.oid,
            "tree_oid": row.tree_oid,
            "parent_oid": row.parent_oid,
            "first_parent_index": row.first_parent_index,
            "committer_epoch": row.committer_epoch,
            "committer_day": row.committer_day.isoformat(),
            "effective_day": row.effective_day.isoformat(),
        }
        for row in records
    ]


def _group_rows(groups: Sequence[ProposalGroup]) -> list[dict[str, Any]]:
    return [
        {
            **asdict(row),
            "committer_day": row.committer_day.isoformat(),
            "effective_day": row.effective_day.isoformat(),
        }
        for row in groups
    ]


def gate_git_identity(receipts: Sequence[Mapping[str, Any]]) -> GateResult:
    expected = {
        (protocol, replica)
        for protocol in ("ethereum", "bitcoin")
        for replica in ("a", "b")
    }
    observed = {
        (str(row["protocol"]), str(row["replica"])) for row in receipts
    }
    exact_identity = True
    for row in receipts:
        protocol = str(row["protocol"])
        if protocol not in {"ethereum", "bitcoin"}:
            exact_identity = False
            continue
        spec = _repository_spec(protocol)
        exact_identity = exact_identity and (
            row.get("remote") == spec.remote
            and row.get("remote_head_symref")
            == f"refs/heads/{spec.branch}"
            and isinstance(row.get("remote_head_oid"), str)
            and HEX40.fullmatch(str(row["remote_head_oid"])) is not None
            and row.get("local_tracking_symref")
            == spec.remote_head_symref
            and row.get("sealed_tip") == spec.sealed_tip
            and row.get("object_format") == spec.object_format
        )
    checks = {
        "four_independent_roots": observed == expected and len(receipts) == 4,
        "exact_frozen_repository_identity": exact_identity,
        "all_fsck": all(row["git_fsck_no_dangling"] for row in receipts),
        "no_shared_alternates": all(
            not row["shared_object_alternates"] for row in receipts
        ),
        "all_clean_no_checkout": all(
            row["worktree_porcelain_empty"] for row in receipts
        ),
        "disk_guard": all(
            int(row["disk_used_gib"]) <= prereg.DISK_LIMIT_GIB
            for row in receipts
        ),
    }
    return GateResult(
        name="sealed_git_identity_and_object_integrity",
        passed=all(checks.values()),
        metrics={"checks": checks, "receipts": list(receipts)},
        failure=";".join(key for key, value in checks.items() if not value),
    )


def gate_commit_chains(
    chains: Mapping[tuple[str, str], Sequence[CommitRecord]],
) -> GateResult:
    comparisons: dict[str, Any] = {}
    failures: list[str] = []
    for protocol in ("ethereum", "bitcoin"):
        left = _commit_rows(chains[(protocol, "a")])
        right = _commit_rows(chains[(protocol, "b")])
        left_hash = canonical_hash(left)
        right_hash = canonical_hash(right)
        records = chains[(protocol, "a")]
        monotonic = all(
            earlier.effective_day <= later.effective_day
            for earlier, later in zip(
                records,
                records[1:],
            )
        )
        continuity = bool(records) and (
            records[0].parent_oid is None
            and all(
                later.parent_oid == earlier.oid
                for earlier, later in zip(records, records[1:])
            )
            and [row.first_parent_index for row in records]
            == list(range(len(records)))
            and records[-1].oid == _repository_spec(protocol).sealed_tip
        )
        passed = (
            left_hash == right_hash
            and monotonic
            and continuity
            and bool(left)
        )
        comparisons[protocol] = {
            "records": len(left),
            "replica_a_sha256": left_hash,
            "replica_b_sha256": right_hash,
            "effective_day_monotonic": monotonic,
            "first_parent_continuity": continuity,
            "passed": passed,
        }
        if not passed:
            failures.append(protocol)
    return GateResult(
        name="first_parent_traversal_and_causal_clock",
        passed=not failures,
        metrics=comparisons,
        failure=";".join(failures),
    )


def gate_proposal_groups(
    groups: Mapping[tuple[str, str], Sequence[ProposalGroup]],
    issues: Mapping[tuple[str, str], Sequence[str]],
) -> GateResult:
    comparisons: dict[str, Any] = {}
    failures: list[str] = []
    for protocol in ("ethereum", "bitcoin"):
        left = _group_rows(groups[(protocol, "a")])
        right = _group_rows(groups[(protocol, "b")])
        left_issues = list(issues[(protocol, "a")])
        right_issues = list(issues[(protocol, "b")])
        identity_ok = all(
            row.protocol == protocol
            and SOURCE_START.date()
            <= row.effective_day
            < SOURCE_END_EXCLUSIVE.date()
            and row.event_type in {"CREATE", "UPDATE", "DELETE"}
            and row.event_id
            == _event_id(
                row.protocol,
                row.commit_oid,
                row.proposal_number,
                row.old_blob_oid,
                row.new_blob_oid,
            )
            for row in groups[(protocol, "a")]
        )
        passed = (
            bool(left)
            and left == right
            and identity_ok
            and not left_issues
            and not right_issues
        )
        comparisons[protocol] = {
            "groups": len(left),
            "replica_a_sha256": canonical_hash(left),
            "replica_b_sha256": canonical_hash(right),
            "replica_a_issues": left_issues,
            "replica_b_issues": right_issues,
            "event_identity_and_source_interval": identity_ok,
            "passed": passed,
        }
        if not passed:
            failures.append(protocol)
    return GateResult(
        name="path_object_grammar_and_unique_proposal_tree",
        passed=not failures,
        metrics=comparisons,
        failure=";".join(failures),
    )


def gate_event_parser_replay(
    events: Mapping[tuple[str, str], Sequence[ProposalEvent]],
) -> GateResult:
    comparisons: dict[str, Any] = {}
    failures: list[str] = []
    for protocol in ("ethereum", "bitcoin"):
        left = [event_row(row) for row in events[(protocol, "a")]]
        right = [event_row(row) for row in events[(protocol, "b")]]
        identity_ok = all(
            row.protocol == protocol
            and SOURCE_START.date()
            <= row.effective_day
            < SOURCE_END_EXCLUSIVE.date()
            and row.event_type in {"CREATE", "UPDATE", "DELETE"}
            and row.event_id
            == _event_id(
                row.protocol,
                row.commit_oid,
                row.proposal_number,
                row.old_blob_oid,
                row.new_blob_oid,
            )
            for row in events[(protocol, "a")]
        )
        passed = bool(left) and left == right and identity_ok
        comparisons[protocol] = {
            "events": len(left),
            "replica_a_sha256": sha256_bytes(jsonl_bytes(left)),
            "replica_b_sha256": sha256_bytes(jsonl_bytes(right)),
            "event_identity_and_source_interval": identity_ok,
            "passed": passed,
        }
        if not passed:
            failures.append(protocol)
    return GateResult(
        name="historical_blob_preamble_dependency_integrity",
        passed=not failures,
        metrics=comparisons,
        failure=";".join(failures),
    )


def gate_independent_replay(
    events: Mapping[tuple[str, str], Sequence[ProposalEvent]],
    cards_a: Sequence[DailyCard],
    cards_b: Sequence[DailyCard],
) -> GateResult:
    event_rows_a = [
        event_row(row)
        for protocol in ("ethereum", "bitcoin")
        for row in events[(protocol, "a")]
    ]
    event_rows_b = [
        event_row(row)
        for protocol in ("ethereum", "bitcoin")
        for row in events[(protocol, "b")]
    ]
    card_rows_a = [card_row(row) for row in cards_a]
    card_rows_b = [card_row(row) for row in cards_b]
    checks = {
        "events": event_rows_a == event_rows_b,
        "cards": card_rows_a == card_rows_b,
    }
    return GateResult(
        name="independent_replay_and_canonical_manifest_identity",
        passed=all(checks.values()),
        metrics={
            "checks": checks,
            "replica_a_events": rows_fingerprint(event_rows_a),
            "replica_b_events": rows_fingerprint(event_rows_b),
            "replica_a_cards": rows_fingerprint(card_rows_a),
            "replica_b_cards": rows_fingerprint(card_rows_b),
        },
        failure=";".join(key for key, value in checks.items() if not value),
    )


def gate_future_append(
    events: Sequence[ProposalEvent],
    baseline_cards: Sequence[DailyCard],
) -> GateResult:
    if not events:
        return GateResult(
            name="future_append_invariance",
            passed=False,
            metrics={"events": 0},
            failure="no source event available",
        )
    seed = events[0]
    sentinel_day = date(2024, 1, 2)
    if seed.protocol == "bitcoin":
        used_numbers = {
            row.proposal_number
            for row in events
            if row.protocol == "bitcoin"
        }
        sentinel_number = next(
            (
                candidate
                for candidate in range(9_999, 0, -1)
                if candidate not in used_numbers
            ),
            0,
        )
        if sentinel_number == 0:
            raise RuntimeError("PSIM has no unused four-digit BIP sentinel")
    else:
        sentinel_number = (
            max(
                row.proposal_number
                for row in events
                if row.protocol == "ethereum"
            )
            + 100_000
        )
    sentinel = replace(
        seed,
        proposal_number=sentinel_number,
        commit_oid="f" * 40,
        first_parent_index=max(row.first_parent_index for row in events) + 1,
        committer_day=sentinel_day,
        effective_day=sentinel_day,
        event_type="CREATE",
        event_id=_event_id(
            seed.protocol,
            "f" * 40,
            sentinel_number,
            None,
            "e" * 40,
        ),
        old_path=None,
        new_path=(
            f"EIPS/eip-{sentinel_number}.md"
            if seed.protocol == "ethereum"
            else f"bip-{sentinel_number:04d}.mediawiki"
        ),
        old_blob_oid=None,
        new_blob_oid="e" * 40,
        old_blob_sha256=None,
        new_blob_sha256="d" * 64,
        old_sections=(),
        new_sections=("ABSTRACT", "MOTIVATION"),
        changed_sections=("ABSTRACT", "MOTIVATION"),
        dependency_delta_state="NO_PRIOR",
        dependency_edge_delta_count=0,
        line_change_count=2,
        changed_section_count=2,
        window_revision_count=0,
        window_age_days=0,
        update_gap_days=None,
        intent_text=(
            "ABSTRACT|ADD|synthetic future append\n"
            "MOTIVATION|ADD|synthetic future append"
        ),
        memorization_excluded=False,
        available_at=_available_at(sentinel_day),
    )
    appended_cards = build_daily_cards([*events, sentinel])
    baseline = {
        (row.schedule, row.decision_at): row.local_payload_sha256
        for row in baseline_cards
        if row.decision_at < SOURCE_END_EXCLUSIVE
    }
    appended = {
        (row.schedule, row.decision_at): row.local_payload_sha256
        for row in appended_cards
        if row.decision_at < SOURCE_END_EXCLUSIVE
    }
    checks = {
        "pre_2024_event_rows_unchanged": [
            event_row(row) for row in events
        ]
        == [event_row(row) for row in [*events, sentinel][:-1]],
        "pre_2024_cards_unchanged": baseline == appended,
        "sentinel_outside_source": sentinel.effective_day >= date(2024, 1, 1),
        "sentinel_path_grammar": (
            sentinel.new_path is not None
            and _path_identity(seed.protocol, sentinel.new_path)
            == (
                sentinel_number,
                "md" if seed.protocol == "ethereum" else "mediawiki",
            )
        ),
    }
    return GateResult(
        name="future_append_invariance",
        passed=all(checks.values()),
        metrics={
            "checks": checks,
            "baseline_pre_2024_cards_sha256": canonical_hash(
                [
                    {
                        "schedule": schedule,
                        "decision_at": format_time(decision_at),
                        "local_payload_sha256": payload_sha256,
                    }
                    for (schedule, decision_at), payload_sha256 in sorted(
                        baseline.items()
                    )
                ]
            ),
            "appended_pre_2024_cards_sha256": canonical_hash(
                [
                    {
                        "schedule": schedule,
                        "decision_at": format_time(decision_at),
                        "local_payload_sha256": payload_sha256,
                    }
                    for (schedule, decision_at), payload_sha256 in sorted(
                        appended.items()
                    )
                ]
            ),
        },
        failure=";".join(key for key, value in checks.items() if not value),
    )


def gate_pairing_reset_quarantine(
    events: Sequence[ProposalEvent],
    cards: Sequence[DailyCard],
) -> GateResult:
    first_by_identity: dict[tuple[str, int], ProposalEvent] = {}
    for event in events:
        first_by_identity.setdefault(
            (event.protocol, event.proposal_number),
            event,
        )
    reset_ok = all(
        event.window_revision_count == 0
        and event.window_age_days == 0
        and event.update_gap_days is None
        and event.prior_dependency_state == "PRE_WINDOW_UNKNOWN"
        and event.old_blob_role
        == (
            "NO_OLD_BLOB"
            if event.old_blob_oid is None
            else "PRE_WINDOW_BASELINE"
        )
        for event in first_by_identity.values()
    )
    later_boundary_ok = all(
        event.old_blob_role != "PRE_WINDOW_BASELINE"
        and event.prior_dependency_state == "IN_WINDOW_KNOWN"
        for event in events
        if event is not first_by_identity[
            (event.protocol, event.proposal_number)
        ]
    )
    quarantine_ok = all(
        event.memorization_excluded
        == (
            event.proposal_number
            in prereg.MEMORIZATION_QUARANTINE[event.protocol]
        )
        for event in events
    )
    availability_ok = all(
        (
            event.available_at[row.name].date() - event.effective_day
        ).days
        == row.delay_calendar_days
        and event.available_at[row.name].time() == time(hour=12)
        for event in events
        for row in prereg.ARCHIVE_SCHEDULES
    )
    allowed_states = {
        "SAME_DAY_CARTESIAN",
        "TRAILING_90D",
        "NO_COUNTERPART",
        "NO_ANCHOR",
    }
    relation_ok = all(
        unit["counterpart_state"] in allowed_states
        for card in cards
        for unit in card.local_payload["relation_units"]
    )
    schedule_ok = {card.schedule for card in cards} == {
        row.name for row in prereg.ARCHIVE_SCHEDULES
    }
    checks = {
        "boundary_reset": reset_ok,
        "later_events_not_pre_window": later_boundary_ok,
        "memorization_quarantine": quarantine_ok,
        "four_schedule_delays": availability_ok,
        "relation_states": relation_ok,
        "schedule_roster": schedule_ok,
    }
    return GateResult(
        name="pairing_reset_quarantine_and_four_schedule_identity",
        passed=all(checks.values()),
        metrics={
            "checks": checks,
            "first_proposal_events": len(first_by_identity),
            "quarantined_events": sum(
                row.memorization_excluded for row in events
            ),
            "cards": len(cards),
        },
        failure=";".join(key for key, value in checks.items() if not value),
    )


def gate_forbidden_access(ledger: AccessLedger) -> GateResult:
    snapshot = ledger.snapshot()
    failures = [
        name for name in FORBIDDEN_ACCESS_FIELDS if snapshot[name] != 0
    ]
    return GateResult(
        name="forbidden_access_zero",
        passed=not failures,
        metrics={
            "ledger": snapshot,
            "forbidden_fields": list(FORBIDDEN_ACCESS_FIELDS),
        },
        failure=";".join(failures),
    )


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
        candidate = repository_path(path)
        if os.path.lexists(candidate):
            failures.append(f"exists:{path}")
        try:
            _safe_destination(path)
        except RuntimeError as exc:
            failures.append(str(exc))
    return GateResult(
        name="terminal_publication",
        passed=not failures,
        metrics={"paths": [str(path) for path in paths]},
        failure=";".join(failures),
    )


def _synthetic_event(
    *,
    protocol: str,
    proposal_number: int,
    effective_day: date,
    event_type: str,
    revision: int,
    first_parent_index: int,
) -> ProposalEvent:
    commit_oid = hashlib.sha1(
        (
            f"commit|{protocol}|{proposal_number}|"
            f"{effective_day.isoformat()}|{event_type}|{revision}"
        ).encode("ascii"),
        usedforsecurity=False,
    ).hexdigest()
    blob_oids = [
        hashlib.sha1(
            f"blob|{protocol}|{proposal_number}|{index}".encode("ascii"),
            usedforsecurity=False,
        ).hexdigest()
        for index in range(4)
    ]
    old_blob_oid = None if event_type == "CREATE" else blob_oids[revision - 1]
    new_blob_oid = None if event_type == "DELETE" else blob_oids[revision]
    event_id = _event_id(
        protocol,
        commit_oid,
        proposal_number,
        old_blob_oid,
        new_blob_oid,
    )
    dependency_state = {
        "CREATE": "NO_PRIOR",
        "UPDATE": "ADDED" if revision % 2 else "REMOVED",
        "DELETE": "DELETED",
    }[event_type]
    old_present = event_type != "CREATE"
    new_present = event_type != "DELETE"
    section = MODEL_SECTION_ORDER[
        (proposal_number + revision) % len(MODEL_SECTION_ORDER)
    ]
    direction = "REMOVE" if event_type == "DELETE" else "ADD"
    return ProposalEvent(
        protocol=protocol,
        proposal_number=proposal_number,
        commit_oid=commit_oid,
        first_parent_index=first_parent_index,
        committer_day=effective_day,
        effective_day=effective_day,
        event_type=event_type,
        event_id=event_id,
        old_path=(
            None
            if not old_present
            else (
                f"EIPS/eip-{proposal_number}.md"
                if protocol == "ethereum"
                else f"bip-{proposal_number:04d}.mediawiki"
            )
        ),
        new_path=(
            None
            if not new_present
            else (
                f"EIPS/eip-{proposal_number}.md"
                if protocol == "ethereum"
                else f"bip-{proposal_number:04d}.mediawiki"
            )
        ),
        old_blob_oid=old_blob_oid,
        new_blob_oid=new_blob_oid,
        old_blob_sha256=(
            None
            if old_blob_oid is None
            else sha256_bytes(f"raw:{old_blob_oid}".encode("ascii"))
        ),
        new_blob_sha256=(
            None
            if new_blob_oid is None
            else sha256_bytes(f"raw:{new_blob_oid}".encode("ascii"))
        ),
        old_blob_role=(
            "NO_OLD_BLOB"
            if old_blob_oid is None
            else (
                "PRE_WINDOW_BASELINE"
                if revision == 0
                else "IN_WINDOW_PRIOR"
            )
        ),
        prior_dependency_state=(
            "PRE_WINDOW_UNKNOWN"
            if revision == 0
            else "IN_WINDOW_KNOWN"
        ),
        old_sections=() if not old_present else (section,),
        new_sections=() if not new_present else (section,),
        changed_sections=(section,),
        dependency_delta_state=dependency_state,
        dependency_edge_delta_count=0 if event_type == "CREATE" else 1,
        line_change_count=revision + 1,
        changed_section_count=1,
        window_revision_count=revision,
        window_age_days=0,
        update_gap_days=None if revision == 0 else 0,
        intent_text=(
            f"{section}|{direction}|{protocol} synthetic intent "
            f"{proposal_number} revision {revision}"
        ),
        memorization_excluded=False,
        available_at=_available_at(effective_day),
    )


def synthetic_events() -> list[ProposalEvent]:
    rows: list[ProposalEvent] = []
    index = 0
    for year in range(2020, 2024):
        for month in range(1, 13):
            effective_day = date(year, month, 10)
            for protocol_offset, protocol in enumerate(("ethereum", "bitcoin")):
                proposal_number = (
                    5_000
                    + (year - 2020) * 100
                    + month * 2
                    + protocol_offset
                    + 1
                )
                for revision, event_type in enumerate(
                    ("CREATE", "UPDATE", "UPDATE", "DELETE")
                ):
                    rows.append(
                        _synthetic_event(
                            protocol=protocol,
                            proposal_number=proposal_number,
                            effective_day=effective_day,
                            event_type=event_type,
                            revision=revision,
                            first_parent_index=index,
                        )
                    )
                    index += 1
    return rows


def _synthetic_commit_raw(
    *,
    tree_oid: str,
    parent_oid: str | None,
    epoch: int,
) -> bytes:
    lines = [f"tree {tree_oid}"]
    if parent_oid is not None:
        lines.append(f"parent {parent_oid}")
    lines.extend(
        (
            "author Synthetic <synthetic@example.test> "
            f"{epoch} +0000",
            "committer Synthetic <synthetic@example.test> "
            f"{epoch} +0000",
        )
    )
    return ("\n".join(lines) + "\n\nsynthetic\n").encode("utf-8")


def build_self_check_manifest() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    events = synthetic_events()
    cards = build_daily_cards(events)
    controls = build_control_metrics(events, cards)
    split_metrics = split_support_metrics(events, cards)

    checks["synthetic_event_count"] = len(events) == 384
    checks["synthetic_protocols"] = {
        row.protocol for row in events
    } == {"ethereum", "bitcoin"}
    checks["synthetic_event_types"] = {
        row.event_type for row in events
    } == {"CREATE", "UPDATE", "DELETE"}
    checks["synthetic_event_identity"] = all(
        row.event_id
        == _event_id(
            row.protocol,
            row.commit_oid,
            row.proposal_number,
            row.old_blob_oid,
            row.new_blob_oid,
        )
        for row in events
    )
    checks["synthetic_event_path_grammar"] = all(
        all(
            path is None
            or (
                (identity := _path_identity(row.protocol, path)) is not None
                and identity[0] == row.proposal_number
            )
            for path in (row.old_path, row.new_path)
        )
        for row in events
    )
    checks["synthetic_card_count"] = (
        len(cards)
        == len(_decision_times()) * len(prereg.ARCHIVE_SCHEDULES)
    )
    checks["synthetic_all_controls_pass"] = all(
        row["passed"] for row in controls.values()
    )
    checks["synthetic_split_gate"] = gate_split_support(split_metrics).passed
    checks["synthetic_vocabulary_gate"] = gate_vocabulary(events).passed
    checks["synthetic_daily_gate"] = gate_daily_cards(cards).passed
    checks["synthetic_control_gate"] = gate_control_sensitivity(controls).passed
    checks["synthetic_future_append_gate"] = gate_future_append(
        events,
        cards,
    ).passed
    checks["synthetic_pairing_gate"] = gate_pairing_reset_quarantine(
        events,
        cards,
    ).passed
    checks["synthetic_forbidden_gate"] = gate_forbidden_access(
        AccessLedger.zero()
    ).passed

    eip = prereg.parse_eip_preamble(
        b"---\neip: 123\ntitle: Synthetic\nrequires: 1, 2\n---\n"
        b"# Abstract\nSynthetic\n"
    )
    bip = prereg.parse_bip_preamble(
        b"<pre>\n  BIP: 123\n  Title: Synthetic\n"
        b"  Requires: 1, 2\n</pre>\n== Abstract ==\nSynthetic\n"
    )
    checks["synthetic_eip_parser"] = eip["eip"] == "123"
    checks["synthetic_bip_parser"] = bip["bip"] == "123"
    checks["synthetic_dependency_parser"] = (
        prereg.parse_dependency_ids("1, 2", self_id=123) == (1, 2)
    )

    commit_body = _synthetic_commit_raw(
        tree_oid="1" * 40,
        parent_oid=None,
        epoch=1_577_836_800,
    )
    commit_oid = git_object_sha1("commit", commit_body)
    parsed_commit = parse_commit_object(
        "ethereum",
        commit_oid,
        commit_body,
        0,
        None,
    )
    checks["synthetic_commit_parser"] = (
        parsed_commit.tree_oid == "1" * 40
        and parsed_commit.effective_day == date(2020, 1, 1)
    )
    raw_delta = (
        b":000000 100644 "
        + ZERO_OID.encode("ascii")
        + b" "
        + b"2" * 40
        + b" A\x00EIPS/eip-123.md\x00"
    )
    parsed_delta = parse_raw_path_delta(raw_delta)
    checks["synthetic_path_delta_parser"] = (
        len(parsed_delta) == 1
        and parsed_delta[0].path == "EIPS/eip-123.md"
    )
    checks["synthetic_event_id_stable"] = (
        _event_id("ethereum", "1" * 40, 123, None, "2" * 40)
        == _event_id("ethereum", "1" * 40, 123, None, "2" * 40)
    )
    checks["synthetic_path_identity_eip"] = (
        _path_identity("ethereum", "EIPS/eip-123.md") == (123, "md")
    )
    checks["synthetic_path_identity_bip"] = (
        _path_identity("bitcoin", "bip-0123.mediawiki")
        == (123, "mediawiki")
    )
    checks["synthetic_path_reject"] = (
        _path_identity("bitcoin", "bip-123.mediawiki") is None
    )
    checks["synthetic_bucket_sentinel"] = (
        bucket_token("update_gap_days", None)
        == "UPDATE_GAP_DAYS_SENTINEL"
    )
    checks["synthetic_bucket_overflow"] = (
        bucket_token("line_change_count", 999)
        == "LINE_CHANGE_COUNT_500_PLUS"
    )
    checks["synthetic_gzip_deterministic"] = (
        deterministic_gzip(b"psim") == deterministic_gzip(b"psim")
    )
    checks["synthetic_json_canonical"] = (
        canonical_hash({"b": 2, "a": 1})
        == canonical_hash({"a": 1, "b": 2})
    )
    checks["synthetic_quarantine_exact"] = (
        prereg.MEMORIZATION_QUARANTINE["ethereum"][0] == 20
        and prereg.MEMORIZATION_QUARANTINE["bitcoin"][0] == 32
    )
    checks["synthetic_d90_primary"] = [
        row.name for row in prereg.ARCHIVE_SCHEDULES if row.primary_economic_clock
    ] == ["ARCHIVE_D90"]
    checks["synthetic_model_not_loaded"] = AccessLedger.zero().models_loaded == 0
    checks["synthetic_no_network"] = AccessLedger.zero().network_commands == 0
    checks["synthetic_no_source_rows"] = (
        AccessLedger.zero().source_path_rows_opened == 0
    )

    failed = sorted(name for name, passed in checks.items() if not passed)
    payload = {
        "protocol_version": SELF_CHECK_PROTOCOL,
        "policy_id": POLICY_ID,
        "checks": dict(sorted(checks.items())),
        "failed": failed,
        "synthetic": {
            "events": rows_fingerprint(
                [event_row(row) for row in events]
            ),
            "daily_cards": rows_fingerprint(
                [card_row(row) for row in cards]
            ),
            "controls_sha256": canonical_hash(controls),
            "split_support_sha256": canonical_hash(split_metrics),
        },
        "forbidden_access": AccessLedger.zero().snapshot(),
        "network_calls": 0,
        "source_event_rows_opened": 0,
        "outcomes_opened": False,
    }
    return {**payload, "manifest_hash": canonical_hash(payload)}


def self_check_bytes() -> bytes:
    payload = build_self_check_manifest()
    if payload["failed"]:
        raise RuntimeError(
            "PSIM synthetic self-check failed: "
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
        raise RuntimeError(f"PSIM path is not committed: {relative}")
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError(f"PSIM path commit changed: {relative}")
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
        raise RuntimeError(f"PSIM committed implementation differs: {path}")
    return _binding(path, commit, digest)


def _validate_binding(
    binding: Mapping[str, Any],
    *,
    path: str | Path,
    expected_commit: str | None = None,
    expected_sha256: str | None = None,
) -> None:
    relative = Path(path).as_posix()
    if set(binding) != {"path", "commit", "sha256"}:
        raise RuntimeError("PSIM binding schema changed")
    commit = binding.get("commit")
    digest = binding.get("sha256")
    if (
        binding.get("path") != relative
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
        raise RuntimeError(f"PSIM binding mismatch: {relative}")


def _load_preregistration() -> dict[str, Any]:
    path = repository_path(PREREGISTRATION_PATH)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("PSIM preregistration is absent or unsafe")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("PSIM preregistration is unreadable") from error
    if (
        not isinstance(payload, dict)
        or raw != canonical_json_bytes(payload)
        or sha256_bytes(raw) != PREREGISTRATION_SHA256
    ):
        raise RuntimeError("PSIM preregistration bytes changed")
    core = {
        key: value for key, value in payload.items() if key != "manifest_hash"
    }
    if (
        payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH
        or payload.get("manifest_hash") != canonical_hash(core)
        or payload != prereg.build_preregistration()
    ):
        raise RuntimeError("PSIM preregistration manifest changed")
    return payload


def python_runtime() -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    return {
        "python": {
            "path": str(executable),
            "sha256": sha256_file(executable),
            "version": sys.version,
        },
        "git_version": _git_output("--version"),
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
        "implementation_contract": _implementation_binding(
            IMPLEMENTATION_CONTRACT_PATH
        ),
    }
    _validate_binding(
        bindings["decision"],
        path=DECISION_PATH,
        expected_commit=DECISION_COMMIT,
        expected_sha256=DECISION_SHA256,
    )
    _validate_binding(
        bindings["preregistration"],
        path=PREREGISTRATION_PATH,
        expected_commit=PREREGISTRATION_COMMIT,
        expected_sha256=PREREGISTRATION_SHA256,
    )
    _validate_binding(
        bindings["preregistration_producer"],
        path=PREREGISTRATION_SCRIPT_PATH,
        expected_commit=PREREGISTRATION_COMMIT,
        expected_sha256=PREREGISTRATION_SCRIPT_SHA256,
    )
    _validate_binding(
        bindings["preregistration_document"],
        path=PREREGISTRATION_DOC_PATH,
        expected_commit=PREREGISTRATION_COMMIT,
        expected_sha256=PREREGISTRATION_DOC_SHA256,
    )
    source_core = {
        "source_start": format_time(SOURCE_START),
        "source_end_exclusive": format_time(SOURCE_END_EXCLUSIVE),
        "card_end_exclusive": format_time(CARD_END_EXCLUSIVE),
        "repositories": [
            {
                **asdict(row),
                "document_formats": list(row.document_formats),
            }
            for row in prereg.REPOSITORIES
        ],
        "schedules": [asdict(row) for row in prereg.ARCHIVE_SCHEDULES],
        "splits": list(prereg.SPLITS),
        "gates": list(GATE_NAMES),
        "controls": list(prereg.RELATION_CONTROLS),
        "parser_reference": registration["parser_contract"][
            "reference_parser"
        ],
    }
    return {
        "runtime": python_runtime(),
        **bindings,
        "preregistration_manifest_hash": registration["manifest_hash"],
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
            "PSIM self-check subprocess failed: "
            + completed.stderr.decode("utf-8", errors="replace")
        )
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("PSIM self-check output is not JSON") from error
    if completed.stdout != canonical_json_bytes(payload):
        raise RuntimeError("PSIM self-check output is noncanonical")
    core = {
        key: value for key, value in payload.items() if key != "manifest_hash"
    }
    if (
        payload.get("protocol_version") != SELF_CHECK_PROTOCOL
        or payload.get("policy_id") != POLICY_ID
        or payload.get("manifest_hash") != canonical_hash(core)
        or payload.get("failed") != []
        or payload.get("network_calls") != 0
        or payload.get("source_event_rows_opened") != 0
        or payload.get("outcomes_opened") is not False
        or payload.get("forbidden_access")
        != AccessLedger.zero().snapshot()
    ):
        raise RuntimeError("PSIM self-check manifest changed")
    return {
        "argv": argv,
        "exit_code": completed.returncode,
        "stdout_sha256": sha256_bytes(completed.stdout),
        "manifest_hash": payload["manifest_hash"],
        "network_calls": 0,
        "source_event_rows_opened": 0,
        "outcomes_opened": False,
        "forbidden_access": AccessLedger.zero().snapshot(),
    }


def _pytest_counts(stdout: str, stderr: str) -> dict[str, int]:
    summary_lines = [
        line.strip()
        for line in (stdout + "\n" + stderr).splitlines()
        if re.search(
            r"\b(?:passed|failed|skipped|errors?|xfailed|xpassed)\b",
            line,
        )
    ]
    if not summary_lines:
        raise RuntimeError("PSIM pytest summary is absent")
    counts = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
    }
    aliases = {"error": "errors", "errors": "errors"}
    for count, label in re.findall(
        r"([0-9]+)\s+"
        r"(passed|failed|skipped|errors?|xfailed|xpassed)\b",
        summary_lines[-1],
    ):
        counts[aliases.get(label, label)] += int(count)
    if counts["passed"] <= 0:
        raise RuntimeError("PSIM pytest passed count is absent")
    return counts


def _run_pytest_verification() -> dict[str, Any]:
    argv = [".venv/bin/pytest", "-q", TEST_PATH.as_posix()]
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
            "PSIM exact pytest verification failed\n"
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
        raise RuntimeError("PSIM seal creation requires a clean worktree")
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
            "PSIM runner, tests, and implementation contract "
            "must share current HEAD"
        )
    core = {
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
    return {**core, "seal_hash": canonical_hash(core)}


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
        raise RuntimeError("PSIM execution seal is absent or unsafe")
    seal_commit = _assert_committed(EXECUTION_SEAL_PATH)
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("PSIM execution seal is unreadable") from error
    if raw != canonical_json_bytes(payload):
        raise RuntimeError("PSIM execution seal bytes are noncanonical")
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
    core = {
        key: value for key, value in payload.items() if key != "seal_hash"
    }
    if (
        set(payload) != expected
        or payload.get("protocol_version") != SEAL_PROTOCOL
        or payload.get("policy_id") != POLICY_ID
        or payload.get("seal_hash") != canonical_hash(core)
        or payload.get("authority") != static_authority()
        or payload.get("forbidden_access")
        != AccessLedger.zero().snapshot()
    ):
        raise RuntimeError("PSIM execution seal core changed")
    shared_commit = payload.get("shared_commit")
    if not isinstance(shared_commit, str) or HEX40.fullmatch(shared_commit) is None:
        raise RuntimeError("PSIM sealed commit is malformed")
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
        or len(parent_row) != 2
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
            "PSIM seal commit is not the exact current execution HEAD"
        )
    expected_verification = {
        "self_check": _run_self_check_subprocess(),
        "pytest": _run_pytest_verification(),
    }
    if payload.get("synthetic_verification") != expected_verification:
        raise RuntimeError("PSIM sealed synthetic verification changed")
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
    if tuple(metrics) != tuple(prereg.RELATION_CONTROLS):
        raise RuntimeError("PSIM control order differs from preregistration")
    if gate.name != "relation_control_sensitivity":
        raise RuntimeError("PSIM control report received the wrong gate")
    card_rows = [card_row(row) for row in cards]
    core = {
        "protocol_version": CONTROL_REPORT_PROTOCOL,
        "policy_id": POLICY_ID,
        "baseline_cards": rows_fingerprint(card_rows),
        "control_order": list(prereg.RELATION_CONTROLS),
        "metrics": dict(metrics),
        "gate": gate.payload(),
        "outcomes_opened": False,
        "profitability_result": False,
        "forbidden_access": {
            name: 0 for name in FORBIDDEN_ACCESS_FIELDS
        },
    }
    return {**core, "report_hash": canonical_hash(core)}


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
    control_report = build_control_report(cards, controls, control_gate)
    event_jsonl = jsonl_bytes(event_rows)
    card_jsonl = jsonl_bytes(card_rows)
    control_metrics = dict(controls)
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
            rows=len(prereg.RELATION_CONTROLS),
            row_hash=canonical_hash(control_metrics),
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
        raise RuntimeError("PSIM gate order changed")
    if decision == "pass":
        if (
            len(gate_rows) != len(GATE_NAMES)
            or first_failure is not None
            or artifacts is None
            or any(row["passed"] is not True for row in gate_rows)
        ):
            raise RuntimeError("PSIM pass report is incomplete")
    elif decision == "reject":
        if not gate_rows or first_failure is None or artifacts is not None:
            raise RuntimeError("PSIM rejection report is incomplete")
    else:
        raise RuntimeError("PSIM result decision changed")
    source_opened = bool(
        source_audit.get("proposal_path_incidence_opened", False)
        or ledger.source_path_rows_opened
        or ledger.proposal_blobs_opened
    )
    core = {
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
    return {**core, "result_hash": canonical_hash(core)}


def _read_canonical_json(path: str | Path) -> dict[str, Any]:
    target = repository_path(path)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"PSIM artifact path is unsafe: {path}")
    raw = target.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"PSIM artifact is unreadable: {path}") from error
    if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
        raise RuntimeError(f"PSIM artifact is noncanonical: {path}")
    return payload


def _entry_exists(path: str | Path) -> bool:
    return os.path.lexists(repository_path(path))


def _validate_result_report(
    payload: Mapping[str, Any],
    *,
    config: Config,
    decision: str,
) -> None:
    core = {
        key: value for key, value in payload.items() if key != "result_hash"
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
        or payload.get("result_hash") != canonical_hash(core)
        or not isinstance(gates, list)
        or [row.get("name") for row in gates]
        != list(GATE_NAMES[: len(gates)])
        or payload.get("first_failure") != _first_failure(gates)
        or not isinstance(access, dict)
        or set(access) != set(AccessLedger.zero().snapshot())
        or any(type(value) is not int or value < 0 for value in access.values())
    ):
        raise RuntimeError("PSIM terminal result report changed")
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
            raise RuntimeError("PSIM pass report is invalid")
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
                raise RuntimeError("PSIM pass artifact manifest is invalid")
    else:
        if (
            not gates
            or payload.get("first_failure") is None
            or payload.get("artifacts") is not None
        ):
            raise RuntimeError("PSIM rejection report is invalid")


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
        _validate_result_report(payload, config=config, decision="reject")
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
                raise RuntimeError("PSIM pass artifact hash changed")
        return payload
    raise RuntimeError("PSIM terminal state is partial or conflicting")


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
        raise RuntimeError("PSIM pass artifact exists before rejection")
    _write_once_bytes(config.rejection_path, canonical_json_bytes(payload))


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
        raise RuntimeError("PSIM pass artifact group changed")
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
                    "PSIM existing pass artifact conflicts with publication"
                )
        return
    if rejection_exists or any(existing.values()):
        raise RuntimeError("PSIM terminal target already exists")
    staged: list[tuple[Path, Path]] = []
    published: list[tuple[Path, Path]] = []
    try:
        for relative, raw in expected_entries.items():
            target = _safe_destination(relative)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.",
                suffix=".psim-stage",
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
        error = RuntimeError("PSIM gate builder returned wrong identity")
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
            raise ValueError(f"PSIM path has a symlink ancestor: {cursor}")
    return absolute


def _validate_source_configuration(config: Config) -> None:
    if config.network_timeout_seconds <= 0:
        raise ValueError("PSIM network timeout must be positive")
    if not config.source_root.is_absolute():
        raise ValueError("PSIM source root must be absolute")
    source_root = _validate_no_symlink_ancestors(config.source_root)
    if source_root.exists() and not source_root.is_dir():
        raise ValueError("PSIM source root is unsafe")
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
        raise ValueError("PSIM official output paths are frozen")
    if len({repository_path(path) for path in output_paths}) != len(output_paths):
        raise ValueError("PSIM output paths are not unique")
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
        raise RuntimeError("PSIM run lock identity changed")
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
            raise RuntimeError("PSIM terminal authority changed")
        return existing

    seal = validate_execution_seal()
    authority = static_authority()
    authority_record = _authority_report(seal, authority)
    if not _worktree_clean():
        raise RuntimeError("PSIM official run requires a clean worktree")
    run_lock_path, run_lock_bytes = _acquire_run_lock()
    try:
        concurrent_terminal = terminal_state(config)
    except BaseException:
        _release_run_lock(run_lock_path, run_lock_bytes)
        raise
    if concurrent_terminal is not None:
        _release_run_lock(run_lock_path, run_lock_bytes)
        raise RuntimeError("PSIM terminal state appeared concurrently")

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
        "disk_limit_gib": prereg.DISK_LIMIT_GIB,
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
            if all((protocol, "a") in events for protocol in ("ethereum", "bitcoin"))
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
                if destination.exists():
                    raise RuntimeError(
                        f"PSIM fresh clone root already exists: {destination}"
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
                    "groups": _group_rows(groups[(protocol, replica)]),
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
                    event_row(row) for row in events[(protocol, replica)]
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
            raise RuntimeError("PSIM control evidence disappeared")
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
        raise RuntimeError("PSIM pass artifacts were not prepared")

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
