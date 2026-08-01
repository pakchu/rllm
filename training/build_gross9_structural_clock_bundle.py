"""Claim and one-shot builder for the G9CB-12 structural clock bundle.

The import-time and claim paths are deliberately stdlib-only.  Generic Gross9
runtime modules are imported only by a fresh worker process after the durable
attempt-consumed sentinel exists.  This module never imports or delegates to
candidate-specific replay code.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import concurrent.futures
import concurrent.futures.process
import copy
import csv
import ctypes
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import errno
import fnmatch
import fcntl
import gzip
import hashlib
import hmac
import importlib
import importlib.machinery
import importlib.metadata
import io
import json
import mmap
import multiprocessing
import multiprocessing.connection
import multiprocessing.process
import multiprocessing.shared_memory
import os
from pathlib import Path, PurePosixPath
import platform
import pty
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, NoReturn, Sequence
import zlib

from training import preregister_gross9_structural_clock_bundle as prereg


IDENTITY = "G9CB-12-SOURCE-SUPPORT"
PROTOCOL_VERSION = "gross9_structural_clock_bundle_g9cb12_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = Path("training/build_gross9_structural_clock_bundle.py")
BUILDER_TEST_PATH = Path("tests/test_build_gross9_structural_clock_bundle.py")
PREREGISTER_PATH = Path("training/preregister_gross9_structural_clock_bundle.py")
PREREGISTRATION_PATH = prereg.PREREGISTRATION_PATH
CLAIM_PATH = Path(
    "results/"
    "gross9_structural_clock_bundle_g9cb12_access_claim_2026-07-31.json"
)
SENTINEL_PATH = Path(
    "results/"
    "gross9_structural_clock_bundle_g9cb12_attempt_consumed_2026-07-31.json"
)
CSV_PATH = Path(
    "results/gross9_structural_clock_bundle_g9cb12_2026-07-31.csv.gz"
)
MANIFEST_PATH = Path(
    "results/"
    "gross9_structural_clock_bundle_g9cb12_manifest_2026-07-31.json"
)
WORKER_LEDGER_PATHS = (
    Path(
        "results/"
        "gross9_structural_clock_bundle_g9cb12_worker_capability_consumed_pass1_"
        "2026-07-31.json"
    ),
    Path(
        "results/"
        "gross9_structural_clock_bundle_g9cb12_worker_capability_consumed_pass2_"
        "2026-07-31.json"
    ),
)
H12_HANDOFF_PATH = Path(
    "results/gross9_structural_clock_bundle_g9cb12_v12_handoff_"
    "2026-07-31.json"
)
H12_SUPERVISOR_SENTINEL_PATH = Path(
    "results/gross9_structural_clock_bundle_g9cb12_h12_supervisor_"
    "attempt_consumed_2026-07-31.json"
)
V12_COMMAND = (
    "PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m "
    "training.build_gross9_structural_clock_bundle --verify-publication"
)
H12_COMMAND = (
    "PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m "
    "training.build_gross9_structural_clock_bundle --publish-v12-handoff"
)
H12_SUPERVISOR_ENV = "G9CB12_H12_SUPERVISOR_SENTINEL_SHA256"
H12_CAPABILITY_FD_ENV = "G9CB12_H12_V12_CAPABILITY_FD"
H12_CAPABILITY_SHA256_ENV = "G9CB12_H12_V12_CAPABILITY_SHA256"
H12_SUPERVISOR_PID_ENV = "G9CB12_H12_SUPERVISOR_PID"
H12_UV_EXECUTABLE = Path("/home/pakchu/.local/bin/uv")
H12_UV_EXECUTABLE_SHA256 = (
    "085e6be0fbb5f63c7ba39829703a7229cd62d2bd0b78ae145da9bf897e0fc007"
)
H12_UV_EXECUTABLE_SIZE = 54_537_024
H12_TOP_LEVEL_KEYS = (
    "active_alpha_goal",
    "identity",
    "ledger_kind",
    "next_workflow",
    "no_economics",
    "no_future_commit_prediction",
    "predecessor_bindings",
    "schema_version",
    "source_generation",
    "v12_stdout_hash",
)
H12_BINDING_KEYS = (
    "commit",
    "parent_commit",
    "stage",
    "tracked_paths",
)
V12_STDOUT_KEYS = (
    "claim_commit",
    "claim_hash",
    "csv_gzip_sha256",
    "final_manifest_hash",
    "head",
    "identity",
    "interval_count",
    "preregistration_manifest_hash",
    "preregistration_seal_commit",
    "protocol_implementation_commit",
    "protocol_version",
    "publication_commit",
    "sentinel_manifest_hash",
)
H12_SUPERVISOR_KEYS = (
    "attempt_hash",
    "capability_sha256",
    "expected_handoff_path",
    "h12_command",
    "identity",
    "one_shot",
    "repository_head",
    "repository_parent",
    "resume_allowed",
    "retry_allowed",
    "supervisor_pid",
    "uv_executable",
    "uv_executable_sha256",
    "v12_command",
    "zero_economics",
)
H12_STAGE_PATHS = {
    "S12": (
        "training/materialize_gross9_structural_clock_g9cb12_sources.py",
        "tests/test_materialize_gross9_structural_clock_g9cb12_sources.py",
    ),
    "M12": (
        "results/gross9_structural_clock_bundle_g9cb12_source_support_"
        "attempt_consumed_2026-07-31.json",
        "configs/shadow/gross9_structural_clock_bundle_g9cb12_sources_"
        "2026-07-31.json",
        "results/gross9_structural_clock_bundle_g9cb12_source_support_"
        "2026-07-31.json",
    ),
    "Q12": (
        "training/build_gross9_structural_clock_bundle.py",
        "training/preregister_gross9_structural_clock_bundle.py",
        "tests/test_build_gross9_structural_clock_bundle.py",
        "tests/test_preregister_gross9_structural_clock_bundle.py",
        "tests/test_gross9_structural_clock_bundle_preregistration_artifact.py",
    ),
    "P12": (PREREGISTRATION_PATH.as_posix(),),
    "C12": (CLAIM_PATH.as_posix(),),
    "D12": (
        SENTINEL_PATH.as_posix(),
        WORKER_LEDGER_PATHS[0].as_posix(),
        WORKER_LEDGER_PATHS[1].as_posix(),
        CSV_PATH.as_posix(),
        MANIFEST_PATH.as_posix(),
    ),
}
H12_STAGE_WORKTREE_MODES = {
    "S12": 0o644,
    "M12": 0o444,
    "Q12": 0o644,
    "P12": 0o444,
    "C12": 0o444,
    "D12": 0o444,
}
ACTIVE_PREREGISTRATION_DIFF = (
    f"A\t{PREREGISTRATION_PATH.as_posix()}",
)
CLAIM_DIFF = (f"A\t{CLAIM_PATH.as_posix()}",)
PUBLICATION_DIFF = tuple(
    sorted(
        f"A\t{path.as_posix()}"
        for path in (
            SENTINEL_PATH,
            *WORKER_LEDGER_PATHS,
            CSV_PATH,
            MANIFEST_PATH,
        )
    )
)
DOMAIN_START = "2023-06-01T00:00:00Z"
DOMAIN_END = "2026-06-01T00:00:00Z"
CSV_COLUMNS = (
    "identity",
    "sleeve",
    "sleeve_order",
    "configured_weight",
    "interval_index",
    "entry_time_utc",
    "exit_time_utc",
    "side",
)
SLEEVES = (
    {
        "name": "cand_rex_veto_7",
        "order": 0,
        "weight": "1.6",
        "kind": "fixed",
        "hold_bars": 144,
        "sides": (1, -1),
    },
    {
        "name": "fresh_kimchi_fx",
        "order": 1,
        "weight": "2.0",
        "kind": "barrier",
        "hold_bars": 288,
        "take_bps": 400,
        "stop_bps": 250,
        "sides": (1, -1),
    },
    {
        "name": "frozen_annual_rank7",
        "order": 2,
        "weight": "3.0",
        "kind": "rank7",
        "sides": (1,),
    },
    {
        "name": "markov_transition_long",
        "order": 3,
        "weight": "2.0",
        "kind": "fixed",
        "hold_bars": 576,
        "sides": (1,),
    },
    {
        "name": "rex_taker_low_range_position",
        "order": 4,
        "weight": "0.4",
        "kind": "fixed",
        "hold_bars": 144,
        "sides": (1, -1),
    },
)
SLEEVE_BY_NAME = {row["name"]: row for row in SLEEVES}
SPLIT_BOUNDS = (
    ("train", "2020-09-01", "2024-01-01"),
    ("test2024", "2024-01-01", "2025-01-01"),
    ("eval2025", "2025-01-01", "2026-01-01"),
    ("ytd2026", "2026-01-01", "2026-06-03"),
)
RUNTIME_IMPORT_MODULES = (
    "execution.gross9_rank7_clock_runtime",
    "training.gross9_structural_clock_primitives",
)
GENERIC_RUNTIME_MODULES = RUNTIME_IMPORT_MODULES
RANK7_LEARNER = {
    "max_depth": 2,
    "min_samples_leaf": 32,
    "max_features": "0.8",
}
RANK7_SELECTION = {
    "funding_quantile": "0.4",
    "premium_quantile": "0.55",
    "risk_lambda": "0.25",
    "risk_quantile": "0.75",
}
RANK7_LABEL_EXECUTION = {
    "leverage": 0.5,
    "fee_rate": 0.0005,
    "slippage_rate": 0.0001,
}
RANK7_ROWS_USED_COUNTERS = (
    "rank7_training_trades_replayed",
    "rank7_net_labels_computed",
    "rank7_adverse_labels_computed",
    "rank7_price_factor_values_used",
    "rank7_funding_factor_values_used",
    "rank7_funding_debit_factor_values_used",
    "rank7_adverse_price_factor_values_used",
    "rank7_fee_factor_values_used",
    "rank7_bundle_activation_rows_scored",
    "rank7_bundle_parity_rows_compared",
)
GZIP_PREFIX = bytes.fromhex("1f8b08000000000002ff")
TERMINAL_ACTION = "TERMINAL_G9CB12_ATTEMPT_CONSUMED_NO_RETRY"
_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_DIST_NORMALIZE_RE = re.compile(r"[-_.]+")
_STAGED_CSV_NAME = "gross9_structural_clock_bundle.csv.gz"
_STAGED_CORE_NAME = "gross9_structural_clock_bundle_core.json"
_STAGED_RECEIPT_NAME = "gross9_structural_clock_bundle_pass_receipt.json"
_PYCACHE_PREFIX_RELATIVE = Path("results/.g9cb12-bytecode-cache-disabled")
_ABSOLUTE_BINDING_ALLOWLIST = frozenset(
    {
        "/tmp/btcusdt_open_interest_5m_2020_2026.csv",
        (
            "/home/pakchu/rllm/data/"
            "cache_spot_premium_5m_2020-01-01_2026-06-01.csv.gz"
        ),
    }
)
_PREREGISTRATION_ONLY = "preregistration-only"
_PREREGISTRATION_PLUS_CLAIM = "preregistration-plus-claim"
Q12_PREREGISTRATION_PUBLICATION = "Q12_PREREGISTRATION_PUBLICATION"
P12_CLAIM_PREFLIGHT = "P12_CLAIM_PREFLIGHT"
C12_PRODUCTION_PREFLIGHT = "C12_PRODUCTION_PREFLIGHT"
D12_COMMITTED_VERIFICATION = "D12_COMMITTED_VERIFICATION"
PRODUCTION_CHECKPOINTS = (
    "C12_PRODUCTION_PREFLIGHT",
    "CAPABILITY_PROBE_COMPLETE",
    "SLOT1_PREPARED",
    "SENTINEL_LINKED",
    "PASS1_LEDGER_LINKED",
    "PASS1_OUTPUT_READY",
    "SLOT_TRANSITION",
    "PASS2_LEDGER_LINKED",
    "PASS2_OUTPUT_READY",
    "CANONICAL_CSV_LINKED",
    "MANIFEST_LINKED_LAST",
    "FINAL_CLEANUP",
)
HELPER_LOCAL_TRANSIENT_STATES = (
    "CAPABILITY_PROBE_LINKED_TRANSIENT",
    "STAGE_DIRECTORY_CREATED_TRANSIENT",
    "STAGE_FILE_IN_PROGRESS_TRANSIENT",
    "STAGE_CLEANUP_TRANSIENT",
    "UNNAMED_CANONICAL_LINK_TRANSIENT",
)
_PR_SET_PDEATHSIG = 1
_LIBC = ctypes.CDLL(None, use_errno=True)


class TerminalG9CB12Failure(RuntimeError):
    """A terminal protocol or post-sentinel failure."""


def _fail(message: str) -> NoReturn:
    raise TerminalG9CB12Failure(message)


def _validate_production_checkpoint(checkpoint: str) -> None:
    if checkpoint in HELPER_LOCAL_TRANSIENT_STATES:
        _fail(f"helper-local transient is not a stable checkpoint: {checkpoint}")
    if checkpoint not in PRODUCTION_CHECKPOINTS:
        _fail(f"unknown production checkpoint: {checkpoint}")


class _ProductionStateMachine:
    def __init__(self) -> None:
        self._position = -1

    @property
    def current(self) -> str | None:
        if self._position < 0:
            return None
        return PRODUCTION_CHECKPOINTS[self._position]

    def advance(self, checkpoint: str, validate: Any) -> None:
        _validate_production_checkpoint(checkpoint)
        expected_position = self._position + 1
        if (
            expected_position >= len(PRODUCTION_CHECKPOINTS)
            or PRODUCTION_CHECKPOINTS[expected_position] != checkpoint
        ):
            _fail(
                "production checkpoint transition differs: "
                f"{self.current!r} -> {checkpoint!r}"
            )
        validate()
        self._position = expected_position

    def require_complete(self) -> None:
        if self.current != "FINAL_CLEANUP":
            _fail("production checkpoint sequence is incomplete")


def _canonical_json_bytes(payload: Any, *, trailing_lf: bool = True) -> bytes:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if trailing_lf else b"")


def _canonical_h12_json_bytes(
    payload: Any, *, trailing_lf: bool = True
) -> bytes:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if trailing_lf else b"")


def _object_hash(payload: Mapping[str, Any], field: str) -> str:
    core = {key: value for key, value in payload.items() if key != field}
    return hashlib.sha256(_canonical_json_bytes(core, trailing_lf=False)).hexdigest()


def _with_hash(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = _object_hash(result, field)
    return result


def _sha256_bytes(raw: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_canonical_object(
    raw: bytes,
    path_text: str,
    hash_field: str | None = None,
) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TerminalG9CB12Failure(
            f"invalid canonical JSON: {path_text}"
        ) from exc
    if not isinstance(payload, dict) or raw != _canonical_json_bytes(payload):
        _fail(f"noncanonical JSON bytes: {path_text}")
    if hash_field is not None:
        value = payload.get(hash_field)
        if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
            _fail(f"missing {hash_field}: {path_text}")
        if value != _object_hash(payload, hash_field):
            _fail(f"{hash_field} mismatch: {path_text}")
    return payload


def _read_canonical_object(
    path: Path,
    hash_field: str | None = None,
    *,
    path_text: str | None = None,
    raw_cache: Mapping[str, tuple[bytes, os.stat_result]] | None = None,
) -> tuple[dict[str, Any], bytes, os.stat_result]:
    cache_key = path.as_posix() if path_text is None else path_text
    cached = raw_cache.get(cache_key) if raw_cache is not None else None
    if raw_cache is not None and cached is None:
        _fail(f"canonical cache binding is absent: {cache_key}")
    raw, info = (
        cached
        if cached is not None
        else _read_bound_regular_bytes(path, cache_key)
    )
    payload = _decode_canonical_object(raw, cache_key, hash_field)
    return payload, raw, info


def _rooted(root: Path, relative: Path) -> Path:
    relative = Path(relative)
    if relative.is_absolute() or any(
        part in ("", ".", "..") for part in relative.parts
    ):
        _fail(f"canonical output path must be relative: {relative}")
    repository_root = root.resolve()
    target = repository_root / relative
    try:
        target.relative_to(repository_root)
    except ValueError:
        _fail(f"unsafe canonical path: {relative}")
    current = repository_root
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            _fail(f"canonical path has a symlinked parent: {relative}")
        if current.exists() and not current.is_dir():
            _fail(f"canonical path parent is not a directory: {relative}")
    return target


def _git(root: Path, *arguments: str, allow_failure: bool = False) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode and not allow_failure:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        _fail(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout


def _git_text(root: Path, *arguments: str) -> str:
    return _git(root, *arguments).decode("utf-8").strip()


def _tracked_results_top_level_entries(output: str) -> set[str]:
    """Project normalized tracked results paths onto entries below results/."""
    entries: set[str] = set()
    for path_text in output.splitlines():
        components = path_text.split("/")
        if (
            len(components) < 2
            or components[0] != "results"
            or any(component in {"", ".", ".."} for component in components)
            or "\\" in path_text
        ):
            _fail("malformed tracked results path")
        entries.add(components[1])
    return entries


def _single_parent_commit(root: Path, commit: str) -> str:
    fields = _git_text(
        root,
        "rev-list",
        "--parents",
        "-n",
        "1",
        commit,
    ).split()
    if len(fields) != 2 or fields[0] != commit:
        _fail(f"{commit}: expected exactly one Git parent")
    return fields[1]


def _commit_name_status(
    root: Path,
    parent: str,
    child: str,
) -> tuple[str, ...]:
    output = _git_text(
        root,
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        parent,
        child,
    )
    return tuple(line for line in output.splitlines() if line)


def _require_clean_pushed_branch(root: Path, expected_branch: str | None) -> str:
    head = _git_text(root, "rev-parse", "--verify", "HEAD")
    upstream = _git_text(root, "rev-parse", "--verify", "@{upstream}")
    if head != upstream:
        _fail("HEAD does not equal the pushed upstream commit")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all").strip():
        _fail("worktree or index is not clean")
    branch = _git_text(root, "branch", "--show-current")
    if not branch:
        _fail("detached HEAD is forbidden")
    if expected_branch is not None and branch != expected_branch:
        _fail(f"expected branch {expected_branch!r}, found {branch!r}")
    return head


def _validate_preregistration_seal_head(
    root: Path,
    preregistration: Mapping[str, Any],
    head: str,
    *,
    classify_worktree: bool = True,
) -> None:
    implementation = preregistration.get("protocol_implementation_commit")
    if not isinstance(implementation, str) or not re.fullmatch(
        r"[0-9a-f]{40}", implementation
    ):
        _fail("protocol implementation commit is invalid")
    if _single_parent_commit(root, head) != implementation:
        _fail("HEAD is not the direct preregistration-seal child of Q")
    if _commit_name_status(
        root,
        implementation,
        head,
    ) != ACTIVE_PREREGISTRATION_DIFF:
        _fail("preregistration-seal commit diff differs")
    path = PREREGISTRATION_PATH.as_posix()
    if not _git(
        root,
        "ls-files",
        "--error-unmatch",
        "--",
        path,
        allow_failure=True,
    ).strip():
        _fail("active preregistration is not tracked")
    if classify_worktree:
        pair = _validate_git_pair_preflight(
            root,
            path,
            repository_relative=True,
            declaration={},
            verify_git=True,
            require_tracked=True,
        )
        if pair is None or pair[1] != "100644":
            _fail("active preregistration Git mode differs")


def _validate_committed_publication_topology(
    root: Path,
    preregistration: Mapping[str, Any],
    head: str,
) -> dict[str, str]:
    publication = head
    claim = _single_parent_commit(root, publication)
    preregistration_seal = _single_parent_commit(root, claim)
    implementation = _single_parent_commit(root, preregistration_seal)
    if (
        preregistration.get("protocol_implementation_commit")
        != implementation
    ):
        _fail("publication chain protocol implementation commit differs")
    if _commit_name_status(
        root,
        implementation,
        preregistration_seal,
    ) != ACTIVE_PREREGISTRATION_DIFF:
        _fail("publication chain Q-to-P diff differs")
    if _commit_name_status(
        root,
        preregistration_seal,
        claim,
    ) != CLAIM_DIFF:
        _fail("publication chain P-to-C diff differs")
    if _commit_name_status(
        root,
        claim,
        publication,
    ) != PUBLICATION_DIFF:
        _fail("publication chain C-to-D diff differs")
    return {
        "protocol_implementation_commit": implementation,
        "preregistration_seal_commit": preregistration_seal,
        "claim_commit": claim,
        "publication_commit": publication,
    }


def _decode_h12_canonical_object(raw: bytes, label: str) -> dict[str, Any]:
    if not isinstance(raw, bytes):
        _fail(f"{label} is not bytes")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TerminalG9CB12Failure(
            f"{label} is not valid duplicate-free UTF-8 JSON"
        ) from exc
    if (
        not isinstance(payload, dict)
        or raw != _canonical_h12_json_bytes(payload)
    ):
        _fail(f"{label} is not canonical JSON with one trailing LF")
    return payload


def _commit_tree_mode(root: Path, commit: str, path_text: str) -> str:
    if not _COMMIT_RE.fullmatch(commit):
        _fail("H12 stage commit is not lowercase 40-hex")
    raw = _git(
        root,
        "ls-tree",
        "-z",
        "--full-tree",
        commit,
        "--",
        path_text,
    )
    entries = [entry for entry in raw.split(b"\0") if entry]
    if len(entries) != 1 or b"\t" not in entries[0]:
        _fail(f"H12 stage path is absent or ambiguous: {path_text}")
    metadata, encoded_path = entries[0].split(b"\t", 1)
    fields = metadata.split()
    if (
        len(fields) != 3
        or fields[1] != b"blob"
        or not re.fullmatch(rb"[0-9a-f]{40}", fields[2])
        or encoded_path != path_text.encode("utf-8")
    ):
        _fail(f"H12 stage tree entry differs: {path_text}")
    try:
        return fields[0].decode("ascii")
    except UnicodeDecodeError as exc:
        raise TerminalG9CB12Failure(
            f"H12 stage mode is not ASCII: {path_text}"
        ) from exc


def _expected_h12_predecessor_bindings(
    root: Path, d12_head: str
) -> list[dict[str, Any]]:
    if not _COMMIT_RE.fullmatch(d12_head):
        _fail("D12 head is not lowercase 40-hex")
    commits: dict[str, str] = {"D12": d12_head}
    commits["C12"] = _single_parent_commit(root, commits["D12"])
    commits["P12"] = _single_parent_commit(root, commits["C12"])
    commits["Q12"] = _single_parent_commit(root, commits["P12"])
    commits["M12"] = _single_parent_commit(root, commits["Q12"])
    commits["S12"] = _single_parent_commit(root, commits["M12"])
    t11 = _single_parent_commit(root, commits["S12"])
    if commits["S12"] != prereg.G9CB12_SOURCE_SUPPORT_COMMIT:
        _fail("H12 S12 commit differs from authenticated source support")
    if commits["M12"] != prereg.G9CB12_SOURCE_MANIFEST_COMMIT:
        _fail("H12 M12 commit differs from authenticated source manifest")
    if t11 != prereg.G9CB11_TERMINAL_EVIDENCE_COMMIT:
        _fail("H12 S12 parent differs from authenticated T11")

    parents = {
        "S12": t11,
        "M12": commits["S12"],
        "Q12": commits["M12"],
        "P12": commits["Q12"],
        "C12": commits["P12"],
        "D12": commits["C12"],
    }
    statuses = {
        "S12": "A",
        "M12": "A",
        "Q12": "M",
        "P12": "A",
        "C12": "A",
        "D12": "A",
    }
    for stage in H12_STAGE_PATHS:
        paths = H12_STAGE_PATHS[stage]
        expected_diff = tuple(
            f"{statuses[stage]}\t{path_text}"
            for path_text in sorted(paths)
        )
        if _commit_name_status(
            root, parents[stage], commits[stage]
        ) != expected_diff:
            _fail(f"H12 {stage} parent diff differs")
        for path_text in paths:
            if _commit_tree_mode(root, commits[stage], path_text) != "100644":
                _fail(f"H12 {stage} path Git mode differs: {path_text}")

    bindings = [
        {
            "commit": commits[stage],
            "parent_commit": parents[stage],
            "stage": stage,
            "tracked_paths": list(H12_STAGE_PATHS[stage]),
        }
        for stage in H12_STAGE_PATHS
    ]
    return _normalize_h12_predecessor_bindings(bindings)


def _normalize_h12_predecessor_bindings(
    bindings: Any,
) -> list[dict[str, Any]]:
    if not isinstance(bindings, list) or len(bindings) != len(H12_STAGE_PATHS):
        _fail("H12 predecessor_bindings is not the exact six-row array")
    normalized: list[dict[str, Any]] = []
    stages = tuple(H12_STAGE_PATHS)
    for index, (row, stage) in enumerate(zip(bindings, stages, strict=True)):
        if not isinstance(row, dict) or tuple(row) != H12_BINDING_KEYS:
            _fail(f"H12 predecessor binding {index} keys/order differ")
        commit = row.get("commit")
        parent = row.get("parent_commit")
        tracked_paths = row.get("tracked_paths")
        if (
            not isinstance(commit, str)
            or not _COMMIT_RE.fullmatch(commit)
            or not isinstance(parent, str)
            or not _COMMIT_RE.fullmatch(parent)
            or row.get("stage") != stage
            or not isinstance(tracked_paths, list)
            or tracked_paths != list(H12_STAGE_PATHS[stage])
            or not all(isinstance(path, str) for path in tracked_paths)
        ):
            _fail(f"H12 predecessor binding differs: {stage}")
        normalized.append(
            {
                "commit": commit,
                "parent_commit": parent,
                "stage": stage,
                "tracked_paths": list(tracked_paths),
            }
        )
    for prior, current in zip(normalized, normalized[1:], strict=False):
        if current["parent_commit"] != prior["commit"]:
            _fail("H12 predecessor direct-parent chain differs")
    learned_oids = [normalized[0]["parent_commit"]] + [
        row["commit"] for row in normalized
    ]
    if len(set(learned_oids)) != len(learned_oids):
        _fail("H12 predecessor chain repeats a commit")
    return normalized


def _validated_v12_stdout(
    raw: bytes, predecessor_bindings: list[dict[str, Any]]
) -> dict[str, Any]:
    payload = _decode_h12_canonical_object(raw, "V12 stdout")
    if tuple(payload) != V12_STDOUT_KEYS:
        _fail("V12 stdout keys/order differ")
    for key in (
        "claim_commit",
        "head",
        "preregistration_seal_commit",
        "protocol_implementation_commit",
        "publication_commit",
    ):
        value = payload.get(key)
        if not isinstance(value, str) or not _COMMIT_RE.fullmatch(value):
            _fail(f"V12 stdout commit differs: {key}")
    for key in (
        "claim_hash",
        "csv_gzip_sha256",
        "final_manifest_hash",
        "preregistration_manifest_hash",
        "sentinel_manifest_hash",
    ):
        value = payload.get(key)
        if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
            _fail(f"V12 stdout hash differs: {key}")
    if (
        payload.get("identity") != IDENTITY
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or type(payload.get("interval_count")) is not int
        or payload["interval_count"] < 0
        or payload["head"] != payload["publication_commit"]
    ):
        _fail("V12 stdout literal/type contract differs")
    by_stage = {row["stage"]: row for row in predecessor_bindings}
    if (
        payload["protocol_implementation_commit"] != by_stage["Q12"]["commit"]
        or payload["preregistration_seal_commit"] != by_stage["P12"]["commit"]
        or payload["claim_commit"] != by_stage["C12"]["commit"]
        or payload["publication_commit"] != by_stage["D12"]["commit"]
    ):
        _fail("V12 stdout does not bind the authenticated Q12-P12-C12-D12 chain")
    return payload


def _h12_handoff_payload(
    predecessor_bindings: list[dict[str, Any]], v12_stdout: bytes
) -> dict[str, Any]:
    bindings = _normalize_h12_predecessor_bindings(predecessor_bindings)
    _validated_v12_stdout(v12_stdout, bindings)
    return {
        "active_alpha_goal": "incomplete",
        "identity": IDENTITY,
        "ledger_kind": "gross9_structural_clock_bundle_g9cb12_v12_handoff_v1",
        "next_workflow": "ralplan",
        "no_economics": True,
        "no_future_commit_prediction": True,
        "predecessor_bindings": bindings,
        "schema_version": 1,
        "source_generation": "G9CB12",
        "v12_stdout_hash": _sha256_bytes(v12_stdout),
    }


def _contains_h12_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return "<" in value or ">" in value
    if isinstance(value, Mapping):
        return any(
            _contains_h12_placeholder(key)
            or _contains_h12_placeholder(member)
            for key, member in value.items()
        )
    if isinstance(value, list):
        return any(_contains_h12_placeholder(member) for member in value)
    return False


def validate_g9cb12_h12_handoff(
    payload: Mapping[str, Any] | bytes,
    *,
    v12_stdout: bytes,
    predecessor_bindings: list[dict[str, Any]],
) -> dict[str, Any]:
    observed = (
        _decode_h12_canonical_object(payload, "H12 handoff")
        if isinstance(payload, bytes)
        else payload
    )
    if not isinstance(observed, dict) or tuple(observed) != H12_TOP_LEVEL_KEYS:
        _fail("H12 handoff top-level keys/order differ")
    if (
        type(observed.get("schema_version")) is not int
        or observed.get("schema_version") != 1
        or type(observed.get("no_economics")) is not bool
        or observed.get("no_economics") is not True
        or type(observed.get("no_future_commit_prediction")) is not bool
        or observed.get("no_future_commit_prediction") is not True
    ):
        _fail("H12 handoff integer/boolean contract differs")
    for key, literal in (
        ("active_alpha_goal", "incomplete"),
        ("identity", IDENTITY),
        (
            "ledger_kind",
            "gross9_structural_clock_bundle_g9cb12_v12_handoff_v1",
        ),
        ("next_workflow", "ralplan"),
        ("source_generation", "G9CB12"),
    ):
        if type(observed.get(key)) is not str or observed.get(key) != literal:
            _fail(f"H12 handoff literal differs: {key}")
    stdout_hash = observed.get("v12_stdout_hash")
    if (
        not isinstance(stdout_hash, str)
        or not _SHA_RE.fullmatch(stdout_hash)
        or stdout_hash != _sha256_bytes(v12_stdout)
    ):
        _fail("H12 handoff V12 stdout hash differs")
    normalized = _normalize_h12_predecessor_bindings(
        observed.get("predecessor_bindings")
    )
    expected = _h12_handoff_payload(predecessor_bindings, v12_stdout)
    if normalized != expected["predecessor_bindings"] or observed != expected:
        _fail("H12 handoff schema or authenticated bindings differ")
    if _contains_h12_placeholder(observed):
        _fail("H12 handoff contains an unresolved placeholder")
    return copy.deepcopy(expected)


def _h12_supervisor_payload(
    head: str,
    parent: str,
    capability_sha256: str,
    supervisor_pid: int,
) -> dict[str, Any]:
    if not _COMMIT_RE.fullmatch(head) or not _COMMIT_RE.fullmatch(parent):
        _fail("H12 supervisor repository commit binding differs")
    if not _SHA_RE.fullmatch(capability_sha256):
        _fail("H12 supervisor capability hash differs")
    if type(supervisor_pid) is not int or supervisor_pid <= 0:
        _fail("H12 supervisor PID differs")
    core = {
        "capability_sha256": capability_sha256,
        "expected_handoff_path": H12_HANDOFF_PATH.as_posix(),
        "h12_command": H12_COMMAND,
        "identity": "G9CB-12-H12-SUPERVISOR",
        "one_shot": True,
        "repository_head": head,
        "repository_parent": parent,
        "resume_allowed": False,
        "retry_allowed": False,
        "supervisor_pid": supervisor_pid,
        "uv_executable": H12_UV_EXECUTABLE.as_posix(),
        "uv_executable_sha256": H12_UV_EXECUTABLE_SHA256,
        "v12_command": V12_COMMAND,
        "zero_economics": True,
    }
    attempt_hash = _sha256_bytes(
        _canonical_h12_json_bytes(core, trailing_lf=False)
    )
    return {"attempt_hash": attempt_hash, **core}


def _validated_h12_supervisor_payload(raw: bytes) -> dict[str, Any]:
    payload = _decode_h12_canonical_object(raw, "H12 supervisor sentinel")
    if tuple(payload) != H12_SUPERVISOR_KEYS:
        _fail("H12 supervisor sentinel keys/order differ")
    expected = _h12_supervisor_payload(
        str(payload.get("repository_head")),
        str(payload.get("repository_parent")),
        str(payload.get("capability_sha256")),
        payload.get("supervisor_pid"),
    )
    if payload != expected:
        _fail("H12 supervisor sentinel schema/self-hash differs")
    return payload


def _read_h12_supervisor_sentinel(
    root: Path, expected_sha256: str
) -> tuple[dict[str, Any], bytes]:
    if not _SHA_RE.fullmatch(expected_sha256):
        _fail("H12 supervisor environment hash differs")
    path = _rooted(root, H12_SUPERVISOR_SENTINEL_PATH)
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(descriptor)
        raw = _read_publication_descriptor(descriptor, before.st_size)
        after = os.fstat(descriptor)
        edge = os.lstat(path)
        if (
            _descriptor_token(before) != _descriptor_token(after)
            or _descriptor_token(after) != _descriptor_token(edge)
            or not stat.S_ISREG(after.st_mode)
            or stat.S_IMODE(after.st_mode) != 0o444
            or after.st_nlink != 1
            or _sha256_bytes(raw) != expected_sha256
        ):
            _fail("H12 supervisor sentinel inode/hash binding differs")
    finally:
        os.close(descriptor)
    return _validated_h12_supervisor_payload(raw), raw


def _path_lexists(path: Path) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    return True


def _h12_v12_artifact_names(names: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name in names
            if name.startswith("gross9_structural_clock_bundle_g9cb12_v12")
        )
    )


def _authenticated_h12_uv_executable() -> str:
    descriptor = os.open(
        H12_UV_EXECUTABLE,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(descriptor)
        raw = _read_publication_descriptor(descriptor, before.st_size)
        after = os.fstat(descriptor)
        edge = os.lstat(H12_UV_EXECUTABLE)
        if (
            _descriptor_token(before) != _descriptor_token(after)
            or _descriptor_token(after) != _descriptor_token(edge)
            or not stat.S_ISREG(after.st_mode)
            or stat.S_IMODE(after.st_mode) != 0o755
            or after.st_nlink != 1
            or after.st_size != H12_UV_EXECUTABLE_SIZE
            or _sha256_bytes(raw) != H12_UV_EXECUTABLE_SHA256
        ):
            _fail("H12 uv executable binding differs")
    finally:
        os.close(descriptor)
    return H12_UV_EXECUTABLE.as_posix()


def _consume_h12_v12_capability(
    descriptor_text: str, expected_sha256: str
) -> None:
    if (
        not re.fullmatch(r"(?:0|[1-9][0-9]*)", descriptor_text)
        or not _SHA_RE.fullmatch(expected_sha256)
    ):
        _fail("H12 V12 capability environment differs")
    descriptor = int(descriptor_text)
    token = bytearray()
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISFIFO(info.st_mode):
            _fail("H12 V12 capability carrier is not a FIFO")
        while len(token) < 32:
            chunk = os.read(descriptor, 32 - len(token))
            if not chunk:
                _fail("H12 V12 capability reached EOF before 32 bytes")
            token.extend(chunk)
        if os.read(descriptor, 1) != b"":
            _fail("H12 V12 capability contains extra bytes")
        if _sha256_bytes(token) != expected_sha256:
            _fail("H12 V12 capability token hash differs")
    except OSError as exc:
        raise TerminalG9CB12Failure(
            "H12 V12 capability carrier cannot be consumed"
        ) from exc
    finally:
        token[:] = b"\0" * len(token)
        try:
            os.close(descriptor)
        except OSError:
            pass


def _linux_process_parent(pid: int) -> int:
    if type(pid) is not int or pid <= 0:
        _fail("H12 process PID differs")
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError) as exc:
        raise TerminalG9CB12Failure(
            "H12 process ancestry cannot be authenticated"
        ) from exc
    closing = raw.rfind(")")
    fields = raw[closing + 2 :].split() if closing >= 0 else []
    if len(fields) < 2 or not fields[1].isdigit():
        _fail("H12 process ancestry record differs")
    return int(fields[1])


def _repository_namespace_snapshot(
    root: Path,
) -> dict[str, tuple[int, ...]]:
    snapshot: dict[str, tuple[int, ...]] = {}
    for current, directories, files in os.walk(
        root, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        if current_path == root and ".git" in directories:
            directories.remove(".git")
        directories.sort()
        files.sort()
        for name in (*directories, *files):
            candidate = current_path / name
            relative = candidate.relative_to(root).as_posix()
            info = os.lstat(candidate)
            snapshot[relative] = (
                info.st_dev,
                info.st_ino,
                stat.S_IFMT(info.st_mode),
                stat.S_IMODE(info.st_mode),
                info.st_nlink,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            )
    return snapshot


def _h12_stage_worktree_snapshot(
    root: Path, predecessor_bindings: list[dict[str, Any]]
) -> dict[str, tuple[int, ...]]:
    bindings = _normalize_h12_predecessor_bindings(predecessor_bindings)
    snapshot: dict[str, tuple[int, ...]] = {}
    for row in bindings:
        stage = row["stage"]
        expected_mode = H12_STAGE_WORKTREE_MODES[stage]
        for path_text in row["tracked_paths"]:
            info = os.lstat(_rooted(root, Path(path_text)))
            if (
                not stat.S_ISREG(info.st_mode)
                or stat.S_IMODE(info.st_mode) != expected_mode
                or info.st_nlink != 1
            ):
                _fail(f"H12 {stage} worktree mode/type differs: {path_text}")
            snapshot[path_text] = _descriptor_token(info)
    return snapshot


def _require_h12_stage_worktree_unchanged(
    root: Path, expected: Mapping[str, tuple[int, ...]]
) -> None:
    for path_text, token in expected.items():
        if _descriptor_token(os.lstat(_rooted(root, Path(path_text)))) != token:
            _fail(f"H12 stage worktree binding changed: {path_text}")


def _require_h12_v12_supervision(root: Path) -> dict[str, Any]:
    expected_sha256 = os.environ.get(H12_SUPERVISOR_ENV, "")
    sentinel, _raw = _read_h12_supervisor_sentinel(
        root, expected_sha256
    )
    capability_sha256 = os.environ.get(H12_CAPABILITY_SHA256_ENV, "")
    supervisor_pid_text = os.environ.get(H12_SUPERVISOR_PID_ENV, "")
    if (
        capability_sha256 != sentinel["capability_sha256"]
        or not re.fullmatch(r"[1-9][0-9]*", supervisor_pid_text)
        or int(supervisor_pid_text) != sentinel["supervisor_pid"]
    ):
        _fail("H12 V12 supervisor capability binding differs")
    parent_pid = os.getppid()
    if (
        parent_pid != sentinel["supervisor_pid"]
        and _linux_process_parent(parent_pid) != sentinel["supervisor_pid"]
    ):
        _fail("V12 is not a direct or uv-mediated H12 child")
    _consume_h12_v12_capability(
        os.environ.get(H12_CAPABILITY_FD_ENV, ""),
        capability_sha256,
    )
    if (
        sentinel["uv_executable"] != H12_UV_EXECUTABLE.as_posix()
        or sentinel["uv_executable_sha256"]
        != H12_UV_EXECUTABLE_SHA256
        or Path(sys.prefix).resolve() != (root / ".venv").resolve()
    ):
        _fail("V12 interpreter/supervisor executable binding differs")
    if _path_lexists(_rooted(root, H12_HANDOFF_PATH)):
        _fail("V12 handoff already exists")
    head = _require_clean_pushed_branch(root, prereg.EXPECTED_BRANCH)
    if (
        sentinel["repository_head"] != head
        or sentinel["repository_parent"] != _single_parent_commit(root, head)
    ):
        _fail("H12 supervisor repository binding differs")
    if _git(
        root,
        "ls-files",
        "--error-unmatch",
        "--",
        H12_SUPERVISOR_SENTINEL_PATH.as_posix(),
        allow_failure=True,
    ).strip():
        _fail("H12 supervisor sentinel must remain untracked")
    results_names = os.listdir(_rooted(root, Path("results")))
    if _h12_v12_artifact_names(results_names):
        _fail("V12 file artifact exists")
    return sentinel


_PATH_ALIASES = ("path", "logical_path", "repository_path")


def _iter_bindings(payload: Any) -> Iterable[Mapping[str, Any]]:
    """Yield qualifying path/SHA mappings in the G9CB-1B traversal order."""

    if isinstance(payload, Mapping):
        digest = payload.get("sha256")
        present_aliases = [key for key in _PATH_ALIASES if key in payload]
        if isinstance(digest, str) and present_aliases:
            values = [payload[key] for key in present_aliases]
            if any(not isinstance(value, str) for value in values):
                _fail("path/SHA binding has a non-string path alias")
            if len(set(values)) != 1:
                _fail("path/SHA binding has conflicting path aliases")
            yield payload
        for key in sorted(payload):
            if key == "protocol_implementation":
                continue
            yield from _iter_bindings(payload[key])
    elif isinstance(payload, list):
        for value in payload:
            yield from _iter_bindings(value)


def _binding_path(binding: Mapping[str, Any]) -> str:
    values = [
        binding[key]
        for key in _PATH_ALIASES
        if key in binding and isinstance(binding[key], str)
    ]
    if not values:
        _fail("qualifying path/SHA binding has no string path alias")
    if len(set(values)) != 1:
        _fail("path/SHA binding has conflicting path aliases")
    return str(values[0])


def _require_bound_regular_lstat(path: Path, path_text: str) -> None:
    try:
        mode = os.lstat(path).st_mode
    except OSError as exc:
        raise TerminalG9CB12Failure(
            f"bound input cannot be inspected: {path_text}"
        ) from exc
    if not stat.S_ISREG(mode):
        _fail(f"bound input is not a regular file: {path_text}")


def _bound_regular_path(root: Path, path_text: str) -> tuple[Path, bool]:
    if not path_text or "\x00" in path_text:
        _fail("bound input path is empty or contains NUL")
    if path_text.startswith("/"):
        root_text = os.path.abspath(os.fspath(root))
        if path_text == root_text or path_text.startswith(root_text + "/"):
            _fail(
                "bound absolute repository input must be repository-relative: "
                f"{path_text}"
            )
        if (
            "//" in path_text
            or Path(path_text).as_posix() != path_text
            or any(part in ("", ".", "..") for part in path_text[1:].split("/"))
        ):
            _fail(f"bound absolute input is not canonical text: {path_text}")
        return Path(path_text), False

    if "\\" in path_text:
        _fail(f"bound repository path is not POSIX text: {path_text}")
    components = path_text.split("/")
    if any(component in ("", ".", "..") for component in components):
        _fail(f"bound repository path is not normalized: {path_text}")
    pure = PurePosixPath(path_text)
    if pure.is_absolute() or pure.as_posix() != path_text:
        _fail(f"bound repository path is not normalized: {path_text}")
    return root.joinpath(*components), True


def _validate_zero_access(payload: Mapping[str, Any]) -> None:
    try:
        prereg.validate_zero_access_schema(payload)
    except (TypeError, ValueError) as exc:
        raise TerminalG9CB12Failure(
            f"preregistration zero-access schema differs: {exc}"
        ) from exc


def _expected_authority_amendment_bindings() -> list[dict[str, Any]]:
    return [
        {
            "identity": "G9CB-1A",
            "path": prereg.RANK7_AUTHORITY_AMENDMENT_PATH.as_posix(),
            "path_type": "regular_file",
            "sha256": prereg.RANK7_AUTHORITY_AMENDMENT_SHA256,
            "git_blob": prereg.RANK7_AUTHORITY_AMENDMENT_GIT_BLOB,
            "git_mode": "100644",
            "authority_commit": prereg.RANK7_AUTHORITY_AMENDMENT_COMMIT,
        },
        {
            "identity": "G9CB-1B",
            "path": prereg.RUNTIME_ISOLATION_AMENDMENT_PATH.as_posix(),
            "path_type": "regular_file",
            "sha256": prereg.RUNTIME_ISOLATION_AMENDMENT_SHA256,
            "git_blob": prereg.RUNTIME_ISOLATION_AMENDMENT_GIT_BLOB,
            "git_mode": "100644",
            "authority_commit": prereg.RUNTIME_ISOLATION_AMENDMENT_COMMIT,
        },
        {
            "identity": "G9CB-1C",
            "path": (
                prereg.PREREGISTRATION_CORRECTION_AMENDMENT_PATH.as_posix()
            ),
            "path_type": "regular_file",
            "sha256": (
                prereg.PREREGISTRATION_CORRECTION_AMENDMENT_SHA256
            ),
            "git_blob": (
                prereg.PREREGISTRATION_CORRECTION_AMENDMENT_GIT_BLOB
            ),
            "git_mode": "100644",
            "authority_commit": (
                prereg.PREREGISTRATION_CORRECTION_AMENDMENT_COMMIT
            ),
        },
    ]


def _authority_amendment_bindings(
    preregistration: Mapping[str, Any],
    *,
    expected: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    bindings = preregistration.get("bindings")
    observed = (
        bindings.get("authority_amendments")
        if isinstance(bindings, Mapping)
        else None
    )
    expected_rows = (
        _expected_authority_amendment_bindings()
        if expected is None
        else [dict(row) for row in expected]
    )
    if observed != expected_rows:
        _fail("authority amendment bindings mismatch")
    return expected_rows


_EXPECTED_SECURITY_PROFILE_KEYS = frozenset(
    {
        "identity",
        "expected_branch",
        "authority_commit",
        "protocol_implementation_commit",
        "preregistration_seal_commit",
        "failed_predecessor_preregistrations",
        "failed_predecessor_attempts",
        "failed_predecessor_closures",
        "failed_predecessor_prepublication_closures",
        "failed_predecessor_pre_sentinel_closures",
        "successor_preregistrations",
        "authority_amendments",
        "protocol_paths",
        "protocol_diff",
        "preregistration_diff",
        "claim_diff",
        "publication_diff",
    }
)


def _fixed_security_expectations() -> dict[str, Any]:
    return {
        "failed_predecessor_preregistrations": copy.deepcopy(
            prereg.expected_failed_predecessor_preregistration_bindings()
        ),
        "failed_predecessor_attempts": copy.deepcopy(
            prereg.expected_failed_predecessor_attempts()
        ),
        "failed_predecessor_closures": copy.deepcopy(
            prereg.expected_failed_predecessor_closures()
        ),
        "failed_predecessor_prepublication_closures": copy.deepcopy(
            prereg.expected_failed_predecessor_prepublication_closures()
        ),
        "failed_predecessor_pre_sentinel_closures": copy.deepcopy(
            prereg.expected_failed_predecessor_pre_sentinel_closures()
        ),
        "successor_preregistrations": copy.deepcopy(
            prereg.expected_successor_preregistration_bindings()
        ),
        "authority_amendments": copy.deepcopy(
            _expected_authority_amendment_bindings()
        ),
        "protocol_paths": sorted(
            path.as_posix() for path in prereg.PROTOCOL_PATHS
        ),
        "protocol_diff": list(prereg.SUCCESSOR_PROTOCOL_DIFF),
        "preregistration_diff": list(ACTIVE_PREREGISTRATION_DIFF),
        "claim_diff": list(CLAIM_DIFF),
        "publication_diff": list(PUBLICATION_DIFF),
    }


def _validated_expected_security_profile(
    root: Path,
    preregistration: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    topology_prevalidated: bool = False,
) -> dict[str, Any]:
    required = _EXPECTED_SECURITY_PROFILE_KEYS
    normalized = copy.deepcopy(dict(profile))
    if not required.issubset(normalized) or set(normalized) - (
        required
        | {
            "claim_commit",
            "publication_commit",
        }
    ):
        _fail("expected security profile schema differs")
    for key in (
        "authority_commit",
        "protocol_implementation_commit",
        "preregistration_seal_commit",
    ):
        if not isinstance(normalized.get(key), str) or not re.fullmatch(
            r"[0-9a-f]{40}", normalized[key]
        ):
            _fail(f"expected security profile commit differs: {key}")
    for key in ("claim_commit", "publication_commit"):
        if key in normalized and (
            not isinstance(normalized[key], str)
            or not re.fullmatch(r"[0-9a-f]{40}", normalized[key])
        ):
            _fail(f"expected security profile commit differs: {key}")
    if (
        normalized["identity"] != IDENTITY
        or normalized["expected_branch"] != prereg.EXPECTED_BRANCH
        or preregistration.get("protocol_implementation_commit")
        != normalized["protocol_implementation_commit"]
        or _expected_branch(preregistration) != normalized["expected_branch"]
    ):
        _fail("expected security profile identity binding differs")
    implementation = normalized["protocol_implementation_commit"]
    authority = normalized["authority_commit"]
    seal = normalized["preregistration_seal_commit"]
    if (
        normalized["protocol_diff"] != list(prereg.SUCCESSOR_PROTOCOL_DIFF)
        or normalized["preregistration_diff"]
        != list(ACTIVE_PREREGISTRATION_DIFF)
        or normalized["claim_diff"] != list(CLAIM_DIFF)
        or normalized["publication_diff"] != list(PUBLICATION_DIFF)
        or normalized["protocol_paths"]
        != sorted(path.as_posix() for path in prereg.PROTOCOL_PATHS)
    ):
        _fail("expected security profile path or diff contract differs")
    if not topology_prevalidated:
        if (
            _single_parent_commit(root, implementation) != authority
            or _commit_name_status(root, authority, implementation)
            != tuple(normalized["protocol_diff"])
            or _single_parent_commit(root, seal) != implementation
            or _commit_name_status(root, implementation, seal)
            != tuple(normalized["preregistration_diff"])
        ):
            _fail("expected security profile M12/Q12/P12 topology differs")
    bindings = preregistration.get("bindings")
    if not isinstance(bindings, Mapping):
        _fail("expected security profile preregistration bindings are absent")
    for profile_key, binding_key in (
        (
            "failed_predecessor_preregistrations",
            "failed_predecessor_preregistrations",
        ),
        ("failed_predecessor_attempts", "failed_predecessor_attempts"),
        ("failed_predecessor_closures", "failed_predecessor_closures"),
        (
            "failed_predecessor_prepublication_closures",
            "failed_predecessor_prepublication_closures",
        ),
        (
            "failed_predecessor_pre_sentinel_closures",
            "failed_predecessor_pre_sentinel_closures",
        ),
        ("successor_preregistrations", "successor_preregistrations"),
        ("authority_amendments", "authority_amendments"),
    ):
        if normalized[profile_key] != bindings.get(binding_key):
            _fail(f"expected security profile binding differs: {profile_key}")
    if _planned_protocol_paths(preregistration) != normalized["protocol_paths"]:
        _fail("expected security profile protocol path binding differs")
    return normalized


def _official_expected_security_profile(
    root: Path, preregistration: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        implementation = prereg.validate_protocol_commit_topology(root)
        additions = tuple(
            line
            for line in _git_text(
                root,
                "log",
                "--diff-filter=A",
                "--format=%H",
                "--",
                PREREGISTRATION_PATH.as_posix(),
            ).splitlines()
            if line
        )
    except (
        OSError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as exc:
        raise TerminalG9CB12Failure(
            "official M12/Q12/P12 security topology differs"
        ) from exc
    if len(additions) != 1:
        _fail("official preregistration seal addition history differs")
    seal = additions[0]
    profile = {
        "identity": IDENTITY,
        "expected_branch": prereg.EXPECTED_BRANCH,
        # Historical field name retained for the sealed worker protocol.  For
        # G9CB12 this is the direct Q12 implementation parent, M12.
        "authority_commit": prereg.PROTOCOL_IMPLEMENTATION_PARENT_COMMIT,
        "protocol_implementation_commit": implementation,
        "preregistration_seal_commit": seal,
        **_fixed_security_expectations(),
    }
    return _validated_expected_security_profile(root, preregistration, profile)


def _validate_failed_predecessor_attempt_binding(
    preregistration: Mapping[str, Any],
) -> list[dict[str, Any]]:
    expected = prereg.expected_failed_predecessor_attempts()
    bindings = preregistration.get("bindings")
    observed = (
        bindings.get("failed_predecessor_attempts")
        if isinstance(bindings, Mapping)
        else None
    )
    if observed != expected:
        _fail("failed predecessor attempt closed binding differs")
    return expected


def _validate_failed_predecessor_closure_binding(
    preregistration: Mapping[str, Any],
) -> list[dict[str, Any]]:
    expected = prereg.expected_failed_predecessor_closures()
    bindings = preregistration.get("bindings")
    observed = (
        bindings.get("failed_predecessor_closures")
        if isinstance(bindings, Mapping)
        else None
    )
    if observed != expected:
        _fail("failed predecessor closure binding differs")
    return expected


def _validate_failed_predecessor_prepublication_closure_binding(
    preregistration: Mapping[str, Any],
) -> list[dict[str, Any]]:
    expected = prereg.expected_failed_predecessor_prepublication_closures()
    bindings = preregistration.get("bindings")
    observed = (
        bindings.get("failed_predecessor_prepublication_closures")
        if isinstance(bindings, Mapping)
        else None
    )
    if observed != expected:
        _fail("failed predecessor prepublication closure binding differs")
    return expected


def _validate_failed_predecessor_pre_sentinel_closure_binding(
    preregistration: Mapping[str, Any],
) -> list[dict[str, Any]]:
    expected = prereg.expected_failed_predecessor_pre_sentinel_closures()
    bindings = preregistration.get("bindings")
    observed = (
        bindings.get("failed_predecessor_pre_sentinel_closures")
        if isinstance(bindings, Mapping)
        else None
    )
    if observed != expected:
        _fail("failed predecessor pre-sentinel closure binding differs")
    return expected


def _validate_guarded_prepublication_closure_binding(
    preregistration: Mapping[str, Any], _root: Path
) -> list[dict[str, Any]]:
    return _validate_failed_predecessor_prepublication_closure_binding(
        preregistration
    )


def _validate_failed_predecessor_permanent_state(
    results_fd: int,
    failed_attempts: Sequence[Mapping[str, Any]],
    failed_closures: Sequence[Mapping[str, Any]] = (),
    failed_prepublication_closures: Sequence[Mapping[str, Any]] = (),
    failed_pre_sentinel_closures: Sequence[Mapping[str, Any]] = (),
    *,
    retained_directories: Mapping[str, int] | None = None,
) -> None:
    if not failed_prepublication_closures:
        failed_prepublication_closures = (
            prereg.expected_failed_predecessor_prepublication_closures()
        )
    if not failed_pre_sentinel_closures:
        failed_pre_sentinel_closures = (
            prereg.expected_failed_predecessor_pre_sentinel_closures()
        )
    names = set(_directory_entries(results_fd))

    def absent(path_text: str, message: str) -> None:
        leaf = Path(path_text).name
        try:
            os.stat(leaf, dir_fd=results_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        _fail(message)

    if len(failed_attempts) != 2:
        _fail("failed predecessor attempt permanent-state schema differs")
    for row in failed_attempts:
        identity = str(row["identity"])
        for path_text in row["permanently_absent_outputs"]:
            absent(
                str(path_text),
                f"permanently absent {identity} output exists: {path_text}",
            )
        bytecode = row["residue"].get("bytecode_cache")
        if isinstance(bytecode, Mapping):
            absent(
                str(bytecode["path"]),
                f"{identity} bytecode residue differs",
            )
        slot1_leaf = Path(
            str(row["residue"]["slot1_stage"]["path"])
        ).name
        path_text = str(row["residue"]["slot1_stage"]["path"])
        stage_fd = (
            retained_directories.get(path_text)
            if retained_directories is not None
            else None
        )
        owned_stage_fd = stage_fd is None
        if stage_fd is None:
            stage_fd = os.open(
                slot1_leaf,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=results_fd,
            )
        try:
            info = os.fstat(stage_fd)
            if (
                not stat.S_ISDIR(info.st_mode)
                or stat.S_IMODE(info.st_mode) != 0o700
                or _directory_entries(stage_fd)
            ):
                _fail(f"{identity} slot-1 residue state differs")
        finally:
            if owned_stage_fd:
                os.close(stage_fd)
        absent(
            str(row["residue"]["slot2_stage"]["path"]),
            f"{identity} slot-2 residue state differs",
        )
    if (
        len(failed_closures) != 1
        or failed_closures[0].get("identity") != "G9CB-4"
        or not isinstance(failed_closures[0].get("residue"), Mapping)
        or not isinstance(
            failed_closures[0].get("permanently_absent_outputs"), list
        )
    ):
        _fail("failed predecessor closure permanent-state schema differs")
    for row in failed_closures:
        for path_text in row["permanently_absent_outputs"]:
            absent(
                str(path_text),
                f"permanently absent G9CB-4 output exists: {path_text}",
            )
        residue = row["residue"]
        absent(
            str(residue["bytecode_cache"]["path"]),
            "G9CB-4 bytecode residue differs",
        )
        if any(
            name.startswith(".gross9_structural_clock_bundle_g9cb4_")
            and ".stage-" in name
            for name in names
        ):
            _fail("G9CB-4 publication-stage residue differs")
        if any(
            name.startswith(".gross9-structural-clock-g9cb4-worker-")
            for name in names
        ):
            _fail("G9CB-4 worker-stage residue differs")
    if tuple(
        row.get("identity") for row in failed_prepublication_closures
    ) != ("G9CB-5", "G9CB-6"):
        _fail("failed predecessor prepublication closure schema differs")
    for row in failed_prepublication_closures:
        identity = str(row["identity"])
        suffix = identity.lower().replace("-", "")
        for path_text in row["permanently_absent_outputs"]:
            absent(
                str(path_text),
                f"permanently absent {identity} output exists: {path_text}",
            )
        residue = row["residue"]
        absent(
            str(residue["bytecode_cache"]["path"]),
            f"{identity} bytecode residue differs",
        )
        if any(
            name.startswith(f".gross9_structural_clock_bundle_{suffix}_")
            and ".stage-" in name
            for name in names
        ):
            _fail(f"{identity} publication-stage residue differs")
        if any(
            name.startswith(f".gross9-structural-clock-{suffix}-worker-")
            for name in names
        ):
            _fail(f"{identity} worker-stage residue differs")
        if isinstance(residue.get("capability_probes"), Mapping) and any(
            name.startswith(f".{suffix}-otmpfile-probe-") for name in names
        ):
            _fail(f"{identity} capability-probe residue differs")
    if tuple(
        row.get("identity") for row in failed_pre_sentinel_closures
    ) != ("G9CB-7",):
        _fail("failed predecessor pre-sentinel closure schema differs")
    for row in failed_pre_sentinel_closures:
        for path_text in row["permanently_absent_outputs"]:
            absent(
                str(path_text),
                f"permanently absent G9CB-7 output exists: {path_text}",
            )
        residue = row["residue"]
        absent(
            str(residue["bytecode_cache"]["path"]),
            "G9CB-7 bytecode residue differs",
        )
        if any(
            name.startswith(".gross9_structural_clock_bundle_g9cb7_")
            and ".stage-" in name
            for name in names
        ):
            _fail("G9CB-7 publication-stage residue differs")
        if any(
            name.startswith(".gross9-structural-clock-g9cb7-worker-")
            for name in names
        ):
            _fail("G9CB-7 worker-stage residue differs")
        if any(name.startswith(".g9cb7-otmpfile-probe-") for name in names):
            _fail("G9CB-7 capability-probe residue differs")


def _validate_predecessor_inventory_standalone(
    root: Path,
    failed_attempts: Sequence[Mapping[str, Any]],
    failed_closures: Sequence[Mapping[str, Any]],
    failed_prepublication_closures: Sequence[Mapping[str, Any]],
    failed_pre_sentinel_closures: Sequence[Mapping[str, Any]],
) -> None:
    context = _PublicationContext(root)
    try:
        _validate_failed_predecessor_permanent_state(
            context.results_fd,
            failed_attempts,
            failed_closures,
            failed_prepublication_closures,
            failed_pre_sentinel_closures,
        )
    finally:
        context.close()


def _validate_guarded_preregistration_authentication(
    preregistration_binding: Mapping[str, Any],
    implementation_commit: str,
    parent_authentication: Mapping[str, Any] | None,
    claim_preregistration: Mapping[str, Any] | None,
) -> None:
    if not isinstance(parent_authentication, Mapping) or set(
        parent_authentication
    ) != {
        "environment",
        "hashed_inputs",
        "preregistration_authentication",
        "runtime_import_closure",
    }:
        _fail("parent authentication schema differs")
    record = parent_authentication.get("preregistration_authentication")
    expected_keys = {
        "manifest_hash",
        "path",
        "protocol_implementation_commit",
        "sha256",
    }
    if not isinstance(record, Mapping) or set(record) != expected_keys:
        _fail("preregistration authentication schema differs")
    if (
        type(record.get("path")) is not str
        or record["path"] != PREREGISTRATION_PATH.as_posix()
        or type(record.get("sha256")) is not str
        or not _SHA_RE.fullmatch(record["sha256"])
        or type(record.get("manifest_hash")) is not str
        or not _SHA_RE.fullmatch(record["manifest_hash"])
        or type(record.get("protocol_implementation_commit")) is not str
        or not re.fullmatch(
            r"[0-9a-f]{40}", record["protocol_implementation_commit"]
        )
    ):
        _fail("preregistration authentication field shape differs")
    expected_record = {
        "manifest_hash": preregistration_binding["manifest_hash"],
        "path": preregistration_binding["path"],
        "protocol_implementation_commit": implementation_commit,
        "sha256": preregistration_binding["sha256"],
    }
    if dict(record) != expected_record:
        _fail("preregistration authentication field binding differs")
    if not isinstance(claim_preregistration, Mapping) or dict(
        claim_preregistration
    ) != dict(preregistration_binding):
        _fail("claim preregistration binding differs")


def validate_preregistration(
    root: Path = REPOSITORY_ROOT,
    *,
    validation_mode: str = "parent",
    parent_authentication: Mapping[str, Any] | None = None,
    claim_preregistration: Mapping[str, Any] | None = None,
    raw_cache: Mapping[str, tuple[bytes, os.stat_result]] | None = None,
    expected_security_profile: Mapping[str, Any] | None = None,
    profile_topology_prevalidated: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate the canonical preregistration without opening source values."""

    if validation_mode not in {"parent", "guarded_worker", "synthetic"}:
        _fail("preregistration validation mode is invalid")
    path = _rooted(root, PREREGISTRATION_PATH)
    payload, raw, info = _read_canonical_object(
        path,
        "manifest_hash",
        path_text=PREREGISTRATION_PATH.as_posix(),
        raw_cache=raw_cache,
    )
    if payload.get("protocol_version") != prereg.PROTOCOL_VERSION:
        _fail("operative preregistration protocol version mismatch")
    if payload.get("identity") != IDENTITY:
        _fail("preregistration identity mismatch")
    _validate_zero_access(payload)
    if stat.S_IMODE(info.st_mode) != 0o444:
        _fail("active preregistration filesystem mode differs")
    independence = payload.get("candidate_independence")
    if not isinstance(independence, Mapping) or independence.get(
        "candidate_identity_present"
    ) is not False:
        _fail("candidate identity is present")
    if independence.get("candidate_artifacts_opened") is not False:
        _fail("candidate artifact access is present")
    bindings = payload.get("bindings")
    if not isinstance(bindings, Mapping):
        _fail("preregistration bindings object is absent")
    security_profile = (
        _validated_expected_security_profile(
            root,
            payload,
            expected_security_profile,
            topology_prevalidated=profile_topology_prevalidated,
        )
        if expected_security_profile is not None
        else None
    )
    required_binding_keys = {
        "protocol",
        "authority_amendments",
        "failed_predecessor_preregistrations",
        "failed_predecessor_attempts",
        "direct_authority",
        "config_metadata_evidence",
        "runtime_import_roots",
        "runtime_import_closure",
        "rank7_bundle",
        "source_manifest_ordered_inventory",
        "environment",
    }
    if not required_binding_keys.issubset(bindings):
        _fail("preregistration bindings schema is incomplete")
    successor_binding_keys = {
        "failed_predecessor_closures",
        "failed_predecessor_prepublication_closures",
        "failed_predecessor_pre_sentinel_closures",
        "successor_preregistrations",
    }
    missing_successor_keys = successor_binding_keys.difference(bindings)
    if (
        missing_successor_keys
        and root == REPOSITORY_ROOT
        and (root / ".git").exists()
    ):
        _fail("preregistration bindings schema is incomplete")
    for prohibited in (
        "authority_amendment",
        "superseded_preregistration",
        "adapter_import_roots",
        "adapter_import_closure",
    ):
        if prohibited in bindings:
            _fail(f"superseded preregistration binding is present: {prohibited}")
    _authority_amendment_bindings(
        payload,
        expected=(
            security_profile["authority_amendments"]
            if security_profile is not None
            else None
        ),
    )
    if security_profile is not None:
        predecessors = security_profile[
            "failed_predecessor_preregistrations"
        ]
        failed_attempts = security_profile["failed_predecessor_attempts"]
        failed_closures = security_profile["failed_predecessor_closures"]
        failed_prepublication_closures = security_profile[
            "failed_predecessor_prepublication_closures"
        ]
        failed_pre_sentinel_closures = security_profile[
            "failed_predecessor_pre_sentinel_closures"
        ]
    elif validation_mode == "synthetic":
        if (root / ".git").exists():
            _fail("synthetic preregistration hook requires a noncanonical root")
        predecessors = (
            prereg.expected_failed_predecessor_preregistration_bindings()
        )
        failed_attempts = prereg.expected_failed_predecessor_attempts()
        failed_closures = prereg.expected_failed_predecessor_closures()
        failed_prepublication_closures = (
            prereg.expected_failed_predecessor_prepublication_closures()
        )
        failed_pre_sentinel_closures = (
            prereg.expected_failed_predecessor_pre_sentinel_closures()
        )
    elif (
        validation_mode == "guarded_worker"
        and "failed_predecessor_closures" in bindings
    ):
        predecessors = (
            prereg.expected_failed_predecessor_preregistration_bindings()
        )
        failed_attempts = _validate_failed_predecessor_attempt_binding(
            payload
        )
        failed_closures = _validate_failed_predecessor_closure_binding(
            payload
        )
        failed_prepublication_closures = (
            _validate_guarded_prepublication_closure_binding(payload, root)
        )
        failed_pre_sentinel_closures = (
            _validate_failed_predecessor_pre_sentinel_closure_binding(payload)
        )
    elif validation_mode == "guarded_worker":
        predecessors = (
            prereg.expected_failed_predecessor_preregistration_bindings()
        )
        failed_attempts = _validate_failed_predecessor_attempt_binding(
            payload
        )
        failed_closures = prereg.expected_failed_predecessor_closures()
        failed_prepublication_closures = (
            _validate_guarded_prepublication_closure_binding(payload, root)
        )
        failed_pre_sentinel_closures = (
            _validate_failed_predecessor_pre_sentinel_closure_binding(payload)
        )
    else:
        try:
            prereg.validate_historical_preregistration_topology(root)
            prereg.validate_failed_v2_preregistration_topology(root)
            prereg.validate_failed_predecessor_attempt_history(root)
            predecessors = (
                prereg.expected_failed_predecessor_preregistration_bindings()
            )
            failed_attempts = prereg.expected_failed_predecessor_attempts()
            failed_closures = prereg.expected_failed_predecessor_closures()
            failed_prepublication_closures = (
                prereg.expected_failed_predecessor_prepublication_closures()
            )
            failed_pre_sentinel_closures = (
                prereg.validate_failed_predecessor_pre_sentinel_closures(root)
                if root == REPOSITORY_ROOT and (root / ".git").exists()
                else prereg.expected_failed_predecessor_pre_sentinel_closures()
            )
        except (
            OSError,
            RuntimeError,
            ValueError,
            subprocess.CalledProcessError,
        ) as exc:
            raise TerminalG9CB12Failure(
                "failed predecessor preregistration evidence differs"
            ) from exc
    if validation_mode in {"parent", "guarded_worker"}:
        if isinstance(raw_cache, _SecureBoundSnapshot):
            results_fd = raw_cache.directory_descriptors.get(
                ("repo", ("results",))
            )
            if results_fd is None:
                _fail("predecessor path-state results descriptor is absent")
            _validate_failed_predecessor_permanent_state(
                results_fd,
                failed_attempts,
                failed_closures,
                failed_prepublication_closures,
                failed_pre_sentinel_closures,
            )
        else:
            _validate_predecessor_inventory_standalone(
                root,
                failed_attempts,
                failed_closures,
                failed_prepublication_closures,
                failed_pre_sentinel_closures,
            )
    if bindings.get("failed_predecessor_preregistrations") != predecessors:
        _fail("failed predecessor preregistration bindings mismatch")
    if bindings.get("failed_predecessor_attempts") != failed_attempts:
        _fail("failed predecessor attempt binding mismatch")
    if (
        "failed_predecessor_closures" in bindings
        and bindings.get("failed_predecessor_closures") != failed_closures
    ):
        _fail("failed predecessor closure binding mismatch")
    if (
        bindings.get("failed_predecessor_prepublication_closures")
        != failed_prepublication_closures
    ):
        _fail("failed predecessor prepublication closure binding mismatch")
    if (
        bindings.get("failed_predecessor_pre_sentinel_closures")
        != failed_pre_sentinel_closures
    ):
        _fail("failed predecessor pre-sentinel closure binding mismatch")
    successor_rows = bindings.get("successor_preregistrations")
    if "successor_preregistrations" in bindings:
        expected_successors = (
            prereg.expected_successor_preregistration_bindings()
            if security_profile is None
            else security_profile["successor_preregistrations"]
        )
        if successor_rows != expected_successors:
            _fail("successor preregistration bindings mismatch")
    recorded_implementation = payload.get("protocol_implementation_commit")
    if not isinstance(recorded_implementation, str) or not re.fullmatch(
        r"[0-9a-f]{40}", recorded_implementation
    ):
        _fail("protocol implementation commit is invalid")
    if security_profile is not None:
        implementation = security_profile[
            "protocol_implementation_commit"
        ]
    elif validation_mode in {"synthetic", "guarded_worker"}:
        implementation = recorded_implementation
    else:
        try:
            implementation = prereg.validate_protocol_commit_topology(root)
        except (
            OSError,
            RuntimeError,
            ValueError,
            subprocess.CalledProcessError,
        ) as exc:
            raise TerminalG9CB12Failure(
                "protocol implementation topology differs"
            ) from exc
    if recorded_implementation != implementation:
        _fail("protocol implementation commit mismatch")
    git_seal = payload.get("git_seal")
    if not isinstance(git_seal, Mapping) or not isinstance(
        git_seal.get("expected_branch"), str
    ):
        _fail("preregistration git_seal is incomplete")

    helper = getattr(prereg, "validate_manifest", None)
    if (
        validation_mode == "parent"
        and security_profile is None
        and callable(helper)
    ):
        try:
            helper(
                payload,
                repository_root=root,
                verify_files=False,
                verify_environment=False,
                verify_git_seal=False,
            )
        except (TypeError, ValueError) as exc:
            raise TerminalG9CB12Failure(
                "preregistration producer validation failed"
            ) from exc

    preregistration_binding = {
        "path": PREREGISTRATION_PATH.as_posix(),
        "sha256": _sha256_bytes(raw),
        "manifest_hash": payload["manifest_hash"],
    }
    if validation_mode == "guarded_worker":
        _validate_guarded_preregistration_authentication(
            preregistration_binding,
            recorded_implementation,
            parent_authentication,
            claim_preregistration,
        )
    return payload, preregistration_binding


def _authenticate_guarded_worker_metadata(
    root: Path,
    parent_authentication: Mapping[str, Any],
    claim_preregistration: Mapping[str, Any] | None,
    *,
    raw_cache: dict[str, tuple[bytes, os.stat_result]] | None = None,
    expected_security_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cache = {} if raw_cache is None else raw_cache
    preregistration, preregistration_binding = validate_preregistration(
        root,
        validation_mode="guarded_worker",
        parent_authentication=parent_authentication,
        claim_preregistration=claim_preregistration,
        raw_cache=cache,
        expected_security_profile=expected_security_profile,
        profile_topology_prevalidated=True,
    )
    hashed_inputs = _validate_regular_hashed_inputs(
        root,
        preregistration,
        verify_git=False,
        raw_cache=cache,
    )
    closures = _validate_static_closures(
        root,
        preregistration,
        verify_git=False,
        raw_cache=cache,
    )
    worker_authentication = {
        "environment": _validate_environment(preregistration, root),
        "hashed_inputs": hashed_inputs,
        "preregistration_authentication": {
            "manifest_hash": preregistration_binding["manifest_hash"],
            "path": preregistration_binding["path"],
            "protocol_implementation_commit": preregistration[
                "protocol_implementation_commit"
            ],
            "sha256": preregistration_binding["sha256"],
        },
        "runtime_import_closure": closures["runtime"],
    }
    if worker_authentication != dict(parent_authentication):
        _fail("worker authentication differs from the parent seal")
    return {
        "preregistration": preregistration,
        "preregistration_binding": preregistration_binding,
        "closures": closures,
        "authentication": worker_authentication,
    }


def _prepare_worker_metadata_snapshot(
    root: Path,
    guard: _WorkerIsolationGuard | None,
    *,
    extra_repository_paths: Sequence[str] = (),
    expected_security_profile: Mapping[str, Any] | None = None,
) -> _SecureBoundSnapshot:
    if guard is not None and (
        guard.repository_fd is None
        or guard.filesystem_root_fd is None
        or guard.results_fd is None
    ):
        _fail("worker metadata snapshot anchors are absent")
    if guard is None:
        snapshot = _SecureBoundSnapshot(root)
    else:
        snapshot = _SecureBoundSnapshot(
            root,
            repository_fd=guard.repository_fd,
            filesystem_root_fd=guard.filesystem_root_fd,
            opener=guard._original_os_open,
            register_descriptor=guard.register_snapshot_descriptor,
        )
        results_info = os.fstat(guard.results_fd)
        snapshot.directory_descriptors[("repo", ("results",))] = (
            guard.results_fd
        )
        snapshot.directory_tokens[("repo", ("results",))] = (
            _descriptor_token(results_info)
        )
        snapshot.directory_edges[("repo", ("results",))] = (
            guard.repository_fd,
            "results",
        )
        snapshot._borrowed_descriptors.add(guard.results_fd)
    try:
        for path in (PREREGISTRATION_PATH, CLAIM_PATH, SENTINEL_PATH):
            snapshot.open_initial(path.as_posix(), True)
        preregistration = _canonical_bound_json(
            snapshot[PREREGISTRATION_PATH.as_posix()][0],
            PREREGISTRATION_PATH.as_posix(),
            "manifest_hash",
        )
        _retain_expected_predecessor_residue_directories(
            snapshot,
            preregistration,
            (
                None
                if expected_security_profile is None
                else expected_security_profile[
                    "failed_predecessor_attempts"
                ]
            ),
        )
        prepared: dict[str, bool] = {
            PREREGISTRATION_PATH.as_posix(): True,
            CLAIM_PATH.as_posix(): True,
            SENTINEL_PATH.as_posix(): True,
        }
        for binding in _iter_bindings(preregistration):
            path_text = _binding_path(binding)
            _candidate, repository_relative = _bound_regular_path(
                root, path_text
            )
            prior = prepared.get(path_text)
            if prior is not None and prior != repository_relative:
                _fail(f"conflicting worker metadata path: {path_text}")
            prepared[path_text] = repository_relative
        bindings = preregistration.get("bindings")
        protocol = (
            bindings.get("protocol")
            if isinstance(bindings, Mapping)
            else None
        )
        if protocol is not None:
            for path_text in _planned_protocol_paths(preregistration):
                prepared.setdefault(path_text, True)
        for path_text in extra_repository_paths:
            _candidate, repository_relative = _bound_regular_path(
                root, path_text
            )
            if not repository_relative:
                _fail("worker extra snapshot path is not repository-relative")
            prepared.setdefault(path_text, True)
        for path_text in sorted(prepared):
            snapshot.open_initial(path_text, prepared[path_text])
        return snapshot
    except BaseException:
        snapshot.close()
        raise


def _authenticate_worker_metadata_entry(
    root: Path,
    parent_authentication: Mapping[str, Any],
    *,
    synthetic: bool,
    synthetic_input_path: str | None = None,
    expected_security_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticate active metadata modes and bytes from one snapshot each."""

    guard = _ACTIVE_WORKER_GUARD
    metadata_snapshot = _prepare_worker_metadata_snapshot(
        root,
        guard if isinstance(guard, _WorkerIsolationGuard) else None,
        extra_repository_paths=(
            () if synthetic_input_path is None else (synthetic_input_path,)
        ),
        expected_security_profile=expected_security_profile,
    )
    metadata_cache: dict[str, tuple[bytes, os.stat_result]] = metadata_snapshot
    sentinel, sentinel_raw, sentinel_info = _read_canonical_object(
        _rooted(root, SENTINEL_PATH),
        "manifest_hash",
        path_text=SENTINEL_PATH.as_posix(),
        raw_cache=metadata_cache,
    )
    claim, claim_raw, claim_info = _read_canonical_object(
        _rooted(root, CLAIM_PATH),
        "claim_hash",
        path_text=CLAIM_PATH.as_posix(),
        raw_cache=metadata_cache,
    )
    if stat.S_IMODE(sentinel_info.st_mode) != 0o444:
        _fail("active sentinel filesystem mode differs")
    if stat.S_IMODE(claim_info.st_mode) != 0o444:
        _fail("active claim filesystem mode differs")
    guarded_metadata = _authenticate_guarded_worker_metadata(
        root,
        parent_authentication,
        claim.get("preregistration"),
        raw_cache=metadata_cache,
        expected_security_profile=expected_security_profile,
    )
    preregistration = guarded_metadata["preregistration"]
    preregistration_binding = guarded_metadata[
        "preregistration_binding"
    ]
    return {
        "claim": claim,
        "claim_raw": claim_raw,
        "sentinel": sentinel,
        "sentinel_raw": sentinel_raw,
        "preregistration": preregistration,
        "preregistration_binding": preregistration_binding,
        "guarded_metadata": guarded_metadata,
        "raw_cache": metadata_cache,
        "snapshot": metadata_snapshot,
    }


def _expected_branch(preregistration: Mapping[str, Any]) -> str | None:
    git_seal = preregistration.get("git_seal")
    if not isinstance(git_seal, Mapping):
        _fail("preregistration git_seal is absent")
    value = git_seal.get("expected_branch")
    if not isinstance(value, str) or not value:
        _fail("preregistration expected branch is absent")
    return value


def _planned_protocol_paths(preregistration: Mapping[str, Any]) -> list[str]:
    bindings = preregistration.get("bindings")
    protocol = bindings.get("protocol") if isinstance(bindings, Mapping) else None
    if not isinstance(protocol, list) or not protocol:
        _fail("preregistration protocol bindings are absent")
    paths: list[str] = []
    for row in protocol:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            _fail("invalid preregistration protocol binding")
        paths.append(row["path"])
    if len(paths) != len(set(paths)):
        _fail("preregistration protocol bindings contain duplicates")
    required = {BUILDER_PATH.as_posix(), BUILDER_TEST_PATH.as_posix()}
    if not required.issubset(paths):
        _fail("builder protocol files are absent from preregistration")
    return paths


def _head_blob_binding(
    root: Path,
    path: str,
    *,
    raw_cache: Mapping[str, tuple[bytes, os.stat_result]] | None = None,
    preclassified_pairs: Mapping[str, tuple[str, str] | None] | None = None,
) -> dict[str, str]:
    if raw_cache is not None or preclassified_pairs is not None:
        cached = raw_cache.get(path) if raw_cache is not None else None
        pair = (
            preclassified_pairs.get(path)
            if preclassified_pairs is not None
            else None
        )
        if cached is None or pair is None:
            _fail(f"required protocol path lacks snapshot binding: {path}")
        raw, info = cached
        if stat.S_IMODE(info.st_mode) & 0o111 or pair[1] != "100644":
            _fail(f"required protocol mode differs: {path}")
        if _git_blob_id(raw) != pair[0]:
            _fail(f"required protocol path differs from HEAD: {path}")
        return {
            "path": path,
            "sha256": _sha256_bytes(raw),
            "git_blob": pair[0],
            "mode": pair[1],
        }
    pair = _validate_git_pair_preflight(
        root,
        path,
        repository_relative=True,
        declaration={},
        verify_git=True,
        require_tracked=True,
    )
    if pair is None or pair[1] != "100644":
        _fail(f"required protocol mode differs: {path}")
    raw, info = _read_bound_regular_bytes(_rooted(root, path), path)
    if stat.S_IMODE(info.st_mode) & 0o111:
        _fail(f"required protocol mode differs: {path}")
    if _git_blob_id(raw) != pair[0]:
        _fail(f"required protocol path differs from HEAD: {path}")
    return {
        "path": path,
        "sha256": _sha256_bytes(raw),
        "git_blob": pair[0],
        "mode": pair[1],
    }


def _tracked_head_bytes(
    root: Path,
    relative: Path,
    *,
    expected_mode: int = 0o444,
) -> bytes:
    path_text = relative.as_posix()
    pair = _validate_git_pair_preflight(
        root,
        path_text,
        repository_relative=True,
        declaration={},
        verify_git=True,
        require_tracked=True,
    )
    raw, info = _read_bound_regular_bytes(_rooted(root, relative), path_text)
    if stat.S_IMODE(info.st_mode) != expected_mode:
        _fail(f"committed artifact filesystem mode differs: {path_text}")
    if pair is None or pair[1] != "100644" or _git_blob_id(raw) != pair[0]:
        _fail(f"committed artifact bytes differ from HEAD: {path_text}")
    return raw


def _claim_payload(
    parent_commit: str,
    prereg_binding: Mapping[str, Any],
    authority_amendments: Sequence[Mapping[str, Any]],
    protocol_bindings: Sequence[Mapping[str, str]],
    opaque_inputs: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    core = {
        "identity": IDENTITY,
        "protocol_version": PROTOCOL_VERSION,
        "status": "pre_access_claim_created",
        "protocol_parent_commit": parent_commit,
        "preregistration": dict(prereg_binding),
        "authority_amendments": [
            dict(row) for row in authority_amendments
        ],
        "protocol_files": [dict(row) for row in protocol_bindings],
        "opaque_inputs_authenticated": [dict(row) for row in opaque_inputs],
        "domain": {"start": DOMAIN_START, "end_exclusive": DOMAIN_END},
        "canonical_outputs": {
            "attempt_consumed": SENTINEL_PATH.as_posix(),
            "worker_capability_consumption_ledgers": [
                path.as_posix() for path in WORKER_LEDGER_PATHS
            ],
            "interval_csv_gzip": CSV_PATH.as_posix(),
            "final_manifest": MANIFEST_PATH.as_posix(),
        },
        "access_at_claim": {
            "runtime_modules_imported": 0,
            "source_value_rows_opened": 0,
            "pre2025_anchor_value_rows_opened": 0,
            "candidate_rows_opened": 0,
            "comparator_clock_rows_opened": 0,
        },
        "candidate_identity_present": False,
        "candidate_artifacts_opened": False,
        "one_shot": True,
        "retry_allowed": False,
        "resume_allowed": False,
    }
    return _with_hash(core, "claim_hash")


_AT_FDCWD = -100
_AT_SYMLINK_FOLLOW = 0x400


def _link_unnamed_procfd_raw(
    unnamed_fd: int, results_fd: int, leaf: str
) -> None:
    source = f"/proc/self/fd/{unnamed_fd}".encode("ascii")
    destination = os.fsencode(leaf)
    result = _LIBC.linkat(
        ctypes.c_int(_AT_FDCWD),
        ctypes.c_char_p(source),
        ctypes.c_int(results_fd),
        ctypes.c_char_p(destination),
        ctypes.c_int(_AT_SYMLINK_FOLLOW),
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(f"write-once path exists: {leaf}")
        raise OSError(error, os.strerror(error), leaf)


def _link_unnamed_procfd(
    unnamed_fd: int, results_fd: int, leaf: str
) -> None:
    _link_unnamed_procfd_raw(unnamed_fd, results_fd, leaf)


def _openat_component(
    directory_fd: int,
    leaf: str,
    flags: int,
    mode: int = 0,
) -> int:
    if (
        not leaf
        or leaf in {".", ".."}
        or "/" in leaf
        or "\\" in leaf
        or "\x00" in leaf
    ):
        _fail("openat leaf component is invalid")
    descriptor = _LIBC.openat(
        ctypes.c_int(directory_fd),
        ctypes.c_char_p(os.fsencode(leaf)),
        ctypes.c_int(flags),
        ctypes.c_uint(mode),
    )
    if descriptor < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), leaf)
    return int(descriptor)


def _directory_entries(descriptor: int) -> tuple[str, ...]:
    guard = _ACTIVE_WORKER_GUARD
    listdir = (
        guard._originals.get((id(os), "listdir"), os.listdir)
        if isinstance(guard, _WorkerIsolationGuard)
        else os.listdir
    )
    return tuple(sorted(listdir(descriptor)))


def _read_publication_descriptor(descriptor: int, size: int) -> bytes:
    return _pread_complete(descriptor, size, True)


def _open_unnamed_completed(
    results_fd: int, raw: bytes, *, mode: int
) -> tuple[int, os.stat_result]:
    flags = (
        os.O_RDWR
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_TMPFILE", 0)
    )
    if not getattr(os, "O_TMPFILE", 0):
        _fail("O_TMPFILE is unavailable")
    descriptor = os.open(".", flags, 0o600, dir_fd=results_fd)
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode) or initial.st_size != 0:
            _fail("unnamed publication inode is not an empty regular file")
        _write_all(descriptor, raw)
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        completed = os.fstat(descriptor)
        if (
            (initial.st_dev, initial.st_ino, stat.S_IFMT(initial.st_mode))
            != (
                completed.st_dev,
                completed.st_ino,
                stat.S_IFMT(completed.st_mode),
            )
            or stat.S_IMODE(completed.st_mode) != mode
            or completed.st_size != len(raw)
            or _read_publication_descriptor(descriptor, completed.st_size)
            != raw
        ):
            _fail("unnamed publication same-FD verification failed")
        return descriptor, completed
    except NameError as exc:
        os.close(descriptor)
        raise TerminalG9CB12Failure(
            "publication helper failure"
        ) from exc
    except BaseException:
        os.close(descriptor)
        raise


def _publish_h12_leaf(
    root: Path,
    relative: Path,
    raw: bytes,
    *,
    expected_entries: tuple[str, ...],
    prelink_recheck: Any | None = None,
) -> dict[str, Any]:
    if (
        relative.parent != Path("results")
        or relative.name in {"", ".", ".."}
        or "/" in relative.name
        or "\\" in relative.name
    ):
        _fail("H12 publication path is not an exact results leaf")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    repository_fd = os.open(root, flags)
    results_fd = canonical_fd = unnamed_fd = -1
    try:
        results_fd = os.open("results", flags, dir_fd=repository_fd)
        repository_token = _directory_identity(os.fstat(repository_fd))
        results_token = _directory_identity(os.fstat(results_fd))
        results_edge = os.stat(
            "results", dir_fd=repository_fd, follow_symlinks=False
        )
        if (
            _directory_identity(results_edge) != results_token
            or _directory_entries(results_fd) != expected_entries
            or relative.name in expected_entries
        ):
            _fail("H12 publication prelink inventory differs")
        unnamed_fd, unnamed_info = _open_unnamed_completed(
            results_fd, raw, mode=0o444
        )
        if prelink_recheck is not None:
            prelink_recheck()
        if (
            _directory_identity(os.fstat(repository_fd)) != repository_token
            or _directory_identity(os.fstat(results_fd)) != results_token
            or _directory_entries(results_fd) != expected_entries
        ):
            _fail("H12 publication directory binding drifted before link")
        _link_unnamed_procfd_raw(unnamed_fd, results_fd, relative.name)
        canonical_fd = os.open(
            relative.name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=results_fd,
        )
        canonical_info = os.fstat(canonical_fd)
        linked_info = os.fstat(unnamed_fd)
        canonical_raw = _read_publication_descriptor(
            canonical_fd, canonical_info.st_size
        )
        added = tuple(sorted((*expected_entries, relative.name)))
        if (
            not stat.S_ISREG(canonical_info.st_mode)
            or stat.S_IMODE(canonical_info.st_mode) != 0o444
            or canonical_info.st_nlink != 1
            or (canonical_info.st_dev, canonical_info.st_ino)
            != (unnamed_info.st_dev, unnamed_info.st_ino)
            or (canonical_info.st_dev, canonical_info.st_ino)
            != (linked_info.st_dev, linked_info.st_ino)
            or canonical_info.st_size != len(raw)
            or canonical_raw != raw
            or _directory_entries(results_fd) != added
        ):
            _fail("H12 canonical publication inode/bytes/delta differs")
        os.fsync(unnamed_fd)
        os.fsync(results_fd)
        if (
            _directory_identity(os.fstat(repository_fd)) != repository_token
            or _directory_identity(os.fstat(results_fd)) != results_token
            or _directory_entries(results_fd) != added
            or _read_publication_descriptor(
                canonical_fd, canonical_info.st_size
            )
            != raw
        ):
            _fail("H12 publication post-fsync verification differs")
        return {
            "mode": stat.S_IMODE(canonical_info.st_mode),
            "path": relative.as_posix(),
            "sha256": _sha256_bytes(raw),
            "size_bytes": canonical_info.st_size,
        }
    finally:
        for descriptor in (canonical_fd, unnamed_fd, results_fd, repository_fd):
            if descriptor >= 0:
                os.close(descriptor)


class _PublicationContext:
    """Retained results-dir publisher using only unnamed O_TMPFILE inodes."""

    def __init__(
        self,
        root: Path,
        *,
        directory_relative: str = "results",
        expected_security_profile: Mapping[str, Any] | None = None,
    ) -> None:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        self.root = root
        self.repository_fd = os.open(root, flags)
        try:
            self.results_fd = os.open(
                directory_relative, flags, dir_fd=self.repository_fd
            )
        except BaseException:
            os.close(self.repository_fd)
            raise
        self.repository_token = _directory_identity(
            os.fstat(self.repository_fd)
        )
        self.results_token = _directory_identity(os.fstat(self.results_fd))
        self.entries = _directory_entries(self.results_fd)
        self._timestamp_generation = 0
        self.timestamp_token = (
            *_descriptor_token(os.fstat(self.results_fd)),
            self._timestamp_generation,
        )
        self.probed = False
        self.directory_relative = directory_relative
        self.stage_descriptors: dict[str, int] = {}
        self.stage_tokens: dict[str, tuple[int, ...]] = {}
        self.stage_entries: dict[str, tuple[str, ...]] = {}
        self.predecessor_descriptors: dict[str, int] = {}
        self.predecessor_tokens: dict[str, tuple[int, ...]] = {}
        self.failed_predecessor_attempts = copy.deepcopy(
            prereg.expected_failed_predecessor_attempts()
            if expected_security_profile is None
            else expected_security_profile["failed_predecessor_attempts"]
        )
        self.failed_predecessor_closures = copy.deepcopy(
            prereg.expected_failed_predecessor_closures()
            if expected_security_profile is None
            else expected_security_profile["failed_predecessor_closures"]
        )
        self.failed_predecessor_prepublication_closures = copy.deepcopy(
            prereg.expected_failed_predecessor_prepublication_closures()
            if expected_security_profile is None
            else expected_security_profile[
                "failed_predecessor_prepublication_closures"
            ]
        )
        self.failed_predecessor_pre_sentinel_closures = copy.deepcopy(
            prereg.expected_failed_predecessor_pre_sentinel_closures()
            if expected_security_profile is None
            else expected_security_profile[
                "failed_predecessor_pre_sentinel_closures"
            ]
        )

    def _require_bound_results(
        self, *, compare_timestamps: bool = True
    ) -> None:
        repository_info = os.fstat(self.repository_fd)
        results_info = os.fstat(self.results_fd)
        path_info = os.stat(
            self.directory_relative,
            dir_fd=self.repository_fd,
            follow_symlinks=False,
        )
        if (
            _directory_identity(repository_info) != self.repository_token
            or _directory_identity(results_info) != self.results_token
            or _directory_identity(path_info) != self.results_token
        ):
            _fail("retained repository/results directory identity drifted")
        if (
            compare_timestamps
            and _descriptor_token(results_info) != self.timestamp_token[:-1]
        ):
            _fail("retained results-directory timestamps/inventory drifted")

    def _rebaseline(self, expected_entries: tuple[str, ...]) -> None:
        self._require_bound_results(compare_timestamps=False)
        if _directory_entries(self.results_fd) != expected_entries:
            _fail("results-directory entry inventory differs")
        if len(set(self.entries) ^ set(expected_entries)) != 1:
            _fail("results-directory rebaseline lacks one authorized delta")
        self.entries = expected_entries
        self._timestamp_generation += 1
        self.timestamp_token = (
            *_descriptor_token(os.fstat(self.results_fd)),
            self._timestamp_generation,
        )

    def retain_predecessor_residues(self) -> None:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        for row in self.failed_predecessor_attempts:
            path_text = str(row["residue"]["slot1_stage"]["path"])
            leaf = Path(path_text).name
            descriptor = self.predecessor_descriptors.get(path_text)
            if descriptor is None:
                descriptor = os.open(leaf, flags, dir_fd=self.results_fd)
                info = os.fstat(descriptor)
                if (
                    not stat.S_ISDIR(info.st_mode)
                    or stat.S_IMODE(info.st_mode) != 0o700
                    or _directory_entries(descriptor)
                ):
                    os.close(descriptor)
                    _fail(f"{row['identity']} slot-1 residue state differs")
                self.predecessor_descriptors[path_text] = descriptor
                self.predecessor_tokens[path_text] = _descriptor_token(info)
            self._require_predecessor_residue(path_text)

    def _require_predecessor_residue(self, path_text: str) -> None:
        descriptor = self.predecessor_descriptors[path_text]
        token = self.predecessor_tokens[path_text]
        leaf = Path(path_text).name
        info = os.fstat(descriptor)
        edge = os.stat(leaf, dir_fd=self.results_fd, follow_symlinks=False)
        if (
            _descriptor_token(info) != token
            or _descriptor_token(edge) != token
            or _directory_entries(descriptor)
        ):
            _fail(f"retained predecessor residue changed: {path_text}")

    def require_stage(
        self,
        stage: Path,
        expected_entries: tuple[str, ...],
        *,
        authorized_delta: bool = False,
    ) -> None:
        descriptor = self.stage_descriptors.get(stage.name)
        token = self.stage_tokens.get(stage.name)
        if descriptor is None or token is None:
            _fail(f"retained worker-stage descriptor is absent: {stage.name}")
        info = os.fstat(descriptor)
        edge = os.stat(
            stage.name, dir_fd=self.results_fd, follow_symlinks=False
        )
        if (
            _directory_identity(info) != _directory_identity(edge)
            or _directory_identity(info) != token[:4]
            or stat.S_IMODE(info.st_mode) != 0o700
            or _directory_entries(descriptor) != expected_entries
        ):
            _fail(f"retained worker-stage state differs: {stage.name}")
        if authorized_delta:
            os.fsync(descriptor)
            self.stage_entries[stage.name] = expected_entries
            self.stage_tokens[stage.name] = _descriptor_token(
                os.fstat(descriptor)
            )
        elif _descriptor_token(info) != token:
            _fail(f"retained worker-stage timestamps drifted: {stage.name}")

    def probe(self) -> None:
        if self.probed:
            return
        self._require_bound_results()
        baseline = self.entries
        leaf = f".g9cb12-otmpfile-probe-{os.getpid()}-{os.urandom(8).hex()}"
        if leaf in baseline:
            _fail("publication capability probe leaf already exists")
        raw = b"G9CB12 O_TMPFILE capability probe\n"
        descriptor, unnamed_info = _open_unnamed_completed(
            self.results_fd, raw, mode=0o444
        )
        canonical_fd = -1
        try:
            try:
                empty_result = _LIBC.linkat(
                    ctypes.c_int(descriptor),
                    ctypes.c_char_p(b""),
                    ctypes.c_int(self.results_fd),
                    ctypes.c_char_p(os.fsencode(leaf)),
                    ctypes.c_int(0x1000),
                )
            except NameError:
                ctypes.set_errno(errno.ENOENT)
                empty_result = -1
            if empty_result == 0:
                empty_added = tuple(sorted((*baseline, leaf)))
                if _directory_entries(self.results_fd) != empty_added:
                    _fail("O_TMPFILE empty-path addition delta differs")
                os.fsync(self.results_fd)
                self._rebaseline(empty_added)
                os.unlink(leaf, dir_fd=self.results_fd)
                if _directory_entries(self.results_fd) != baseline:
                    _fail("O_TMPFILE empty-path removal delta differs")
                os.fsync(self.results_fd)
                self._rebaseline(baseline)
            elif ctypes.get_errno() != errno.ENOENT:
                _fail("O_TMPFILE empty-path capability probe differs")
            _link_unnamed_procfd(descriptor, self.results_fd, leaf)
            added = tuple(sorted((*baseline, leaf)))
            if _directory_entries(self.results_fd) != added:
                _fail("capability probe addition delta differs")
            canonical_fd = os.open(
                leaf,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=self.results_fd,
            )
            canonical_info = os.fstat(canonical_fd)
            canonical_raw = _read_publication_descriptor(
                canonical_fd, canonical_info.st_size
            )
            if (
                (canonical_info.st_dev, canonical_info.st_ino)
                != (unnamed_info.st_dev, unnamed_info.st_ino)
                or not stat.S_ISREG(canonical_info.st_mode)
                or stat.S_IMODE(canonical_info.st_mode) != 0o444
                or canonical_info.st_size != len(raw)
                or canonical_raw != raw
                or _sha256_bytes(canonical_raw) != _sha256_bytes(raw)
            ):
                _fail("capability probe canonical inode verification failed")
            os.fsync(self.results_fd)
            self._rebaseline(added)
            os.unlink(leaf, dir_fd=self.results_fd)
            os.fsync(self.results_fd)
            self._rebaseline(baseline)
        except NameError as exc:
            raise TerminalG9CB12Failure(
                "publication helper failure"
            ) from exc
        finally:
            if canonical_fd >= 0:
                os.close(canonical_fd)
            os.close(descriptor)
        self.probed = True

    def publish(
        self,
        relative: Path,
        raw: bytes,
        *,
        mode: int = 0o444,
        prelink_recheck: Any | None = None,
    ) -> dict[str, Any]:
        if relative.parent not in {
            Path("results"),
            Path("."),
        } or relative.name in ("", ".", ".."):
            _fail(f"publication path is not an exact results leaf: {relative}")
        self.probe()
        if relative.name in self.entries:
            raise FileExistsError(f"write-once path exists: {relative}")
        descriptor, unnamed_info = _open_unnamed_completed(
            self.results_fd, raw, mode=mode
        )
        canonical_fd = -1
        try:
            if prelink_recheck is not None:
                prelink_recheck()
            self._require_bound_results()
            if _directory_entries(self.results_fd) != self.entries:
                _fail("results-directory drift before canonical link")
            _link_unnamed_procfd(
                descriptor, self.results_fd, relative.name
            )
            expected_entries = tuple(sorted((*self.entries, relative.name)))
            canonical_fd = os.open(
                relative.name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=self.results_fd,
            )
            canonical_info = os.fstat(canonical_fd)
            canonical_raw = _read_publication_descriptor(
                canonical_fd, canonical_info.st_size
            )
            if (
                (canonical_info.st_dev, canonical_info.st_ino)
                != (unnamed_info.st_dev, unnamed_info.st_ino)
                or stat.S_IMODE(canonical_info.st_mode) != mode
                or canonical_info.st_size != len(raw)
                or canonical_raw != raw
                or _sha256_bytes(canonical_raw) != _sha256_bytes(raw)
            ):
                _fail("canonical publication inode verification failed")
            if _directory_entries(self.results_fd) != expected_entries:
                _fail("canonical publication one-leaf delta differs")
            os.fsync(self.results_fd)
            self._rebaseline(expected_entries)
            return {
                "canonical_inode": (
                    canonical_info.st_dev,
                    canonical_info.st_ino,
                ),
                "unnamed_inode": (
                    unnamed_info.st_dev,
                    unnamed_info.st_ino,
                ),
                "sha256": _sha256_bytes(canonical_raw),
                "size_bytes": canonical_info.st_size,
                "mode": stat.S_IMODE(canonical_info.st_mode),
            }
        except NameError as exc:
            raise TerminalG9CB12Failure(
                "publication helper failure"
            ) from exc
        finally:
            if canonical_fd >= 0:
                os.close(canonical_fd)
            os.close(descriptor)

    def close(self) -> None:
        for descriptor in self.predecessor_descriptors.values():
            try:
                os.close(descriptor)
            except OSError:
                pass
        self.predecessor_descriptors.clear()
        for descriptor in self.stage_descriptors.values():
            try:
                os.close(descriptor)
            except OSError:
                pass
        self.stage_descriptors.clear()
        os.close(self.results_fd)
        os.close(self.repository_fd)


def _is_forbidden_g9cb12_helper_name(name: str) -> bool:
    canonical = (
        PREREGISTRATION_PATH,
        CLAIM_PATH,
        SENTINEL_PATH,
        *WORKER_LEDGER_PATHS,
        CSV_PATH,
        MANIFEST_PATH,
    )
    return (
        name.startswith(
            (
                ".g9cb12-otmpfile-probe-",
                ".gross9-structural-clock-g9cb12-worker-",
            )
        )
        or any(name.startswith(f".{path.name}.stage-") for path in canonical)
        or (
            name.startswith(".gross9_structural_clock_bundle_g9cb12_")
            and ".stage-" in name
        )
    )


def _validate_closed_entry_phase(
    context: _PublicationContext, phase: str
) -> None:
    phase_states = {
        Q12_PREREGISTRATION_PUBLICATION: (False, False, False),
        P12_CLAIM_PREFLIGHT: (True, False, False),
        C12_PRODUCTION_PREFLIGHT: (True, True, False),
        D12_COMMITTED_VERIFICATION: (True, True, True),
    }
    if phase not in phase_states:
        _fail("closed entry-point phase is invalid")
    prereg_present, claim_present, publications_present = phase_states[phase]
    context._require_bound_results()
    observed_entries = _directory_entries(context.results_fd)
    if observed_entries != context.entries:
        _fail(f"{phase} results inventory drifted")
    names = set(observed_entries)
    expected_presence = {
        PREREGISTRATION_PATH.name: prereg_present,
        CLAIM_PATH.name: claim_present,
        **{
            path.name: publications_present
            for path in (
                SENTINEL_PATH,
                *WORKER_LEDGER_PATHS,
                CSV_PATH,
                MANIFEST_PATH,
            )
        },
    }
    for leaf, required in expected_presence.items():
        if (leaf in names) != required:
            _fail(f"{phase} active path-state differs: {leaf}")
        if required:
            info = os.stat(
                leaf, dir_fd=context.results_fd, follow_symlinks=False
            )
            if (
                not stat.S_ISREG(info.st_mode)
                or stat.S_IMODE(info.st_mode) != 0o444
            ):
                _fail(f"{phase} active leaf mode/type differs: {leaf}")
    tracked_results: set[str] = set()
    if (context.root / ".git").exists():
        tracked_results = _tracked_results_top_level_entries(
            _git_text(context.root, "ls-files", "--", "results")
        )
    predecessor_residue_names = {
        Path(str(row["residue"]["slot1_stage"]["path"])).name
        for row in context.failed_predecessor_attempts
    }
    active_untracked = {
        leaf for leaf, required in expected_presence.items() if required
    }
    expected_inventory = (
        tracked_results | predecessor_residue_names | active_untracked
    )
    supervisor_present = H12_SUPERVISOR_SENTINEL_PATH.name in names
    if supervisor_present:
        if phase != D12_COMMITTED_VERIFICATION:
            _fail(f"{phase} unexpected H12 supervisor sentinel exists")
        _read_h12_supervisor_sentinel(
            context.root,
            os.environ.get(H12_SUPERVISOR_ENV, ""),
        )
        expected_inventory.add(H12_SUPERVISOR_SENTINEL_PATH.name)
    if names != expected_inventory:
        _fail(f"{phase} exact results inventory differs")
    if _PYCACHE_PREFIX_RELATIVE.name in names:
        _fail(f"{phase} fixed pycache path exists")
    if any(_is_forbidden_g9cb12_helper_name(name) for name in names):
        _fail(f"{phase} forbidden G9CB12 helper path exists")
    context.retain_predecessor_residues()
    _validate_failed_predecessor_permanent_state(
        context.results_fd,
        context.failed_predecessor_attempts,
        context.failed_predecessor_closures,
        context.failed_predecessor_prepublication_closures,
        context.failed_predecessor_pre_sentinel_closures,
        retained_directories=context.predecessor_descriptors,
    )


def _validate_production_checkpoint_two(
    context: _PublicationContext,
    stage_one: Path,
    stage_two: Path,
) -> None:
    context._require_bound_results()
    names = set(_directory_entries(context.results_fd))
    required = {PREREGISTRATION_PATH.name, CLAIM_PATH.name, stage_one.name}
    forbidden = {
        SENTINEL_PATH.name,
        *(path.name for path in WORKER_LEDGER_PATHS),
        CSV_PATH.name,
        MANIFEST_PATH.name,
        stage_two.name,
        _PYCACHE_PREFIX_RELATIVE.name,
    }
    if not required.issubset(names) or names & forbidden:
        _fail("production checkpoint-2 path-state differs")
    info = os.stat(
        stage_one.name,
        dir_fd=context.results_fd,
        follow_symlinks=False,
    )
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700:
        _fail("production checkpoint-2 stage mode/type differs")
    context.require_stage(stage_one, ())


def _validate_production_namespace(
    context: _PublicationContext,
    checkpoint: str,
    stage_one: Path,
    stage_two: Path,
) -> None:
    context._require_bound_results()
    names = set(_directory_entries(context.results_fd))
    if tuple(sorted(names)) != context.entries:
        _fail(f"{checkpoint} results inventory drifted")
    publication_order = (
        SENTINEL_PATH.name,
        WORKER_LEDGER_PATHS[0].name,
        WORKER_LEDGER_PATHS[1].name,
        CSV_PATH.name,
        MANIFEST_PATH.name,
    )
    publication_counts = {
        "C12_PRODUCTION_PREFLIGHT": 0,
        "CAPABILITY_PROBE_COMPLETE": 0,
        "SLOT1_PREPARED": 0,
        "SENTINEL_LINKED": 1,
        "PASS1_LEDGER_LINKED": 2,
        "PASS1_OUTPUT_READY": 2,
        "SLOT_TRANSITION": 2,
        "PASS2_LEDGER_LINKED": 3,
        "PASS2_OUTPUT_READY": 3,
        "CANONICAL_CSV_LINKED": 4,
        "MANIFEST_LINKED_LAST": 5,
        "FINAL_CLEANUP": 5,
    }
    count = publication_counts[checkpoint]
    for index, leaf in enumerate(publication_order):
        if (leaf in names) != (index < count):
            _fail(f"{checkpoint} publication state differs: {leaf}")
    stage_expectation = {
        "C12_PRODUCTION_PREFLIGHT": (),
        "CAPABILITY_PROBE_COMPLETE": (),
        "SLOT1_PREPARED": (stage_one,),
        "SENTINEL_LINKED": (stage_one,),
        "PASS1_LEDGER_LINKED": (stage_one,),
        "PASS1_OUTPUT_READY": (stage_one,),
        "SLOT_TRANSITION": (stage_two,),
        "PASS2_LEDGER_LINKED": (stage_two,),
        "PASS2_OUTPUT_READY": (stage_two,),
        "CANONICAL_CSV_LINKED": (stage_two,),
        "MANIFEST_LINKED_LAST": (stage_two,),
        "FINAL_CLEANUP": (),
    }[checkpoint]
    for stage in (stage_one, stage_two):
        if (stage.name in names) != (stage in stage_expectation):
            _fail(f"{checkpoint} worker-stage state differs: {stage.name}")
    output_ready = checkpoint in {
        "PASS1_OUTPUT_READY",
        "PASS2_OUTPUT_READY",
        "CANONICAL_CSV_LINKED",
        "MANIFEST_LINKED_LAST",
    }
    for stage in stage_expectation:
        expected_entries = (
            tuple(sorted(
                (_STAGED_CSV_NAME, _STAGED_CORE_NAME, _STAGED_RECEIPT_NAME)
            ))
            if output_ready
            else ()
        )
        context.require_stage(
            stage,
            expected_entries,
            authorized_delta=checkpoint
            in {"PASS1_OUTPUT_READY", "PASS2_OUTPUT_READY"},
        )
        stage_fd = context.stage_descriptors[stage.name]
        for leaf in expected_entries:
            leaf_info = os.stat(
                leaf, dir_fd=stage_fd, follow_symlinks=False
            )
            if (
                not stat.S_ISREG(leaf_info.st_mode)
                or stat.S_IMODE(leaf_info.st_mode) != 0o400
            ):
                _fail(f"{checkpoint} worker-stage leaf differs: {leaf}")
    if _PYCACHE_PREFIX_RELATIVE.name in names:
        _fail(f"{checkpoint} fixed pycache path exists")
    if any(_is_forbidden_g9cb12_helper_name(name) for name in names):
        allowed_stages = {stage.name for stage in stage_expectation}
        if any(
            _is_forbidden_g9cb12_helper_name(name)
            and name not in allowed_stages
            for name in names
        ):
            _fail(f"{checkpoint} forbidden G9CB12 helper path exists")
    context.retain_predecessor_residues()
    _validate_failed_predecessor_permanent_state(
        context.results_fd,
        context.failed_predecessor_attempts,
        context.failed_predecessor_closures,
        context.failed_predecessor_prepublication_closures,
        context.failed_predecessor_pre_sentinel_closures,
        retained_directories=context.predecessor_descriptors,
    )


def _validate_production_ledger_checkpoint(
    context: _PublicationContext, ledger: Path
) -> None:
    context._require_bound_results()
    if ledger.name not in _directory_entries(context.results_fd):
        _fail(f"production ledger checkpoint is absent: {ledger}")
    info = os.stat(
        ledger.name,
        dir_fd=context.results_fd,
        follow_symlinks=False,
    )
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o444
    ):
        _fail(f"production ledger checkpoint differs: {ledger}")


def _atomic_link_write_once(
    path: Path,
    raw: bytes,
    *,
    mode: int = 0o444,
    prelink_recheck: Any | None = None,
    publication_context: _PublicationContext | None = None,
) -> dict[str, Any]:
    """Publish complete bytes create-only from a verified unnamed inode."""

    publication_flags = getattr(os, "O_TMPFILE", 0)
    if not publication_flags:
        _fail("O_TMPFILE is unavailable")
    owned = publication_context is None
    if publication_context is None:
        publication_context = _PublicationContext(
            path.parent, directory_relative="."
        )
    try:
        return publication_context.publish(
            (
                Path("results") / path.name
                if publication_context.directory_relative == "results"
                else Path(path.name)
            ),
            raw,
            mode=mode,
            prelink_recheck=prelink_recheck,
        )
    finally:
        if owned:
            publication_context.close()


def _final_parent_snapshot_recheck(
    root: Path,
    snapshot: _SecureBoundSnapshot,
    initial_pairs: Mapping[str, tuple[str, str] | None],
    context: _PublicationContext,
    phase: str,
    expected_branch: str,
    path_state_check: Any | None = None,
) -> None:
    snapshot.verify_final()
    for path_text in sorted(snapshot.repository_relative):
        repository_relative = snapshot.repository_relative[path_text]
        initial = initial_pairs[path_text]
        if repository_relative:
            declaration: Mapping[str, Any] = (
                {"git_blob": initial[0], "git_mode": initial[1]}
                if initial is not None
                else {"git_blob": None, "git_mode": None}
            )
            observed = _validate_git_pair_preflight(
                root,
                path_text,
                repository_relative=True,
                declaration=declaration,
                verify_git=True,
                require_tracked=initial is not None,
                process_runner=_FINAL_GIT_PROCESS,
            )
            if observed != initial:
                _fail(f"final bound Git classification differs: {path_text}")
    if path_state_check is None:
        _validate_closed_entry_phase(context, phase)
    else:
        path_state_check()
    _require_clean_pushed_branch(root, expected_branch)


def validate_claim_preflight(
    root: Path = REPOSITORY_ROOT,
    *,
    _retain_publication_resources: bool = False,
    expected_security_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the complete metadata-only claim preflight without writing a claim."""

    if not (root / ".git").exists():
        validate_preregistration(root)
    head = _require_clean_pushed_branch(root, prereg.EXPECTED_BRANCH)
    head_raw = _git(
        root, "show", f"HEAD:{PREREGISTRATION_PATH.as_posix()}"
    )
    preregistration_head = _decode_canonical_object(
        head_raw,
        PREREGISTRATION_PATH.as_posix(),
        "manifest_hash",
    )
    parent = _require_clean_pushed_branch(
        root,
        _expected_branch(preregistration_head),
    )
    if head != parent:
        _fail("claim preflight HEAD changed during bootstrap")
    _validate_preregistration_seal_head(
        root,
        preregistration_head,
        parent,
        classify_worktree=False,
    )
    security_profile = (
        _validated_expected_security_profile(
            root,
            preregistration_head,
            expected_security_profile,
        )
        if expected_security_profile is not None
        else _official_expected_security_profile(
            root, preregistration_head
        )
    )
    snapshot, pairs = _preauthenticate_parent_snapshot(
        root,
        preregistration_head,
        content_mode=_PREREGISTRATION_ONLY,
        expected_security_profile=security_profile,
    )
    preregistration_pair = pairs.get(PREREGISTRATION_PATH.as_posix())
    if preregistration_pair is None or preregistration_pair[1] != "100644":
        snapshot.close()
        _fail("active preregistration Git mode differs")
    context = _PublicationContext(
        root, expected_security_profile=security_profile
    )
    try:
        preregistration, prereg_binding = validate_preregistration(
            root,
            raw_cache=snapshot,
            expected_security_profile=security_profile,
        )
        if preregistration != preregistration_head:
            _fail("worktree preregistration differs from HEAD bootstrap")
        opaque_inputs = _validate_regular_hashed_inputs(
            root,
            preregistration,
            raw_cache=snapshot,
            preclassified_pairs=dict(pairs),
        )
        environment = _validate_environment(preregistration, root)
        closures = _validate_static_closures(
            root,
            preregistration,
            raw_cache=snapshot,
        )
        bindings = [
            _head_blob_binding(
                root,
                path,
                raw_cache=snapshot,
                preclassified_pairs=pairs,
            )
            for path in _planned_protocol_paths(preregistration)
        ]
        _validate_closed_entry_phase(context, P12_CLAIM_PREFLIGHT)
        context.probe()
        snapshot.rebaseline_directory_timestamps(
            matching_identity=context.results_token
        )
        _validate_closed_entry_phase(context, P12_CLAIM_PREFLIGHT)
        result = {
            "preregistration": preregistration,
            "preregistration_binding": prereg_binding,
            "authority_amendments": _authority_amendment_bindings(
                preregistration,
                expected=security_profile["authority_amendments"],
            ),
            "opaque_inputs": opaque_inputs,
            "environment": environment,
            "closures": closures,
            "protocol_parent_commit": parent,
            "protocol_bindings": bindings,
        }
        if _retain_publication_resources:
            result["_snapshot"] = snapshot
            result["_pairs"] = pairs
            result["_publication_context"] = context
            return result
        _final_parent_snapshot_recheck(
            root,
            snapshot,
            pairs,
            context,
            P12_CLAIM_PREFLIGHT,
            prereg.EXPECTED_BRANCH,
        )
        return result
    finally:
        if not _retain_publication_resources:
            snapshot.close()
            context.close()


def create_claim_only(
    root: Path = REPOSITORY_ROOT,
    *,
    expected_security_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    _validate_bytecode_preflight(root)
    preflight = validate_claim_preflight(
        root,
        _retain_publication_resources=True,
        expected_security_profile=expected_security_profile,
    )
    prereg_binding = preflight["preregistration_binding"]
    authority_amendments = preflight["authority_amendments"]
    parent = preflight["protocol_parent_commit"]
    bindings = preflight["protocol_bindings"]
    opaque_inputs = preflight["opaque_inputs"]
    payload = _claim_payload(
        parent,
        prereg_binding,
        authority_amendments,
        bindings,
        opaque_inputs,
    )
    raw = _canonical_json_bytes(payload)
    snapshot = preflight["_snapshot"]
    pairs = preflight["_pairs"]
    context = preflight["_publication_context"]
    try:
        _atomic_link_write_once(
            _rooted(root, CLAIM_PATH),
            raw,
            prelink_recheck=lambda: _final_parent_snapshot_recheck(
                root,
                snapshot,
                pairs,
                context,
                P12_CLAIM_PREFLIGHT,
                prereg.EXPECTED_BRANCH,
            ),
            publication_context=context,
        )
        binding = {
            "path": CLAIM_PATH.as_posix(),
            "sha256": _sha256_bytes(raw),
            "claim_hash": payload["claim_hash"],
            "protocol_parent_commit": parent,
        }
        return binding
    finally:
        snapshot.close()
        context.close()


def _validate_claim_commit(
    root: Path,
    *,
    raw_cache: dict[str, tuple[bytes, os.stat_result]] | None = None,
    preclassified_pairs: dict[str, tuple[str, str] | None] | None = None,
    _resource_out: dict[str, Any] | None = None,
    expected_security_profile: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    head = _require_clean_pushed_branch(root, prereg.EXPECTED_BRANCH)
    preregistration_head_raw = _git(
        root, "show", f"HEAD:{PREREGISTRATION_PATH.as_posix()}"
    )
    claim_head_raw = _git(root, "show", f"HEAD:{CLAIM_PATH.as_posix()}")
    preregistration_head = _decode_canonical_object(
        preregistration_head_raw,
        PREREGISTRATION_PATH.as_posix(),
        "manifest_hash",
    )
    claim_head = _decode_canonical_object(
        claim_head_raw,
        CLAIM_PATH.as_posix(),
        "claim_hash",
    )
    if _expected_branch(preregistration_head) != prereg.EXPECTED_BRANCH:
        _fail("active preregistration branch binding differs")
    security_profile = (
        _validated_expected_security_profile(
            root, preregistration_head, expected_security_profile
        )
        if expected_security_profile is not None
        else _official_expected_security_profile(root, preregistration_head)
    )
    parent = claim_head.get("protocol_parent_commit")
    parents = _git_text(
        root, "rev-list", "--parents", "-n", "1", "HEAD"
    ).split()
    if len(parents) != 2 or parents != [head, parent]:
        _fail("HEAD is not the direct child of the claimed protocol parent")
    changed = [
        line
        for line in _git_text(
            root,
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "-r",
            str(parent),
            head,
        ).splitlines()
        if line
    ]
    if changed != [f"A\t{CLAIM_PATH.as_posix()}"]:
        _fail("HEAD is not a claim-only direct-child commit")
    if str(parent) != security_profile["preregistration_seal_commit"]:
        _fail("claim parent differs from expected security profile")
    if (
        "claim_commit" in security_profile
        and head != security_profile["claim_commit"]
    ):
        _fail("claim commit differs from expected security profile")
    security_profile["claim_commit"] = head

    snapshot, pairs = _preauthenticate_parent_snapshot(
        root,
        preregistration_head,
        content_mode=_PREREGISTRATION_PLUS_CLAIM,
        expected_security_profile=security_profile,
    )
    cache = {} if raw_cache is None else raw_cache
    cache.update(snapshot)
    classified = (
        {} if preclassified_pairs is None else preclassified_pairs
    )
    classified.update(pairs)

    claim_path = _rooted(root, CLAIM_PATH)
    claim, raw, claim_info = _read_canonical_object(
        claim_path,
        "claim_hash",
        path_text=CLAIM_PATH.as_posix(),
        raw_cache=cache,
    )
    if stat.S_IMODE(claim_info.st_mode) != 0o444:
        _fail("active claim filesystem mode differs")
    if claim.get("identity") != IDENTITY or claim.get("one_shot") is not True:
        _fail("claim identity or one-shot status mismatch")
    if claim.get("retry_allowed") is not False or claim.get("resume_allowed") is not False:
        _fail("claim permits retry or resume")
    preregistration, prereg_binding = validate_preregistration(
        root,
        raw_cache=cache,
        expected_security_profile=security_profile,
    )
    if preregistration != preregistration_head or claim != claim_head:
        _fail("active metadata differs from the classified HEAD snapshot")
    if claim.get("preregistration") != prereg_binding:
        _fail("claim preregistration binding mismatch")
    authority_amendments = _authority_amendment_bindings(
        preregistration,
        expected=security_profile["authority_amendments"],
    )
    if claim.get("authority_amendments") != authority_amendments:
        _fail("claim authority amendment bindings mismatch")
    protocol_files = claim.get("protocol_files")
    opaque_inputs = claim.get("opaque_inputs_authenticated")
    if not isinstance(protocol_files, list) or not isinstance(opaque_inputs, list):
        _fail("claim bindings are incomplete")
    if [
        row.get("path") for row in protocol_files if isinstance(row, Mapping)
    ] != _planned_protocol_paths(preregistration):
        _fail("claim protocol binding order differs from preregistration")
    parent = claim.get("protocol_parent_commit")
    if claim != _claim_payload(
        str(parent),
        prereg_binding,
        authority_amendments,
        protocol_files,
        opaque_inputs,
    ):
        _fail("claim schema or frozen contract differs")
    for row in protocol_files:
        if not isinstance(row, Mapping) or _head_blob_binding(
            root,
            str(row.get("path")),
            raw_cache=cache,
            preclassified_pairs=classified,
        ) != row:
            _fail("protocol file binding differs from the claim")
    for relative in (
        SENTINEL_PATH,
        *WORKER_LEDGER_PATHS,
        CSV_PATH,
        MANIFEST_PATH,
    ):
        candidate = _rooted(root, relative)
        if candidate.exists() or candidate.is_symlink():
            raise FileExistsError(f"production output already exists: {relative}")
    result = (
        claim,
        {
            "path": CLAIM_PATH.as_posix(),
            "sha256": _sha256_bytes(raw),
            "claim_hash": claim["claim_hash"],
            "protocol_parent_commit": parent,
            "claim_commit": head,
        },
        preregistration,
        prereg_binding,
    )
    if _resource_out is not None:
        _resource_out["snapshot"] = snapshot
        _resource_out["pairs"] = pairs
        _resource_out["security_profile"] = security_profile or {}
    else:
        try:
            snapshot.verify_final()
        finally:
            snapshot.close()
    return result


def _normalise_distribution_name(name: str) -> str:
    return _DIST_NORMALIZE_RE.sub("-", name).lower()


def _environment_record() -> dict[str, Any]:
    distributions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            normalized = _normalise_distribution_name(name)
            if normalized in distributions:
                _fail(f"duplicate normalized distribution: {normalized}")
            distributions[normalized] = distribution.version
    inventory_raw = _canonical_json_bytes(distributions, trailing_lf=False)
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "libc": " ".join(part for part in platform.libc_ver() if part),
        "zlib_compile": zlib.ZLIB_VERSION,
        "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
        "selected_distributions": {
            name: distributions.get(name, "absent")
            for name in (
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "torch",
                "transformers",
                "peft",
                "datasets",
                "trl",
                "websockets",
                "sqlalchemy",
            )
        },
        "distribution_count": len(distributions),
        "distribution_inventory_sha256": _sha256_bytes(inventory_raw),
        "distribution_inventory": dict(sorted(distributions.items())),
    }


def _find_mapping(payload: Any, required_keys: set[str]) -> Mapping[str, Any] | None:
    if isinstance(payload, Mapping):
        if required_keys.issubset(payload):
            return payload
        for value in payload.values():
            found = _find_mapping(value, required_keys)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_mapping(value, required_keys)
            if found is not None:
                return found
    return None


def _module_file_candidates(
    module: str,
    root: Path,
    available_paths: set[str] | None = None,
) -> list[Path]:
    if not module:
        return []
    parts = module.split(".")
    found: list[Path] = []
    for index in range(1, len(parts)):
        initializer = Path(*parts[:index]) / "__init__.py"
        if (
            initializer.as_posix() in available_paths
            if available_paths is not None
            else (root / initializer).is_file()
        ):
            found.append(initializer)
    module_file = Path(*parts).with_suffix(".py")
    package_file = Path(*parts) / "__init__.py"
    if (
        module_file.as_posix() in available_paths
        if available_paths is not None
        else (root / module_file).is_file()
    ):
        found.append(module_file)
    elif (
        package_file.as_posix() in available_paths
        if available_paths is not None
        else (root / package_file).is_file()
    ):
        found.append(package_file)
    return found


def _path_module_name(path: Path) -> tuple[str, bool]:
    initializer = path.name == "__init__.py"
    parts = list(path.with_suffix("").parts)
    if initializer:
        parts.pop()
    return ".".join(parts), initializer


def _local_import_paths(
    node: ast.Import | ast.ImportFrom,
    current_path: Path,
    root: Path,
    available_paths: set[str] | None = None,
) -> set[Path]:
    modules: set[str] = set()
    current_module, initializer = _path_module_name(current_path)
    package = current_module if initializer else current_module.rpartition(".")[0]
    if isinstance(node, ast.Import):
        modules.update(alias.name for alias in node.names)
    else:
        if node.level:
            package_parts = package.split(".") if package else []
            remove = node.level - 1
            if remove > len(package_parts):
                return set()
            prefix = package_parts[: len(package_parts) - remove]
            if node.module:
                prefix.extend(node.module.split("."))
            base = ".".join(prefix)
        else:
            base = node.module or ""
        if base:
            modules.add(base)
        for alias in node.names:
            if alias.name != "*":
                modules.add(f"{base}.{alias.name}" if base else alias.name)
    paths: set[Path] = set()
    for module in modules:
        paths.update(
            _module_file_candidates(module, root, available_paths)
        )
    return paths


def _discover_import_closure(
    root: Path,
    entry_paths: Sequence[str],
    *,
    raw_cache: Mapping[str, tuple[bytes, os.stat_result]] | None = None,
) -> list[Path]:
    pending = {Path(path) for path in entry_paths}
    discovered: set[Path] = set()
    available_paths = set(raw_cache) if raw_cache is not None else None
    while pending:
        current = min(pending, key=lambda path: path.as_posix())
        pending.remove(current)
        if current in discovered:
            continue
        source_path = root / current
        cached = (
            raw_cache.get(current.as_posix())
            if raw_cache is not None
            else None
        )
        if cached is not None:
            if not stat.S_ISREG(cached[1].st_mode):
                _fail(f"import closure member is not regular: {current}")
        elif isinstance(raw_cache, _SecureBoundSnapshot):
            _fail(f"import closure cache member is absent: {current}")
        elif source_path.is_symlink() or not source_path.is_file():
            _fail(f"import root or closure member is absent: {current}")
        try:
            tree = ast.parse(
                (
                    cached[0].decode("utf-8")
                    if cached is not None
                    else source_path.read_text(encoding="utf-8")
                ),
                filename=current.as_posix(),
            )
        except (UnicodeDecodeError, SyntaxError) as exc:
            raise TerminalG9CB12Failure(
                f"import closure source cannot be parsed: {current}"
            ) from exc
        discovered.add(current)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                pending.update(
                    _local_import_paths(
                        node,
                        current,
                        root,
                        available_paths,
                    )
                    - discovered
                )
    return sorted(discovered, key=lambda path: path.as_posix())


def _validate_environment(
    preregistration: Mapping[str, Any],
    root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    observed = _environment_record()
    bindings = preregistration.get("bindings")
    expected = bindings.get("environment") if isinstance(bindings, Mapping) else None
    if not isinstance(expected, Mapping):
        _fail("preregistration has no complete frozen environment record")
    for key in (
        "python_implementation",
        "python_version",
        "platform",
        "machine",
        "libc",
        "zlib_compile",
        "zlib_runtime",
        "selected_distributions",
        "distribution_count",
        "distribution_inventory_sha256",
        "distribution_inventory",
    ):
        value = observed[key]
        if expected.get(key) != value:
            _fail(f"frozen environment mismatch: {key}")
    expected_worker_environment = prereg.worker_process_environment(root)
    if expected.get("worker_process_environment") != expected_worker_environment:
        _fail("frozen worker process environment mismatch")
    return dict(expected)


def _git_process(
    root: Path, *arguments: str
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


_FINAL_GIT_PROCESS = _git_process


def _decode_git_stdout(
    completed: subprocess.CompletedProcess[bytes], operation: str
) -> str:
    try:
        return completed.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TerminalG9CB12Failure(
            f"{operation}: Git output is not UTF-8"
        ) from exc


def _parse_stage_zero_binding(output: str, path_text: str) -> tuple[str, str]:
    lines = output.splitlines()
    if len(lines) != 1:
        _fail(f"expected exactly one bound input index entry: {path_text}")
    metadata, separator, observed_path = lines[0].partition("\t")
    fields = metadata.split()
    if (
        not separator
        or observed_path != path_text
        or len(fields) != 3
        or fields[2] != "0"
    ):
        _fail(f"bound input index entry is not exact stage zero: {path_text}")
    mode, blob, _stage = fields
    return blob, mode


def _parse_head_tree_binding(output: str, path_text: str) -> tuple[str, str]:
    lines = output.splitlines()
    if len(lines) != 1:
        _fail(f"expected exactly one bound input HEAD entry: {path_text}")
    metadata, separator, observed_path = lines[0].partition("\t")
    fields = metadata.split()
    if (
        not separator
        or observed_path != path_text
        or len(fields) != 3
        or fields[1] != "blob"
    ):
        _fail(f"bound input HEAD entry is not an exact blob: {path_text}")
    mode, _kind, blob = fields
    return blob, mode


def _validate_git_pair_shape(
    binding: Mapping[str, Any], path_text: str
) -> None:
    has_blob = "git_blob" in binding
    has_mode = "git_mode" in binding
    if has_blob != has_mode:
        _fail(f"partial bound input Git metadata pair: {path_text}")
    if not has_blob:
        return
    blob = binding["git_blob"]
    mode = binding["git_mode"]
    if blob is None and mode is None:
        return
    if (
        type(blob) is not str
        or type(mode) is not str
        or not re.fullmatch(r"[0-9a-f]{40}", blob)
        or mode != "100644"
    ):
        _fail(f"malformed bound input Git metadata pair: {path_text}")


def _validate_git_pair_preflight(
    root: Path,
    path_text: str,
    *,
    repository_relative: bool,
    declaration: Mapping[str, Any],
    verify_git: bool,
    require_tracked: bool = False,
    process_runner: Any | None = None,
) -> tuple[str, str] | None:
    if "git_blob" not in declaration and not require_tracked:
        return None
    blob = declaration.get("git_blob")
    mode = declaration.get("git_mode")
    if not repository_relative:
        if blob is not None or mode is not None:
            _fail(f"absolute bound input declares Git metadata: {path_text}")
        return None
    if not verify_git:
        if isinstance(blob, str) and isinstance(mode, str):
            return blob, mode
        return None

    runner = _git_process if process_runner is None else process_runner
    staged = runner(root, "ls-files", "--stage", "--", path_text)
    tree = runner(root, "ls-tree", "HEAD", "--", path_text)
    matched = runner(
        root, "ls-files", "--error-unmatch", "--", path_text
    )
    staged_output = _decode_git_stdout(staged, "git ls-files --stage")
    tree_output = _decode_git_stdout(tree, "git ls-tree")
    matched_output = _decode_git_stdout(
        matched, "git ls-files --error-unmatch"
    )
    if blob is None and mode is None and not require_tracked:
        if (
            staged.returncode != 0
            or tree.returncode != 0
            or staged_output
            or tree_output
            or matched.returncode != 1
            or matched_output
        ):
            _fail(f"paired-null bound input Git absence proof differs: {path_text}")
        return None
    if (
        staged.returncode != 0
        or tree.returncode != 0
        or matched.returncode != 0
        or matched_output.rstrip("\n") != path_text
    ):
        _fail(f"tracked bound input Git classification differs: {path_text}")
    index_blob, index_mode = _parse_stage_zero_binding(
        staged_output, path_text
    )
    tree_blob, tree_mode = _parse_head_tree_binding(tree_output, path_text)
    if require_tracked and blob is None and mode is None:
        blob, mode = tree_blob, tree_mode
    if type(blob) is not str or type(mode) is not str:
        _fail(f"malformed bound input Git metadata pair: {path_text}")
    tracked_blob = blob
    tracked_mode = mode
    if (
        index_blob != tracked_blob
        or tree_blob != tracked_blob
        or index_mode != tracked_mode
        or tree_mode != tracked_mode
    ):
        _fail(f"bound input index/HEAD metadata mismatch: {path_text}")
    return tracked_blob, tracked_mode


def _git_blob_id(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw).hexdigest()


def _descriptor_token(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        stat.S_IFMT(info.st_mode),
        stat.S_IMODE(info.st_mode),
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _directory_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        stat.S_IFMT(info.st_mode),
        stat.S_IMODE(info.st_mode),
    )


def _pread_complete(
    descriptor: int,
    size: int,
    use_stream_read: bool = False,
) -> bytes:
    if use_stream_read:
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) != size or os.read(descriptor, 1) != b"":
            _fail("publication same-descriptor read was incomplete")
        return raw
    if size == 0:
        if os.pread(descriptor, 1, 0) != b"":
            _fail("same-descriptor zero-length read was not at EOF")
        return b""
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not chunk:
            break
        chunks.append(chunk)
        offset += len(chunk)
    raw = b"".join(chunks)
    if len(raw) != size:
        _fail("same-descriptor content read was incomplete")
    return raw


class _SecureBoundSnapshot(dict[str, tuple[bytes, os.stat_result]]):
    """Initial-byte cache plus the retained component-wise descriptor graph."""

    def __init__(
        self,
        root: Path,
        *,
        absolute_allowlist: frozenset[str] = _ABSOLUTE_BINDING_ALLOWLIST,
        repository_fd: int | None = None,
        filesystem_root_fd: int | None = None,
        opener: Any = os.open,
        register_descriptor: Any | None = None,
        instrument_reads: bool = True,
    ) -> None:
        super().__init__()
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        self._borrowed_anchors = repository_fd is not None
        if (repository_fd is None) != (filesystem_root_fd is None):
            _fail("snapshot anchor ownership is inconsistent")
        if repository_fd is None:
            try:
                if opener is os.open:
                    repository_fd = os.open(root, directory_flags)
                    filesystem_root_fd = os.open("/", directory_flags)
                else:
                    repository_fd = opener(root, directory_flags)
                    filesystem_root_fd = opener("/", directory_flags)
            except OSError as exc:
                raise TerminalG9CB12Failure(
                    "snapshot anchors cannot be securely opened"
                ) from exc
        if repository_fd is None or filesystem_root_fd is None:
            _fail("snapshot anchors are absent after secure open")
        repository_anchor_fd: int = repository_fd
        filesystem_root_anchor_fd: int = filesystem_root_fd
        self._opener = opener
        self._instrument_reads = instrument_reads
        self._register_descriptor = register_descriptor
        self._borrowed_descriptors = (
            {repository_anchor_fd, filesystem_root_anchor_fd}
            if self._borrowed_anchors
            else set()
        )
        self.root = root
        self.absolute_allowlist = absolute_allowlist
        self.file_descriptors: dict[str, int] = {}
        self.file_tokens: dict[str, tuple[int, ...]] = {}
        self.file_edges: dict[str, tuple[int, str]] = {}
        self.directory_descriptors: dict[tuple[str, tuple[str, ...]], int] = {
            ("repo", ()): repository_anchor_fd,
            ("absolute", ()): filesystem_root_anchor_fd,
        }
        self.directory_tokens = {
            key: _descriptor_token(os.fstat(descriptor))
            for key, descriptor in self.directory_descriptors.items()
        }
        self.directory_edges: dict[
            tuple[str, tuple[str, ...]], tuple[int, str]
        ] = {}
        self.directory_entries: dict[
            tuple[str, tuple[str, ...]], tuple[str, ...]
        ] = {}
        for key, token in self.directory_tokens.items():
            if token[2] != stat.S_IFDIR:
                self.close()
                _fail(f"snapshot anchor is not a directory: {key}")
        self.final_verified = False
        self.closed = False
        self.repository_relative: dict[str, bool] = {}
        self.declarations: dict[str, dict[str, Any]] = {}

    def _parent_fd(
        self, path_text: str, repository_relative: bool
    ) -> tuple[int, str]:
        components = (
            path_text.split("/")
            if repository_relative
            else path_text[1:].split("/")
        )
        if not repository_relative and path_text not in self.absolute_allowlist:
            self.close()
            _fail(f"absolute bound input is outside the allowlist: {path_text}")
        namespace = "repo" if repository_relative else "absolute"
        prefix: tuple[str, ...] = ()
        parent_fd = self.directory_descriptors[(namespace, prefix)]
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        for component in components[:-1]:
            prefix += (component,)
            key = (namespace, prefix)
            descriptor = self.directory_descriptors.get(key)
            if descriptor is None:
                try:
                    descriptor = self._opener(
                        component, flags, dir_fd=parent_fd
                    )
                except OSError as exc:
                    self.close()
                    raise TerminalG9CB12Failure(
                        "bound input parent component cannot be opened "
                        f"no-follow (possible symlink): {path_text}"
                    ) from exc
                info = os.fstat(descriptor)
                if not stat.S_ISDIR(info.st_mode):
                    os.close(descriptor)
                    self.close()
                    _fail(f"bound input parent is not a directory: {path_text}")
                self.directory_descriptors[key] = descriptor
                self.directory_tokens[key] = _descriptor_token(info)
                self.directory_edges[key] = (parent_fd, component)
                if self._register_descriptor is not None:
                    self._register_descriptor(
                        descriptor,
                        f"{namespace}:{'/'.join(prefix)}",
                        info,
                        True,
                    )
            parent_fd = descriptor
        return parent_fd, components[-1]

    def retain_empty_directory(self, path_text: str, *, mode: int) -> None:
        components = tuple(path_text.split("/"))
        if not components or any(part in ("", ".", "..") for part in components):
            _fail("retained directory path is not normalized")
        key = ("repo", components)
        if key in self.directory_descriptors:
            if self.directory_entries.get(key) != ():
                _fail("retained directory declaration differs")
            return
        parent_fd, leaf = self._parent_fd(path_text, True)
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            descriptor = self._opener(leaf, flags, dir_fd=parent_fd)
        except OSError as exc:
            raise TerminalG9CB12Failure(
                f"retained directory cannot be opened no-follow: {path_text}"
            ) from exc
        try:
            info = os.fstat(descriptor)
            if self._register_descriptor is not None:
                self._register_descriptor(descriptor, path_text, info, True)
            entries = _directory_entries(descriptor)
            if (
                not stat.S_ISDIR(info.st_mode)
                or stat.S_IMODE(info.st_mode) != mode
                or entries
            ):
                _fail(f"retained directory state differs: {path_text}")
            self.directory_descriptors[key] = descriptor
            self.directory_tokens[key] = _descriptor_token(info)
            self.directory_edges[key] = (parent_fd, leaf)
            self.directory_entries[key] = entries
        except BaseException:
            os.close(descriptor)
            raise

    def open_initial(
        self, path_text: str, repository_relative: bool
    ) -> tuple[bytes, os.stat_result]:
        if path_text in self:
            return self[path_text]
        parent_fd, leaf = self._parent_fd(path_text, repository_relative)
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            descriptor = self._opener(leaf, flags, dir_fd=parent_fd)
        except OSError as exc:
            self.close()
            raise TerminalG9CB12Failure(
                f"bound input cannot be opened component-wise no-follow: {path_text}"
            ) from exc
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            os.close(descriptor)
            self.close()
            _fail(f"bound input is not a regular file: {path_text}")
        raw, read_info = (
            _read_bound_regular_bytes(descriptor, path_text)
            if self._instrument_reads
            else _read_open_descriptor(descriptor, path_text)
        )
        after = os.fstat(descriptor)
        if (
            _descriptor_token(before) != _descriptor_token(read_info)
            or _descriptor_token(before) != _descriptor_token(after)
        ):
            os.close(descriptor)
            self.close()
            _fail(f"bound input changed during initial authentication: {path_text}")
        object_key = (after.st_dev, after.st_ino)
        if any(
            (token[0], token[1]) == object_key
            for token in self.file_tokens.values()
        ):
            os.close(descriptor)
            self.close()
            _fail("two bound paths alias the same filesystem object")
        self.file_descriptors[path_text] = descriptor
        self.file_tokens[path_text] = _descriptor_token(after)
        self.file_edges[path_text] = (parent_fd, leaf)
        if self._register_descriptor is not None:
            self._register_descriptor(
                descriptor, path_text, after, False
            )
        self[path_text] = (raw, after)
        return raw, after

    def verify_final(self) -> None:
        if self.final_verified:
            _fail("bound snapshot final verification was requested twice")
        for path_text in sorted(self.file_descriptors):
            descriptor = self.file_descriptors[path_text]
            before = os.fstat(descriptor)
            raw = _pread_complete(descriptor, before.st_size)
            after = os.fstat(descriptor)
            if (
                _descriptor_token(before) != self.file_tokens[path_text]
                or _descriptor_token(after) != self.file_tokens[path_text]
                or raw != self[path_text][0]
            ):
                _fail(f"bound input final same-FD verification differs: {path_text}")
            parent_fd, leaf = self.file_edges[path_text]
            try:
                path_info = os.stat(
                    leaf, dir_fd=parent_fd, follow_symlinks=False
                )
            except OSError as exc:
                raise TerminalG9CB12Failure(
                    f"bound input leaf path changed: {path_text}"
                ) from exc
            if _descriptor_token(path_info) != self.file_tokens[path_text]:
                _fail(f"bound input leaf component changed: {path_text}")
        for key, descriptor in self.directory_descriptors.items():
            observed = _descriptor_token(os.fstat(descriptor))
            if observed != self.directory_tokens[key]:
                _fail(f"bound input directory graph drifted: {key}")
            expected_entries = self.directory_entries.get(key)
            if (
                expected_entries is not None
                and _directory_entries(descriptor) != expected_entries
            ):
                _fail(f"retained directory inventory drifted: {key}")
            edge = self.directory_edges.get(key)
            if edge is not None:
                parent_fd, component = edge
                try:
                    path_info = os.stat(
                        component,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                except OSError as exc:
                    raise TerminalG9CB12Failure(
                        f"bound input parent path changed: {key}"
                    ) from exc
                if _descriptor_token(path_info) != self.directory_tokens[key]:
                    _fail(f"bound input parent component changed: {key}")
        self.final_verified = True

    def rebaseline_directory_timestamps(
        self, *, matching_identity: tuple[int, ...]
    ) -> None:
        for key, descriptor in self.directory_descriptors.items():
            info = os.fstat(descriptor)
            if _directory_identity(info) == matching_identity:
                self.directory_tokens[key] = _descriptor_token(info)

    def close(self) -> None:
        if getattr(self, "closed", False):
            return
        for descriptor in list(getattr(self, "file_descriptors", {}).values()):
            try:
                os.close(descriptor)
            except OSError:
                pass
        for descriptor in list(
            getattr(self, "directory_descriptors", {}).values()
        ):
            if descriptor in self._borrowed_descriptors:
                continue
            try:
                os.close(descriptor)
            except OSError:
                pass
        self.closed = True


def _retain_expected_predecessor_residue_directories(
    snapshot: _SecureBoundSnapshot,
    preregistration: Mapping[str, Any],
    expected_attempts: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    if (
        preregistration.get("identity") != IDENTITY
        or preregistration.get("protocol_version") != prereg.PROTOCOL_VERSION
    ):
        return
    attempts = expected_attempts
    if attempts is None:
        bindings = preregistration.get("bindings")
        observed = (
            bindings.get("failed_predecessor_attempts")
            if isinstance(bindings, Mapping)
            else None
        )
        attempts = (
            observed
            if isinstance(observed, list)
            else prereg.expected_failed_predecessor_attempts()
        )
    for row in attempts:
        snapshot.retain_empty_directory(
            str(row["residue"]["slot1_stage"]["path"]),
            mode=0o700,
        )


_ACTIVE_WORKER_GUARD: Any | None = None


def _read_open_descriptor(
    descriptor: int, path_text: str
) -> tuple[bytes, os.stat_result]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        _fail(f"bound input is not a regular file: {path_text}")
    raw = _pread_complete(descriptor, before.st_size)
    after = os.fstat(descriptor)
    if _descriptor_token(before) != _descriptor_token(after):
        _fail(f"bound input changed during authentication: {path_text}")
    return raw, after


def _read_bound_regular_bytes(
    path: Path | int, path_text: str
) -> tuple[bytes, os.stat_result]:
    """Compatibility reader using the same component-wise graph discipline."""

    if isinstance(path, int):
        return _read_open_descriptor(path, path_text)

    repository_relative = not path_text.startswith("/")
    guard = _ACTIVE_WORKER_GUARD
    if guard is not None and (
        not repository_relative
        or guard.root
        == path.parents[len(PurePosixPath(path_text).parts) - 1]
    ):
        components = (
            path_text.split("/")
            if repository_relative
            else path_text[1:].split("/")
        )
        if (
            not repository_relative
            and path_text not in _ABSOLUTE_BINDING_ALLOWLIST
        ):
            _fail(f"absolute bound input is outside the allowlist: {path_text}")
        parent_fd = (
            guard.repository_fd
            if repository_relative
            else guard.filesystem_root_fd
        )
        if parent_fd is None:
            _fail("worker bound-input anchor is absent")
        opened_directories: list[int] = []
        leaf_fd = -1
        try:
            directory_flags = (
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
            )
            for component in components[:-1]:
                parent_fd = guard._original_os_open(
                    component,
                    directory_flags,
                    dir_fd=parent_fd,
                )
                opened_directories.append(parent_fd)
            leaf_fd = guard._original_os_open(
                components[-1],
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_fd,
            )
            before = os.fstat(leaf_fd)
            if not stat.S_ISREG(before.st_mode):
                _fail(f"bound input is not a regular file: {path_text}")
            raw = _pread_complete(leaf_fd, before.st_size)
            after = os.fstat(leaf_fd)
            if _descriptor_token(before) != _descriptor_token(after):
                _fail(f"bound input changed during authentication: {path_text}")
            return raw, after
        finally:
            if leaf_fd >= 0:
                guard._originals.get((os, "close"), os.close)(leaf_fd)
            for descriptor in reversed(opened_directories):
                guard._originals.get((os, "close"), os.close)(descriptor)
    root = REPOSITORY_ROOT
    if repository_relative:
        root = path
        for _component in PurePosixPath(path_text).parts:
            root = root.parent
    snapshot = _SecureBoundSnapshot(
        root,
        absolute_allowlist=(
            _ABSOLUTE_BINDING_ALLOWLIST
            if repository_relative
            else frozenset({path_text})
        ),
        instrument_reads=False,
    )
    try:
        raw, info = snapshot.open_initial(path_text, repository_relative)
        snapshot.verify_final()
        return raw, info
    finally:
        snapshot.close()


def _preauthenticate_parent_snapshot(
    root: Path,
    preregistration: Mapping[str, Any],
    *,
    content_mode: str = _PREREGISTRATION_PLUS_CLAIM,
    extra_tracked_paths: Sequence[Path] = (),
    expected_security_profile: Mapping[str, Any] | None = None,
) -> tuple[
    _SecureBoundSnapshot,
    dict[str, tuple[str, str] | None],
]:
    """Classify all Git pairs, then open/cache each bound leaf exactly once."""

    if content_mode not in {
        _PREREGISTRATION_ONLY,
        _PREREGISTRATION_PLUS_CLAIM,
    }:
        _fail("parent snapshot content mode is invalid")

    prepared: dict[str, tuple[Path, bool]] = {}
    declarations: dict[str, dict[str, Any]] = {}
    required_tracked_paths: set[str] = set()
    for binding in _iter_bindings(preregistration):
        path_text = _binding_path(binding)
        candidate, repository_relative = _bound_regular_path(root, path_text)
        _validate_git_pair_shape(binding, path_text)
        prior = prepared.get(path_text)
        if prior is not None and prior != (candidate, repository_relative):
            _fail(f"conflicting duplicate input binding: {path_text}")
        prepared[path_text] = (candidate, repository_relative)
        declared = declarations.setdefault(path_text, {})
        for key in ("git_blob", "git_mode"):
            if key in binding:
                value = binding[key]
                if key in declared and declared[key] != value:
                    _fail(
                        f"conflicting duplicate input metadata: "
                        f"{path_text}:{key}"
                    )
                declared[key] = value
    metadata_paths = [PREREGISTRATION_PATH]
    if content_mode == _PREREGISTRATION_PLUS_CLAIM:
        metadata_paths.append(CLAIM_PATH)
    for relative in metadata_paths:
        path_text = relative.as_posix()
        prepared[path_text] = (_rooted(root, relative), True)
        declarations.setdefault(path_text, {})
        required_tracked_paths.add(path_text)
    for relative in extra_tracked_paths:
        path_text = relative.as_posix()
        prepared[path_text] = (_rooted(root, relative), True)
        declarations.setdefault(path_text, {})
        required_tracked_paths.add(path_text)
    for path_text in _planned_protocol_paths(preregistration):
        prepared.setdefault(path_text, (_rooted(root, Path(path_text)), True))
        declarations.setdefault(path_text, {})
        required_tracked_paths.add(path_text)

    tracked_pairs: dict[str, tuple[str, str] | None] = {}
    for path_text in sorted(prepared):
        _candidate, repository_relative = prepared[path_text]
        tracked_pairs[path_text] = _validate_git_pair_preflight(
            root,
            path_text,
            repository_relative=repository_relative,
            declaration=declarations[path_text],
            verify_git=True,
            require_tracked=path_text in required_tracked_paths,
        )

    snapshot = _SecureBoundSnapshot(root)
    snapshot.repository_relative = {
        path_text: repository_relative
        for path_text, (_candidate, repository_relative) in prepared.items()
    }
    snapshot.declarations = {
        path_text: dict(declaration)
        for path_text, declaration in declarations.items()
    }
    try:
        _retain_expected_predecessor_residue_directories(
            snapshot,
            preregistration,
            (
                None
                if expected_security_profile is None
                else expected_security_profile[
                    "failed_predecessor_attempts"
                ]
            ),
        )
        for path_text in sorted(prepared):
            candidate, repository_relative = prepared[path_text]
            raw, info = snapshot.open_initial(path_text, repository_relative)
            pair = tracked_pairs[path_text]
            if pair is not None:
                if _git_blob_id(raw) != pair[0]:
                    _fail(f"bound input worktree Git blob mismatch: {path_text}")
                observed_mode = "100755" if info.st_mode & 0o111 else "100644"
                if observed_mode != pair[1]:
                    _fail(f"bound input worktree Git mode mismatch: {path_text}")
        return snapshot, tracked_pairs
    except BaseException:
        snapshot.close()
        raise


def _canonical_bound_json(
    raw: bytes,
    path_text: str,
    hash_field: str,
) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TerminalG9CB12Failure(
            f"historical metadata JSON is invalid: {path_text}"
        ) from exc
    if (
        not isinstance(payload, dict)
        or raw != _canonical_json_bytes(payload)
    ):
        _fail(f"historical metadata JSON is not canonical: {path_text}")
    value = payload.get(hash_field)
    if (
        not isinstance(value, str)
        or not _SHA_RE.fullmatch(value)
        or value != _object_hash(payload, hash_field)
    ):
        _fail(f"historical metadata internal hash differs: {path_text}")
    return payload


def _validate_historical_metadata_bytes(
    path_text: str,
    raw: bytes,
    info: os.stat_result,
) -> None:
    g9cb1 = {
        row["path"]: row
        for row in prereg.expected_failed_predecessor_preregistration_bindings()
    }
    attempts = prereg.expected_failed_predecessor_attempts()
    closure = prereg.expected_failed_predecessor_closures()[0]
    closure_bindings = {
        closure["authority_decision"]["path"]: (
            "authority_decision",
            closure["authority_decision"],
        ),
        closure["preregistration"]["path"]: (
            "preregistration",
            closure["preregistration"],
        ),
    }
    if path_text in closure_bindings:
        key, binding = closure_bindings[path_text]
        if (
            "filesystem_mode_octal" in binding
            and stat.S_IMODE(info.st_mode)
            != int(str(binding["filesystem_mode_octal"]), 8)
        ):
            _fail(f"G9CB-4 closure filesystem mode differs: {path_text}")
        if key == "preregistration":
            payload = _canonical_bound_json(
                raw, path_text, "manifest_hash"
            )
            if (
                payload.get("identity") != "G9CB-4"
                or payload.get("protocol_version")
                != "gross9_structural_clock_bundle_g9cb4_preregistration_v1"
                or payload.get("manifest_hash") != binding["manifest_hash"]
            ):
                _fail(f"G9CB-4 closure metadata differs: {path_text}")
        return
    attempt = attempts[0]
    g9cb2 = {
        attempt[key]["path"]: (key, attempt[key])
        for key in (
            "authority_decision",
            "preregistration",
            "access_claim",
            "attempt_sentinel",
        )
    }
    if path_text in g9cb1:
        binding = g9cb1[path_text]
        if stat.S_IMODE(info.st_mode) != 0o444:
            _fail(f"G9CB-1 historical filesystem mode differs: {path_text}")
        payload = _canonical_bound_json(raw, path_text, "manifest_hash")
        expected_amendments = (
            _expected_authority_amendment_bindings()[:2]
            if binding["protocol_version"]
            == prereg.HISTORICAL_PROTOCOL_VERSION
            else _expected_authority_amendment_bindings()
        )
        if (
            payload.get("identity") != "G9CB-1"
            or payload.get("protocol_version")
            != binding["protocol_version"]
            or payload.get("manifest_hash") != binding["manifest_hash"]
            or payload.get("bindings", {}).get("authority_amendments")
            != expected_amendments
        ):
            _fail(f"G9CB-1 historical metadata fields differ: {path_text}")
        if (
            "protocol_implementation_commit" in binding
            and payload.get("protocol_implementation_commit")
            != binding["protocol_implementation_commit"]
        ):
            _fail(
                "G9CB-1 historical implementation binding differs: "
                f"{path_text}"
            )
        return
    if path_text in g9cb2:
        key, binding = g9cb2[path_text]
        if "filesystem_mode_octal" in binding and stat.S_IMODE(
            info.st_mode
        ) != int(str(binding["filesystem_mode_octal"]), 8):
            _fail(f"G9CB-2 historical filesystem mode differs: {path_text}")
        if key == "authority_decision":
            return
        hash_field = "claim_hash" if key == "access_claim" else "manifest_hash"
        payload = _canonical_bound_json(raw, path_text, hash_field)
        expected_protocol = binding.get(
            "protocol_version",
            attempt["protocol_version"],
        )
        if (
            payload.get("identity") != "G9CB-2"
            or payload.get("protocol_version") != expected_protocol
            or payload.get(hash_field) != binding[hash_field]
        ):
            _fail(f"G9CB-2 historical metadata fields differ: {path_text}")
        amendments = (
            payload.get("bindings", {}).get("authority_amendments")
            if key == "preregistration"
            else payload.get("authority_amendments")
        )
        if amendments != _expected_authority_amendment_bindings():
            _fail(f"G9CB-2 historical amendments differ: {path_text}")
        for field in (
            "protocol_implementation_commit",
            "protocol_parent_commit",
            "claim_commit",
            "status",
            "retry_allowed",
            "resume_allowed",
        ):
            if field in binding and payload.get(field) != binding[field]:
                _fail(f"G9CB-2 historical field differs: {path_text}:{field}")
        return

    g9cb3_attempt = attempts[1]
    g9cb3: dict[str, tuple[str, Mapping[str, Any]]] = {
        g9cb3_attempt[key]["path"]: (key, g9cb3_attempt[key])
        for key in ("authority_decision", "preregistration", "access_claim")
    }
    g9cb3.update(
        {
            binding["path"]: (key, binding)
            for key, binding in g9cb3_attempt["terminal_evidence"].items()
        }
    )
    if path_text not in g9cb3:
        return
    key, binding = g9cb3[path_text]
    if "filesystem_mode_octal" in binding and stat.S_IMODE(
        info.st_mode
    ) != int(str(binding["filesystem_mode_octal"]), 8):
        _fail(f"G9CB-3 historical filesystem mode differs: {path_text}")
    if key == "authority_decision":
        return
    if key == "pass1_worker_ledger":
        try:
            payload = json.loads(
                raw.decode("utf-8"), object_pairs_hook=_unique_object
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TerminalG9CB12Failure(
                f"G9CB-3 worker ledger JSON is invalid: {path_text}"
            ) from exc
        if not isinstance(payload, dict) or raw != _canonical_json_bytes(payload):
            _fail(f"G9CB-3 worker ledger is not canonical: {path_text}")
    else:
        hash_field = "claim_hash" if key == "access_claim" else "manifest_hash"
        payload = _canonical_bound_json(raw, path_text, hash_field)
        if payload.get(hash_field) != binding[hash_field]:
            _fail(f"G9CB-3 historical internal hash differs: {path_text}")
    expected_protocol = binding.get(
        "protocol_version", g9cb3_attempt["protocol_version"]
    )
    if (
        payload.get("identity") != "G9CB-3"
        or payload.get("protocol_version") != expected_protocol
    ):
        _fail(f"G9CB-3 historical metadata fields differ: {path_text}")
    amendments = (
        payload.get("bindings", {}).get("authority_amendments")
        if key == "preregistration"
        else payload.get("authority_amendments")
    )
    if amendments != _expected_authority_amendment_bindings():
        _fail(f"G9CB-3 historical amendments differ: {path_text}")
    for field in (
        "protocol_implementation_commit",
        "protocol_parent_commit",
        "claim_commit",
        "claim_hash",
        "parent_pid",
        "slot",
        "stage_directory",
        "status",
        "retry_allowed",
        "resume_allowed",
    ):
        if field in binding and payload.get(field) != binding[field]:
            _fail(f"G9CB-3 historical field differs: {path_text}:{field}")


def _validate_regular_hashed_inputs(
    root: Path,
    preregistration: Mapping[str, Any],
    *,
    verify_git: bool = True,
    raw_cache: dict[str, tuple[bytes, os.stat_result]] | None = None,
    preclassified_pairs: Mapping[str, tuple[str, str] | None] | None = None,
) -> list[dict[str, Any]]:
    prepared: dict[str, dict[str, Any]] = {}
    declarations: dict[str, dict[str, Any]] = {}
    for binding in _iter_bindings(preregistration):
        path_text = _binding_path(binding)
        digest = binding.get("sha256")
        if not isinstance(digest, str) or not _SHA_RE.fullmatch(digest):
            _fail(f"bound input SHA-256 is not lowercase hexadecimal: {path_text}")
        candidate, repository_relative = _bound_regular_path(root, path_text)
        _validate_git_pair_shape(binding, path_text)
        declared_size: int | None = None
        if "size_bytes" in binding:
            declared_size = binding["size_bytes"]
            if (
                type(declared_size) is not int
                or declared_size < 0
            ):
                _fail(f"bound input size declaration is invalid: {path_text}")
        if "path_type" in binding and binding["path_type"] != "regular_file":
            _fail(f"bound input path_type mismatch: {path_text}")

        declared = declarations.setdefault(path_text, {})
        for key in (
            "path_type",
            "git_blob",
            "git_mode",
            "filesystem_mode_octal",
            "mode_octal",
        ):
            if key in binding:
                value = binding[key]
                if key in declared and declared[key] != value:
                    _fail(
                        f"conflicting duplicate input metadata: {path_text}:{key}"
                    )
                declared[key] = value

        row = {
            "path": path_text,
            "candidate": candidate,
            "repository_relative": repository_relative,
            "sha256": digest,
            "declared_sizes": (
                set() if declared_size is None else {declared_size}
            ),
        }
        prior = prepared.get(path_text)
        if prior is not None:
            if (
                prior["candidate"] != candidate
                or prior["repository_relative"] != repository_relative
                or prior["sha256"] != digest
            ):
                _fail(f"conflicting duplicate input binding: {path_text}")
            if declared_size is not None:
                prior["declared_sizes"].add(declared_size)
                if len(prior["declared_sizes"]) != 1:
                    _fail(f"conflicting duplicate input size: {path_text}")
            continue
        prepared[path_text] = row
    if not prepared:
        _fail("preregistration exposed no path/hash bindings")

    tracked_pairs: dict[str, tuple[str, str] | None] = {}
    for path_text in sorted(prepared):
        row = prepared[path_text]
        if preclassified_pairs is not None:
            if path_text not in preclassified_pairs:
                _fail(f"bound input lacks preclassification: {path_text}")
            tracked_pairs[path_text] = preclassified_pairs[path_text]
        else:
            tracked_pairs[path_text] = _validate_git_pair_preflight(
                root,
                path_text,
                repository_relative=bool(row["repository_relative"]),
                declaration=declarations[path_text],
                verify_git=verify_git,
            )

    if not isinstance(raw_cache, _SecureBoundSnapshot):
        for path_text in sorted(prepared):
            _require_bound_regular_lstat(
                prepared[path_text]["candidate"],
                path_text,
            )

    authenticated: list[dict[str, Any]] = []
    cache = {} if raw_cache is None else raw_cache
    for path_text in sorted(prepared):
        row = prepared[path_text]
        cached = cache.get(path_text)
        if cached is None:
            if isinstance(raw_cache, _SecureBoundSnapshot):
                _fail(f"bound input cache binding is absent: {path_text}")
            raw, info = _read_bound_regular_bytes(row["candidate"], path_text)
            cache[path_text] = (raw, info)
        else:
            raw, info = cached
        actual = _sha256_bytes(raw)
        if actual != row["sha256"]:
            _fail(f"bound input hash mismatch: {path_text}")
        declared_sizes = row["declared_sizes"]
        if declared_sizes and declared_sizes != {info.st_size}:
            _fail(f"bound input size mismatch: {path_text}")
        tracked_pair = tracked_pairs[path_text]
        if tracked_pair is not None:
            if _git_blob_id(raw) != tracked_pair[0]:
                _fail(f"bound input worktree Git blob mismatch: {path_text}")
            actual_git_mode = (
                "100755" if info.st_mode & 0o111 else "100644"
            )
            if actual_git_mode != tracked_pair[1]:
                _fail(f"bound input worktree Git mode mismatch: {path_text}")
        declared_modes = [
            declarations[path_text][key]
            for key in ("filesystem_mode_octal", "mode_octal")
            if key in declarations[path_text]
        ]
        if (
            declared_modes
            and (
                any(type(value) is not str for value in declared_modes)
                or len(set(declared_modes)) != 1
                or not re.fullmatch(r"0[0-7]{3}", declared_modes[0])
                or stat.S_IMODE(info.st_mode)
                != int(declared_modes[0], 8)
            )
        ):
            _fail(f"bound input filesystem mode mismatch: {path_text}")
        _validate_historical_metadata_bytes(path_text, raw, info)
        authenticated.append(
            {
                "path": path_text,
                "sha256": actual,
                "size_bytes": info.st_size,
            }
        )
    return authenticated


def _validate_one_static_closure(
    root: Path,
    preregistration: Mapping[str, Any],
    *,
    roots_key: str,
    closure_key: str,
    verify_git: bool,
    raw_cache: Mapping[str, tuple[bytes, os.stat_result]] | None = None,
) -> list[dict[str, Any]]:
    bindings = preregistration.get("bindings")
    if not isinstance(bindings, Mapping):
        _fail("preregistration bindings are absent")
    roots = bindings.get(roots_key)
    members = bindings.get(closure_key)
    if not isinstance(roots, list) or not all(isinstance(item, str) for item in roots):
        _fail(f"preregistration {roots_key} is invalid")
    if not isinstance(members, list):
        _fail(f"preregistration {closure_key} is invalid")
    discovered = _discover_import_closure(
        root,
        roots,
        raw_cache=raw_cache,
    )
    if [path.as_posix() for path in discovered] != [
        str(row.get("path")) for row in members if isinstance(row, Mapping)
    ]:
        _fail(f"{closure_key} independently discovered path set differs")
    result: list[dict[str, Any]] = []
    for row in members:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            _fail("invalid import closure member")
        path = root / row["path"]
        cached = (
            raw_cache.get(row["path"])
            if raw_cache is not None
            else None
        )
        if cached is not None:
            raw, info = cached
            if not stat.S_ISREG(info.st_mode):
                _fail(f"closure member is not regular: {row['path']}")
        elif isinstance(raw_cache, _SecureBoundSnapshot):
            _fail(f"closure cache member is absent: {row['path']}")
        else:
            if path.is_symlink() or not path.is_file():
                _fail(f"closure member is absent: {row['path']}")
            raw = path.read_bytes()
        try:
            ast.parse(raw, filename=row["path"])
        except SyntaxError as exc:
            raise TerminalG9CB12Failure(f"closure source cannot be parsed: {row['path']}") from exc
        observed = {
            "path": row["path"],
            "path_type": "regular_file",
            "sha256": _sha256_bytes(raw),
            "git_blob": (
                _git_text(root, "rev-parse", f"HEAD:{row['path']}")
                if verify_git
                else row.get("git_blob")
            ),
            "git_mode": (
                "100644" if verify_git else row.get("git_mode")
            ),
            "package_initializer": path.name == "__init__.py",
        }
        for key, value in observed.items():
            if row.get(key) != value:
                _fail(f"closure member metadata mismatch: {row['path']}:{key}")
        result.append(observed)
    if [row["path"] for row in result] != sorted(row["path"] for row in result):
        _fail("import closure inventory is not sorted")
    return result


def _validate_static_closures(
    root: Path,
    preregistration: Mapping[str, Any],
    *,
    verify_git: bool = True,
    raw_cache: Mapping[str, tuple[bytes, os.stat_result]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    bindings = preregistration.get("bindings")
    if not isinstance(bindings, Mapping):
        _fail("preregistration bindings are absent")
    expected_runtime_roots = [
        f"{module.replace('.', '/')}.py" for module in RUNTIME_IMPORT_MODULES
    ]
    if bindings.get("runtime_import_roots") != expected_runtime_roots:
        _fail("runtime import roots differ from builder imports")
    if (
        "adapter_import_roots" in bindings
        or "adapter_import_closure" in bindings
    ):
        _fail("superseded adapter import closure is present")
    runtime = _validate_one_static_closure(
        root,
        preregistration,
        roots_key="runtime_import_roots",
        closure_key="runtime_import_closure",
        verify_git=verify_git,
        raw_cache=raw_cache,
    )
    prohibited_modules = {
        "training.long_regime_combo_scan",
        "training.portfolio_opt_added_alpha_update",
        "training.portfolio_opt_all_discovered_alpha_gross10",
        "training.audit_fresh_kimchi_orthogonal_alpha",
        "training.compare_expanding_extratrees_rank7_refit_cadence_pre2025",
        "training.portfolio_opt_new_alpha_pool",
        "execution.portfolio_live",
        "execution.rank7_runtime",
        "execution.rex_llm_live",
    }
    closure_paths = {row["path"] for row in runtime}
    prohibited_paths = {
        f"{module.replace('.', '/')}.py" for module in prohibited_modules
    }
    overlap = sorted(closure_paths & prohibited_paths)
    if overlap:
        _fail(f"isolated runtime closure imports prohibited modules: {overlap}")
    return {"runtime": runtime}


def _worker_stage_path(root: Path, output_dir: Path) -> str:
    repository_root = root.resolve(strict=True)
    candidate = (
        output_dir
        if output_dir.is_absolute()
        else repository_root / output_dir
    )
    candidate = Path(os.path.abspath(candidate))
    expected_parent = _rooted(repository_root, SENTINEL_PATH).parent
    if candidate.parent != expected_parent:
        _fail("worker staging directory is not in the results filesystem")
    if (
        not candidate.name.startswith(".gross9-structural-clock-g9cb12-worker-")
        or candidate.name == ".gross9-structural-clock-g9cb12-worker-"
    ):
        _fail("worker staging directory name differs")
    return candidate.relative_to(repository_root).as_posix()


def _zero_token(token: bytearray) -> None:
    for index in range(len(token)):
        token[index] = 0


def _fill_random_token(token: bytearray) -> None:
    if len(token) != 32 or any(token):
        _fail("worker capability token buffer is not fresh")
    getrandom = getattr(_LIBC, "getrandom", None)
    if getrandom is None:
        _fail("libc getrandom is unavailable")
    getrandom.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint)
    getrandom.restype = ctypes.c_ssize_t
    storage = (ctypes.c_ubyte * len(token)).from_buffer(token)
    address = ctypes.addressof(storage)
    offset = 0
    while offset < len(token):
        ctypes.set_errno(0)
        count = int(
            getrandom(
                ctypes.c_void_p(address + offset),
                len(token) - offset,
                0,
            )
        )
        if count < 0:
            error = ctypes.get_errno()
            if error == errno.EINTR:
                continue
            _zero_token(token)
            _fail(f"libc getrandom failed: errno={error}")
        if count == 0:
            _zero_token(token)
            _fail("libc getrandom made no progress")
        offset += count


def _write_all(descriptor: int, raw: bytes | bytearray) -> None:
    view = memoryview(raw)
    offset = 0
    while offset < len(view):
        written = os.write(descriptor, view[offset:])
        if written <= 0:
            _fail("capability pipe write made no progress")
        offset += written


def _prepare_worker_capability(
    *,
    root: Path,
    output_dir: Path,
    slot: int,
    parent_pid: int,
    results_fd: int | None = None,
) -> dict[str, Any]:
    if type(slot) is not int or slot not in (1, 2):
        _fail("worker capability slot is invalid")
    if type(parent_pid) is not int or parent_pid <= 0:
        _fail("worker capability parent PID is invalid")
    stage_directory = _worker_stage_path(root, output_dir)
    read_fd = write_fd = ledger_fd = -1
    owned_results_fd = -1
    token = bytearray()
    try:
        read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
        info = os.fstat(read_fd)
        if not stat.S_ISFIFO(info.st_mode):
            _fail("worker capability carrier is not a FIFO")
        token = bytearray(32)
        _fill_random_token(token)
        _write_all(write_fd, token)
        os.close(write_fd)
        write_fd = -1
        if results_fd is None:
            owned_results_fd = os.open(
                root / "results",
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
            results_fd = owned_results_fd
        ledger_fd = os.open(
            ".",
            os.O_RDWR
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_TMPFILE", 0),
            0o600,
            dir_fd=results_fd,
        )
        os.fchmod(ledger_fd, 0o600)
        ledger_info = os.fstat(ledger_fd)
        if (
            not stat.S_ISREG(ledger_info.st_mode)
            or stat.S_IMODE(ledger_info.st_mode) != 0o600
            or ledger_info.st_size != 0
        ):
            _fail("worker ledger carrier is not an empty mode-0600 regular file")
        row = {
            "slot": slot,
            "parent_pid": parent_pid,
            "stage_directory": stage_directory,
            "carrier_kind": "anonymous_pipe_v1",
            "carrier_device": int(info.st_dev),
            "carrier_inode": int(info.st_ino),
            "token_sha256": _sha256_bytes(token),
            "consumed_ledger_path": WORKER_LEDGER_PATHS[
                slot - 1
            ].as_posix(),
            "ledger_carrier_kind": "unnamed_otmpfile_v1",
            "ledger_device": int(ledger_info.st_dev),
            "ledger_inode": int(ledger_info.st_ino),
            "ledger_initial_type": "regular_file",
            "ledger_initial_mode": "0600",
            "ledger_initial_size": 0,
        }
        return {
            "row": row,
            "read_fd": read_fd,
            "ledger_fd": ledger_fd,
            "token": token,
            "stage_path": Path(root).resolve() / stage_directory,
        }
    except BaseException:
        if write_fd >= 0:
            os.close(write_fd)
        if read_fd >= 0:
            os.close(read_fd)
        if ledger_fd >= 0:
            os.close(ledger_fd)
        _zero_token(token)
        raise
    finally:
        if owned_results_fd >= 0:
            os.close(owned_results_fd)


def _normalized_worker_capabilities(
    capabilities: Any,
) -> list[dict[str, Any]]:
    if not isinstance(capabilities, Sequence) or isinstance(
        capabilities, (str, bytes)
    ):
        _fail("worker capabilities are absent")
    rows = [dict(row) for row in capabilities if isinstance(row, Mapping)]
    required = {
        "slot",
        "parent_pid",
        "stage_directory",
        "carrier_kind",
        "carrier_device",
        "carrier_inode",
        "token_sha256",
        "consumed_ledger_path",
        "ledger_carrier_kind",
        "ledger_device",
        "ledger_inode",
        "ledger_initial_type",
        "ledger_initial_mode",
        "ledger_initial_size",
    }
    if len(rows) != 2 or any(set(row) != required for row in rows):
        _fail("worker capability schema differs")
    if any(type(row["slot"]) is not int for row in rows):
        _fail("worker capability slot type differs")
    if [row["slot"] for row in rows] != [1, 2]:
        _fail("worker capability slots differ")
    parent_pids = {row["parent_pid"] for row in rows}
    if (
        len(parent_pids) != 1
        or any(type(row["parent_pid"]) is not int for row in rows)
        or next(iter(parent_pids)) <= 0
    ):
        _fail("worker capability parent PID differs")
    for row in rows:
        stage = row["stage_directory"]
        if (
            not isinstance(stage, str)
            or not stage.startswith("results/.gross9-structural-clock-g9cb12-worker-")
            or row["carrier_kind"] != "anonymous_pipe_v1"
            or type(row["carrier_device"]) is not int
            or type(row["carrier_inode"]) is not int
            or row["carrier_device"] < 0
            or row["carrier_inode"] <= 0
            or not isinstance(row["token_sha256"], str)
            or not _SHA_RE.fullmatch(row["token_sha256"])
            or row["consumed_ledger_path"]
            != WORKER_LEDGER_PATHS[row["slot"] - 1].as_posix()
            or row["ledger_carrier_kind"] != "unnamed_otmpfile_v1"
            or type(row["ledger_device"]) is not int
            or type(row["ledger_inode"]) is not int
            or row["ledger_device"] < 0
            or row["ledger_inode"] <= 0
            or row["ledger_initial_type"] != "regular_file"
            or row["ledger_initial_mode"] != "0600"
            or row["ledger_initial_size"] != 0
        ):
            _fail("worker capability binding differs")
    for key in (
        "stage_directory",
        "consumed_ledger_path",
        "token_sha256",
    ):
        if len({row[key] for row in rows}) != 2:
            _fail(f"worker capability {key} values are not unique")
    if len(
        {(row["carrier_device"], row["carrier_inode"]) for row in rows}
    ) != 2:
        _fail("worker capability carrier identities are not unique")
    if len(
        {(row["ledger_device"], row["ledger_inode"]) for row in rows}
    ) != 2:
        _fail("worker ledger carrier identities are not unique")
    if {
        (row["carrier_device"], row["carrier_inode"]) for row in rows
    } & {
        (row["ledger_device"], row["ledger_inode"]) for row in rows
    }:
        _fail("worker capability and ledger carrier identities alias")
    return rows


def _sentinel_payload(
    claim_binding: Mapping[str, Any],
    prereg_binding: Mapping[str, Any],
    authority_amendments: Sequence[Mapping[str, Any]],
    parent_authentication_sha256: str,
    worker_capabilities: Sequence[Mapping[str, Any]],
    *,
    expected_authority_amendments: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    capabilities = _normalized_worker_capabilities(worker_capabilities)
    amendments = [dict(row) for row in authority_amendments]
    expected_amendments = (
        _expected_authority_amendment_bindings()
        if expected_authority_amendments is None
        else [dict(row) for row in expected_authority_amendments]
    )
    if amendments != expected_amendments:
        _fail("sentinel authority amendments differ")
    if not _SHA_RE.fullmatch(parent_authentication_sha256):
        _fail("sentinel parent authentication hash differs")
    core = {
        "identity": IDENTITY,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_parent_commit": claim_binding["protocol_parent_commit"],
        "claim_commit": claim_binding["claim_commit"],
        "access_claim": {
            key: claim_binding[key] for key in ("path", "sha256", "claim_hash")
        },
        "preregistration": dict(prereg_binding),
        "authority_amendments": amendments,
        "parent_authentication_sha256": parent_authentication_sha256,
        "canonical_csv_path": CSV_PATH.as_posix(),
        "final_manifest_path": MANIFEST_PATH.as_posix(),
        "status": "attempt_consumed_before_runtime_or_value_access",
        "one_shot": True,
        "retry_allowed": False,
        "resume_allowed": False,
        "worker_capabilities": capabilities,
        "access_at_publication": {
            "runtime_modules_imported": 0,
            "source_value_rows_opened": 0,
            "pre2025_anchor_value_rows_opened": 0,
            "candidate_rows_opened": 0,
            "comparator_clock_rows_opened": 0,
        },
    }
    return _with_hash(core, "manifest_hash")


def _consume_worker_capability(
    descriptor: int,
    binding: Mapping[str, Any],
) -> bytearray:
    if type(descriptor) is not int or descriptor < 0:
        _fail("worker capability descriptor is invalid")
    info = os.fstat(descriptor)
    if (
        not stat.S_ISFIFO(info.st_mode)
        or info.st_dev != binding["carrier_device"]
        or info.st_ino != binding["carrier_inode"]
    ):
        _fail("worker capability carrier identity differs")
    token = bytearray(32)
    closed = False
    try:
        offset = 0
        while offset < len(token):
            count = os.readv(descriptor, [memoryview(token)[offset:]])
            if count <= 0:
                _fail("worker capability reached EOF before 32 bytes")
            offset += count
        if os.read(descriptor, 1) != b"":
            _fail("worker capability contains extra bytes")
        os.close(descriptor)
        closed = True
        if _sha256_bytes(token) != binding["token_sha256"]:
            _fail("worker capability token hash differs")
        return token
    except BaseException:
        _zero_token(token)
        if not closed:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


def _establish_parent_death_contract(expected_parent_pid: int) -> None:
    if type(expected_parent_pid) is not int or expected_parent_pid <= 0:
        _fail("expected parent PID is invalid")
    before = os.getppid()
    result = _LIBC.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    after = os.getppid()
    if result != 0:
        error = ctypes.get_errno()
        _fail(f"PR_SET_PDEATHSIG failed: errno={error}")
    if before != expected_parent_pid or after != expected_parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)
        _fail("parent-death race check returned after SIGKILL")


def _source_only_get_code(
    loader: importlib.machinery.SourceFileLoader,
    fullname: str,
) -> Any:
    source_path = loader.get_filename(fullname)
    source_bytes = loader.get_data(source_path)
    return loader.source_to_code(source_bytes, source_path)


def _validate_bytecode_preflight(root: Path) -> None:
    try:
        prereg.validate_repository_bytecode_preflight(root)
    except (OSError, ValueError) as exc:
        raise TerminalG9CB12Failure(str(exc)) from exc


class _WorkerIsolationGuard:
    """Non-removable worker-local process, path, mutation, and IPC guard."""

    _PROCESS_EVENTS = {
        "os.chdir",
        "os.chroot",
        "os.fork",
        "os.forkpty",
        "os.posix_spawn",
        "os.exec",
        "os.system",
        "subprocess.Popen",
    }
    _SOCKET_EVENTS = {
        "mmap.__new__",
        "socket.__new__",
        "socket.bind",
        "socket.connect",
        "socket.listen",
    }
    _AUDIT_PATH_EVENTS = {
        "os.listdir": (0,),
        "os.scandir": (0,),
    }
    _AUDIT_MUTATION_EVENTS = {
        "os.mkdir": ((0,), (2,)),
        "os.chmod": ((0,), (2,)),
        "os.chown": ((0,), (3,)),
        "os.utime": ((0,), (3,)),
        "os.truncate": ((0,), ()),
        "os.setxattr": ((0,), ()),
        "os.removexattr": ((0,), ()),
        "os.remove": ((0,), (1,)),
        "os.rmdir": ((0,), (1,)),
        "os.link": ((0, 1), (2, 3)),
        "os.rename": ((0, 1), (2, 3)),
        "os.symlink": ((0, 1), (2,)),
    }

    def __init__(
        self,
        *,
        root: Path,
        own_stage: str,
        other_stage: str,
        ledger_paths: Sequence[Path],
        repository_fd: int,
        results_fd: int,
        filesystem_root_fd: int,
        ledger_fd: int,
    ) -> None:
        root_path = Path(root)
        if not root_path.is_absolute():
            _fail("worker guard repository root is not absolute")
        self.root = Path(os.path.normpath(os.fspath(root_path)))
        self.cwd = Path(os.path.normpath(os.getcwd()))
        self.pycache_prefix = self.root / _PYCACHE_PREFIX_RELATIVE
        self.own_stage = self._absolute_protocol_path(own_stage)
        self.other_stage = self._absolute_protocol_path(other_stage)
        if tuple(ledger_paths) != WORKER_LEDGER_PATHS:
            _fail("worker guard ledger inventory differs")
        self.ledger_paths = tuple(
            self.root / path for path in ledger_paths
        )
        self.own_ledger: Path | None = None
        self.other_ledger: Path | None = None
        self._forbidden_ledgers = set(self.ledger_paths)
        self._owned_ledger_open_state = "unbound"
        self._owned_ledger_open_count = 0
        self.results_directory = (self.root / SENTINEL_PATH).parent
        self.allowed_mutations: set[str] = set()
        self.descriptors: dict[int, tuple[str, int, int, bool]] = {}
        self.allowed_directory_identities: set[tuple[int, int]] = set()
        self.child_process_creation_events = 0
        self.other_stage_access_events = 0
        self.other_stage_absence_checks = 0
        self.other_slot_ledger_access_events = 0
        self.unauthorized_write_or_ipc_events = 0
        self._other_stage_internal_check = False
        self._installed = False
        self._originals: dict[tuple[int, str], Any] = {}
        self._prebound_guarded_callables: dict[int, tuple[Any, str]] = {}
        self._original_lstat = os.lstat
        self._original_stat = os.stat
        self._original_os_open = os.open
        self._original_readlink = os.readlink
        self._original_realpath = os.path.realpath
        if len({repository_fd, results_fd, filesystem_root_fd, ledger_fd}) != 4:
            _fail("worker guard descriptor graph aliases")
        self.repository_fd = repository_fd
        self.results_fd = results_fd
        self.filesystem_root_fd = filesystem_root_fd
        self.ledger_fd = ledger_fd
        self.stage_fd: int | None = None
        self.stage_entries: tuple[str, ...] = ()
        self.stage_token: tuple[int, ...] | None = None
        self.results_entries: tuple[str, ...] = ()
        self.results_timestamp_token: tuple[int, ...] | None = None
        for descriptor, label in (
            (repository_fd, self.root.as_posix()),
            (results_fd, self.results_directory.as_posix()),
            (filesystem_root_fd, "/"),
        ):
            info = os.fstat(descriptor)
            if not stat.S_ISDIR(info.st_mode):
                _fail(f"worker anchor is not a directory: {label}")
            self.descriptors[descriptor] = (
                label,
                int(info.st_dev),
                int(info.st_ino),
                True,
            )
            self.allowed_directory_identities.add(
                (int(info.st_dev), int(info.st_ino))
            )
            if descriptor == results_fd:
                self.results_entries = _directory_entries(descriptor)
                self.results_timestamp_token = _descriptor_token(info)
        info = os.fstat(ledger_fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_size != 0
        ):
            _fail("worker pending ledger descriptor differs")
        self.descriptors[ledger_fd] = (
            "<unnamed-worker-ledger>",
            int(info.st_dev),
            int(info.st_ino),
            False,
        )

    def _absolute_protocol_path(self, text: str) -> Path:
        if not isinstance(text, str):
            _fail("worker stage invocation path is not text")
        relative = Path(text)
        if relative.is_absolute():
            candidate = Path(os.path.abspath(relative))
        else:
            candidate = self.root / relative
        try:
            candidate.relative_to(self.root)
        except ValueError:
            _fail(f"worker stage path escapes repository: {text}")
        return candidate

    def register_snapshot_descriptor(
        self,
        descriptor: int,
        label: str,
        info: os.stat_result,
        directory: bool,
    ) -> None:
        if descriptor in self.descriptors:
            _fail(f"worker snapshot descriptor was already registered: {label}")
        expected_type = stat.S_IFDIR if directory else stat.S_IFREG
        if stat.S_IFMT(info.st_mode) != expected_type:
            _fail(f"worker snapshot descriptor type differs: {label}")
        self.descriptors[descriptor] = (
            label,
            int(info.st_dev),
            int(info.st_ino),
            directory,
        )
        if directory:
            self.allowed_directory_identities.add(
                (int(info.st_dev), int(info.st_ino))
            )

    @staticmethod
    def _is_within(path: Path, prefix: Path) -> bool:
        return path == prefix or prefix in path.parents

    @staticmethod
    def _proc_descriptor_namespace(path: Path) -> bool:
        parts = path.parts
        if len(parts) >= 3 and parts[:3] == ("/", "dev", "fd"):
            return True
        if len(parts) < 4 or parts[0:2] != ("/", "proc"):
            return False
        rest = list(parts[2:])
        if rest[:2] in (["self", "fd"], ["self", "fdinfo"]):
            return True
        if rest[:2] in (
            ["thread-self", "fd"],
            ["thread-self", "fdinfo"],
        ):
            return True
        if (
            len(rest) >= 4
            and rest[0] == "self"
            and rest[1] == "task"
            and rest[2].isdigit()
            and rest[3] in ("fd", "fdinfo")
        ):
            return True
        if rest and rest[0].isdigit():
            if len(rest) >= 2 and rest[1] in ("fd", "fdinfo"):
                return True
            if (
                len(rest) >= 4
                and rest[1] == "task"
                and rest[2].isdigit()
                and rest[3] in ("fd", "fdinfo")
            ):
                return True
        return False

    def _decode_path(self, value: Any) -> Path:
        if isinstance(value, int):
            _fail(f"integer path descriptors are forbidden here: {value}")
        try:
            raw = os.fspath(value)
        except TypeError as exc:
            raise TerminalG9CB12Failure("guarded path is not path-like") from exc
        if isinstance(raw, bytes):
            text = raw.decode(sys.getfilesystemencoding(), "surrogateescape")
        elif isinstance(raw, str):
            text = raw
        else:
            _fail("guarded path did not normalize to text")
        if "\x00" in text:
            _fail("guarded path contains NUL")
        if not os.path.isabs(text):
            text = os.path.join(self.cwd.as_posix(), text)
        return Path(os.path.normpath(text))

    def _check_forbidden_form(self, path: Path) -> None:
        if self._proc_descriptor_namespace(path):
            self.unauthorized_write_or_ipc_events += 1
            _fail(f"descriptor namespace access is forbidden: {path}")
        if self._is_within(path, self.other_stage):
            if not self._other_stage_internal_check:
                self.other_stage_access_events += 1
                _fail(f"other worker stage access is forbidden: {path}")
        for ledger in self._forbidden_ledgers:
            if self._is_within(path, ledger):
                self.other_slot_ledger_access_events += 1
                _fail(f"worker ledger access is forbidden before binding: {path}")

    def _resolve_symlinks(self, lexical: Path) -> Path:
        candidate = lexical
        for _ in range(64):
            self._check_forbidden_form(candidate)
            parts = candidate.parts
            current = Path(parts[0])
            changed = False
            for index, component in enumerate(parts[1:], start=1):
                current /= component
                try:
                    info = self._original_lstat(current)
                except FileNotFoundError:
                    return candidate
                except OSError as exc:
                    raise TerminalG9CB12Failure(
                        f"guarded path resolution failed: {candidate}"
                    ) from exc
                if stat.S_ISLNK(info.st_mode):
                    target = self._original_readlink(current)
                    target_path = Path(target)
                    if not target_path.is_absolute():
                        target_path = current.parent / target_path
                    remainder = parts[index + 1 :]
                    candidate = Path(
                        os.path.normpath(
                            os.path.join(
                                target_path.as_posix(),
                                *remainder,
                            )
                        )
                    )
                    changed = True
                    break
            if not changed:
                return candidate
        _fail(f"guarded symlink resolution loop: {lexical}")

    def _checked_path(
        self,
        value: Any,
        *,
        mutation: bool = False,
        fifo_open: bool = False,
    ) -> str:
        lexical = self._decode_path(value)
        self._check_forbidden_form(lexical)
        resolved = self._resolve_symlinks(lexical)
        self._check_forbidden_form(resolved)
        for candidate in (lexical, resolved):
            if (
                self._is_within(candidate, self.root)
                or self._is_within(candidate, self.pycache_prefix)
            ) and (
                candidate.suffix == ".pyc"
                or candidate.suffix == ".pyo"
                or "__pycache__" in candidate.parts
            ):
                self.unauthorized_write_or_ipc_events += 1
                _fail(
                    "repository bytecode-cache access is forbidden: "
                    f"{candidate}"
                )
        if fifo_open:
            try:
                if stat.S_ISFIFO(self._original_lstat(resolved).st_mode):
                    self.unauthorized_write_or_ipc_events += 1
                    _fail(f"path-based FIFO access is forbidden: {resolved}")
            except FileNotFoundError:
                pass
        canonical = resolved.as_posix()
        if mutation and canonical not in self.allowed_mutations:
            self.unauthorized_write_or_ipc_events += 1
            _fail(f"filesystem mutation target is not allowlisted: {canonical}")
        return canonical

    @staticmethod
    def _reject_dir_fds(kwargs: Mapping[str, Any]) -> None:
        for key in ("dir_fd", "src_dir_fd", "dst_dir_fd"):
            if key in kwargs and kwargs[key] is not None:
                _fail(f"guarded {key} is forbidden")

    def _observation_dir_fd(
        self, path: Any, kwargs: Mapping[str, Any]
    ) -> int | None:
        for key in ("src_dir_fd", "dst_dir_fd"):
            if key in kwargs and kwargs[key] is not None:
                _fail(f"guarded {key} is forbidden")
        descriptor = kwargs.get("dir_fd")
        if descriptor is None:
            return None
        if (
            type(descriptor) is not int
            or descriptor not in self.descriptors
            or not self.descriptors[descriptor][3]
        ):
            _fail("guarded observation dir_fd is not registered")
        if (
            not isinstance(path, (str, bytes))
            or os.fsdecode(path) in {"", ".", ".."}
            or "/" in os.fsdecode(path)
            or "\\" in os.fsdecode(path)
        ):
            _fail("guarded descriptor-relative component is invalid")
        if (
            descriptor == self.results_fd
            and os.path.basename(os.path.normpath(os.fsdecode(path)))
            in {ledger.name for ledger in self.ledger_paths}
        ):
            self.other_slot_ledger_access_events += 1
            _fail("worker ledger observation is forbidden")
        return descriptor

    def _patch(self, owner: Any, name: str, replacement: Any) -> None:
        if not hasattr(owner, name):
            return
        key = (id(owner), name)
        if key not in self._originals:
            self._originals[key] = getattr(owner, name)
        setattr(owner, name, replacement)

    def _capture_non_audited_prebind_inventory(self) -> None:
        inventory: dict[int, tuple[Any, str]] = {}
        owner_specs = (
            (
                os,
                "os",
                (
                    "stat",
                    "lstat",
                    "access",
                    "readlink",
                    "dup",
                    "dup2",
                    "pipe",
                    "pipe2",
                    "openpty",
                    "mkfifo",
                    "mknod",
                    "memfd_create",
                    "eventfd",
                    "pidfd_open",
                    "write",
                    "pwrite",
                    "writev",
                    "pwritev",
                    "copy_file_range",
                    "sendfile",
                    "splice",
                    "ftruncate",
                    "posix_fallocate",
                    "fchmod",
                    "fchown",
                    "fsync",
                    "fdatasync",
                    "setxattr",
                    "removexattr",
                ),
            ),
            (fcntl, "fcntl", ("fcntl",)),
        )
        for owner, prefix, names in owner_specs:
            for name in names:
                value = getattr(owner, name, None)
                if callable(value):
                    inventory[id(value)] = (value, f"{prefix}.{name}")
        self._prebound_guarded_callables = inventory

    def _reject_preinstall_callable_references(self) -> None:
        scopes: list[tuple[str, dict[str, Any]]] = []
        current_frame = sys._getframe()
        for thread_id, frame in sys._current_frames().items():
            cursor = frame
            depth = 0
            while cursor is not None:
                if cursor is not current_frame:
                    scopes.append(
                        (
                            f"thread-{thread_id}-frame-{depth}",
                            dict(cursor.f_locals),
                        )
                    )
                cursor = cursor.f_back
                depth += 1
        for module_name, module in list(sys.modules.items()):
            if module is None:
                continue
            module_file = getattr(module, "__file__", None)
            repository_module = module_name == "__main__"
            if isinstance(module_file, (str, bytes, os.PathLike)):
                raw = os.fspath(module_file)
                if isinstance(raw, bytes):
                    raw = raw.decode(
                        sys.getfilesystemencoding(),
                        "surrogateescape",
                    )
                candidate = Path(raw)
                if not candidate.is_absolute():
                    candidate = self.cwd / candidate
                try:
                    candidate.relative_to(self.root)
                except ValueError:
                    pass
                else:
                    repository_module = True
            if repository_module:
                scopes.append(
                    (f"module-{module_name}", dict(vars(module)))
                )
        for scope_name, scope in scopes:
            for binding_name, value in scope.items():
                protected = self._prebound_guarded_callables.get(id(value))
                if protected is None or value is not protected[0]:
                    continue
                self.unauthorized_write_or_ipc_events += 1
                _fail(
                    "preinstall retained guarded callable is forbidden: "
                    f"{protected[1]} at {scope_name}:{binding_name}"
                )

    def _reject_process(self, name: str):
        def rejected(*_args: Any, **_kwargs: Any) -> Any:
            self.child_process_creation_events += 1
            _fail(f"worker process/descriptor operation is forbidden: {name}")

        return rejected

    def _reject_ipc(self, name: str):
        def rejected(*_args: Any, **_kwargs: Any) -> Any:
            self.unauthorized_write_or_ipc_events += 1
            _fail(f"worker IPC operation is forbidden: {name}")

        return rejected

    def _rejecting_type(
        self,
        original: type[Any],
        name: str,
        *,
        process: bool,
    ) -> type[Any]:
        guard = self

        class RejectedType(original):
            def __new__(cls, *_args: Any, **_kwargs: Any) -> Any:
                if process:
                    guard.child_process_creation_events += 1
                    _fail(
                        "worker process/descriptor operation is forbidden: "
                        f"{name}"
                    )
                guard.unauthorized_write_or_ipc_events += 1
                _fail(f"worker IPC operation is forbidden: {name}")

        RejectedType.__name__ = original.__name__
        RejectedType.__qualname__ = original.__qualname__
        RejectedType.__module__ = original.__module__
        return RejectedType

    def _wrap_open(self, original: Any):
        def guarded(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            self._reject_dir_fds(kwargs)
            mutation = any(character in str(mode) for character in "wax+")
            self._checked_path(file, mutation=mutation, fifo_open=True)
            return original(file, mode, *args, **kwargs)

        return guarded

    def _wrap_os_open(self, original: Any):
        mutation_mask = (
            os.O_WRONLY
            | os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | os.O_TRUNC
            | os.O_APPEND
        )

        def guarded(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
            tmpfile = getattr(os, "O_TMPFILE", 0)
            mutation = bool(flags & mutation_mask) or bool(
                tmpfile and (flags & tmpfile) == tmpfile
            )
            dir_fd = self._observation_dir_fd(path, kwargs)
            if dir_fd is not None and mutation:
                _fail("guarded descriptor-relative mutation is forbidden")
            canonical = (
                f"{self.descriptors[dir_fd][0]}/{os.fsdecode(path)}"
                if dir_fd is not None
                else self._checked_path(
                    path,
                    mutation=mutation,
                    fifo_open=True,
                )
            )
            descriptor = original(path, flags, *args, **kwargs)
            info = os.fstat(descriptor)
            is_directory = stat.S_ISDIR(info.st_mode)
            if stat.S_ISFIFO(info.st_mode):
                os.close(descriptor)
                self.unauthorized_write_or_ipc_events += 1
                _fail(f"path-resolved FIFO access is forbidden: {canonical}")
            if (
                is_directory
                and dir_fd is None
                and (info.st_dev, info.st_ino)
                not in self.allowed_directory_identities
            ):
                os.close(descriptor)
                self.unauthorized_write_or_ipc_events += 1
                _fail(f"directory descriptor is not authorized: {canonical}")
            self.descriptors[descriptor] = (
                canonical,
                int(info.st_dev),
                int(info.st_ino),
                is_directory,
            )
            if is_directory:
                self.allowed_directory_identities.add(
                    (int(info.st_dev), int(info.st_ino))
                )
            return descriptor

        return guarded

    def _wrap_path_observation(self, original: Any, *, path_index: int = 0):
        def guarded(*args: Any, **kwargs: Any) -> Any:
            if len(args) <= path_index:
                path = "."
            else:
                path = args[path_index]
            if self._observation_dir_fd(path, kwargs) is None:
                self._checked_path(path)
            return original(*args, **kwargs)

        return guarded

    def _wrap_path_mutation(
        self,
        original: Any,
        *,
        path_indexes: tuple[int, ...] = (0,),
    ):
        def guarded(*args: Any, **kwargs: Any) -> Any:
            self._reject_dir_fds(kwargs)
            for index in path_indexes:
                if len(args) > index:
                    self._checked_path(args[index], mutation=True)
            return original(*args, **kwargs)

        return guarded

    def _require_mutable_descriptor(self, descriptor: int) -> None:
        if descriptor in (1, 2):
            return
        record = self.descriptors.get(descriptor)
        if record is None or record[0] not in self.allowed_mutations:
            self.unauthorized_write_or_ipc_events += 1
            _fail(f"descriptor mutation is not authorized: {descriptor}")

    def _wrap_descriptor_mutation(
        self,
        original: Any,
        *,
        descriptor_indexes: tuple[int, ...] = (0,),
        directory_sync: bool = False,
    ):
        def guarded(*args: Any, **kwargs: Any) -> Any:
            for index in descriptor_indexes:
                descriptor = args[index]
                if directory_sync:
                    record = self.descriptors.get(descriptor)
                    if record is not None and record[3] and (
                        record[1],
                        record[2],
                    ) in self.allowed_directory_identities:
                        continue
                self._require_mutable_descriptor(descriptor)
            return original(*args, **kwargs)

        return guarded

    def _install_path_guards(self) -> None:
        original_close = os.close

        def guarded_close(descriptor: int) -> None:
            try:
                original_close(descriptor)
            finally:
                self.descriptors.pop(descriptor, None)

        self._patch(os, "close", guarded_close)
        self._patch(builtins, "open", self._wrap_open(builtins.open))
        self._patch(io, "open", self._wrap_open(io.open))
        self._patch(os, "open", self._wrap_os_open(os.open))

        for owner, names in (
            (
                os,
                (
                    "stat",
                    "lstat",
                    "access",
                    "readlink",
                    "listdir",
                    "scandir",
                    "walk",
                ),
            ),
            (
                os.path,
                (
                    "exists",
                    "lexists",
                    "isfile",
                    "isdir",
                    "islink",
                    "getsize",
                    "realpath",
                ),
            ),
        ):
            for name in names:
                if hasattr(owner, name):
                    original = getattr(owner, name)
                    self._patch(
                        owner,
                        name,
                        self._wrap_path_observation(original),
                    )

        path_observations = (
            "open",
            "read_text",
            "read_bytes",
            "stat",
            "lstat",
            "exists",
            "is_file",
            "is_dir",
            "is_symlink",
            "iterdir",
            "glob",
            "rglob",
        )
        for name in path_observations:
            if hasattr(Path, name):
                original = getattr(Path, name)

                def guarded_path(
                    instance: Path,
                    *args: Any,
                    __original: Any = original,
                    **kwargs: Any,
                ) -> Any:
                    self._reject_dir_fds(kwargs)
                    self._checked_path(instance)
                    return __original(instance, *args, **kwargs)

                self._patch(Path, name, guarded_path)

        path_mutations = (
            "write_text",
            "write_bytes",
            "mkdir",
            "touch",
            "chmod",
            "unlink",
            "symlink_to",
            "hardlink_to",
        )
        for name in path_mutations:
            if hasattr(Path, name):
                original = getattr(Path, name)

                def guarded_mutation(
                    instance: Path,
                    *args: Any,
                    __original: Any = original,
                    **kwargs: Any,
                ) -> Any:
                    self._reject_dir_fds(kwargs)
                    self._checked_path(instance, mutation=True)
                    return __original(instance, *args, **kwargs)

                self._patch(Path, name, guarded_mutation)
        for name in ("rename", "replace"):
            if hasattr(Path, name):
                original = getattr(Path, name)

                def guarded_move(
                    instance: Path,
                    target: Any,
                    *args: Any,
                    __original: Any = original,
                    **kwargs: Any,
                ) -> Any:
                    self._reject_dir_fds(kwargs)
                    self._checked_path(instance, mutation=True)
                    self._checked_path(target, mutation=True)
                    return __original(instance, target, *args, **kwargs)

                self._patch(Path, name, guarded_move)

    def _install_mutation_guards(self) -> None:
        one_path = (
            "mkdir",
            "makedirs",
            "mkfifo",
            "mknod",
            "remove",
            "unlink",
            "rmdir",
            "removedirs",
            "truncate",
            "chmod",
            "chown",
            "utime",
        )
        two_paths = ("link", "symlink", "rename", "replace")
        for name in one_path:
            if hasattr(os, name):
                original = getattr(os, name)
                self._patch(
                    os,
                    name,
                    self._wrap_path_mutation(original),
                )
        for name in two_paths:
            if hasattr(os, name):
                original = getattr(os, name)
                self._patch(
                    os,
                    name,
                    self._wrap_path_mutation(
                        original,
                        path_indexes=(0, 1),
                    ),
                )

        descriptor_specs = {
            "write": ((0,), False),
            "pwrite": ((0,), False),
            "writev": ((0,), False),
            "pwritev": ((0,), False),
            "copy_file_range": ((1,), False),
            "sendfile": ((0,), False),
            "splice": ((1,), False),
            "ftruncate": ((0,), False),
            "posix_fallocate": ((0,), False),
            "fchmod": ((0,), False),
            "fchown": ((0,), False),
            "fsync": ((0,), True),
            "fdatasync": ((0,), True),
            "setxattr": ((0,), False),
            "removexattr": ((0,), False),
        }
        for name, (indexes, directory_sync) in descriptor_specs.items():
            if hasattr(os, name):
                original = getattr(os, name)
                self._patch(
                    os,
                    name,
                    self._wrap_descriptor_mutation(
                        original,
                        descriptor_indexes=indexes,
                        directory_sync=directory_sync,
                    ),
                )
        for name in (
            "TemporaryFile",
            "NamedTemporaryFile",
            "SpooledTemporaryFile",
            "mkstemp",
            "mkdtemp",
        ):
            if hasattr(tempfile, name):
                self._patch(tempfile, name, self._reject_ipc(f"tempfile.{name}"))

    def _install_process_and_ipc_guards(self) -> None:
        for name in (
            "dup",
            "dup2",
            "chdir",
            "fchdir",
            "chroot",
            "fork",
            "forkpty",
            "posix_spawn",
            "posix_spawnp",
            "execl",
            "execle",
            "execlp",
            "execlpe",
            "execv",
            "execve",
            "execvp",
            "execvpe",
            "spawnl",
            "spawnle",
            "spawnlp",
            "spawnlpe",
            "spawnv",
            "spawnve",
            "spawnvp",
            "spawnvpe",
            "system",
            "popen",
        ):
            if hasattr(os, name):
                self._patch(os, name, self._reject_process(f"os.{name}"))

        original_fcntl = fcntl.fcntl

        def guarded_fcntl(fd: int, command: int, *args: Any) -> Any:
            duplicate_commands = {
                value
                for value in (
                    getattr(fcntl, "F_DUPFD", None),
                    getattr(fcntl, "F_DUPFD_CLOEXEC", None),
                )
                if value is not None
            }
            if command in duplicate_commands:
                self.child_process_creation_events += 1
                _fail("fcntl descriptor duplication is forbidden")
            return original_fcntl(fd, command, *args)

        self._patch(fcntl, "fcntl", guarded_fcntl)
        for name in (
            "Popen",
            "run",
            "call",
            "check_call",
            "check_output",
        ):
            original = getattr(subprocess, name)
            self._patch(
                subprocess,
                name,
                self._rejecting_type(
                    original,
                    f"subprocess.{name}",
                    process=True,
                )
                if isinstance(original, type)
                else self._reject_process(f"subprocess.{name}"),
            )
        self._patch(
            multiprocessing.process.BaseProcess,
            "start",
            self._reject_process("multiprocessing.process.BaseProcess.start"),
        )
        process_pool = concurrent.futures.process.ProcessPoolExecutor
        self._patch(
            concurrent.futures.process,
            "ProcessPoolExecutor",
            self._rejecting_type(
                process_pool,
                "concurrent.futures.ProcessPoolExecutor",
                process=True,
            ),
        )

        for name in (
            "pipe",
            "pipe2",
            "openpty",
            "mkfifo",
            "mknod",
            "memfd_create",
            "eventfd",
            "pidfd_open",
        ):
            if hasattr(os, name):
                self._patch(os, name, self._reject_ipc(f"os.{name}"))
        self._patch(pty, "openpty", self._reject_ipc("pty.openpty"))
        for name in ("socket", "socketpair", "fromfd"):
            if hasattr(socket, name):
                original = getattr(socket, name)
                self._patch(socket, name, self._reject_ipc(f"socket.{name}"))
                if isinstance(original, type):
                    self._patch(
                        socket,
                        name,
                        self._rejecting_type(
                            original,
                            f"socket.{name}",
                            process=False,
                        ),
                    )
        for owner, names, prefix in (
            (
                multiprocessing,
                (
                    "Pipe",
                    "Queue",
                    "SimpleQueue",
                    "JoinableQueue",
                    "Manager",
                ),
                "multiprocessing",
            ),
            (
                multiprocessing.connection,
                ("Listener",),
                "multiprocessing.connection",
            ),
            (
                multiprocessing.shared_memory,
                ("SharedMemory",),
                "multiprocessing.shared_memory",
            ),
        ):
            for name in names:
                if hasattr(owner, name):
                    original = getattr(owner, name)
                    self._patch(
                        owner,
                        name,
                        self._rejecting_type(
                            original,
                            f"{prefix}.{name}",
                            process=False,
                        )
                        if isinstance(original, type)
                        else self._reject_ipc(f"{prefix}.{name}"),
                    )
        self._patch(
            mmap,
            "mmap",
            self._rejecting_type(
                mmap.mmap,
                "mmap.mmap",
                process=False,
            ),
        )

    def _install_source_only_loader(self) -> None:
        self._patch(
            importlib.machinery.SourceFileLoader,
            "get_code",
            _source_only_get_code,
        )

    def _audit_mutation_path(self, value: Any) -> None:
        if isinstance(value, int):
            self._require_mutable_descriptor(value)
            return
        self._checked_path(value, mutation=True)

    def _audit(self, event: str, arguments: tuple[Any, ...]) -> None:
        if event in self._PROCESS_EVENTS:
            self.child_process_creation_events += 1
            _fail(f"audit rejected worker process event: {event}")
        if event in self._SOCKET_EVENTS:
            self.unauthorized_write_or_ipc_events += 1
            _fail(f"audit rejected worker IPC event: {event}")
        path_indexes = self._AUDIT_PATH_EVENTS.get(event)
        if path_indexes is not None:
            for index in path_indexes:
                if len(arguments) > index:
                    value = arguments[index]
                    if isinstance(value, int):
                        record = self.descriptors.get(value)
                        if record is None or not record[3]:
                            _fail(
                                "audit rejected unregistered directory "
                                f"descriptor: {event}"
                            )
                    else:
                        self._checked_path(value)
        mutation_spec = self._AUDIT_MUTATION_EVENTS.get(event)
        if mutation_spec is not None:
            mutation_indexes, directory_indexes = mutation_spec
            for index in directory_indexes:
                if (
                    len(arguments) > index
                    and arguments[index] not in (None, -1)
                ):
                    self.unauthorized_write_or_ipc_events += 1
                    _fail(f"audit rejected guarded dir_fd: {event}")
            for index in mutation_indexes:
                if len(arguments) > index:
                    self._audit_mutation_path(arguments[index])
        if event == "open" and arguments:
            path = arguments[0]
            if isinstance(path, (str, bytes, os.PathLike)):
                mode = arguments[1] if len(arguments) > 1 else "r"
                flags = arguments[2] if len(arguments) > 2 else 0
                decoded_path = os.fsdecode(path)
                normalized_leaf = os.path.basename(
                    os.path.normpath(decoded_path)
                )
                ledger_leaves = {ledger.name for ledger in self.ledger_paths}
                if normalized_leaf in ledger_leaves:
                    exact_flags = (
                        os.O_RDONLY
                        | getattr(os, "O_NOFOLLOW", 0)
                        | getattr(os, "O_CLOEXEC", 0)
                    )
                    if not (
                        self.own_ledger is not None
                        and decoded_path == self.own_ledger.name
                        and self._owned_ledger_open_state == "opening"
                        and self._owned_ledger_open_count == 1
                        and flags == exact_flags
                    ):
                        self.other_slot_ledger_access_events += 1
                        _fail("owned worker ledger open is forbidden")
                    return
                mutation = (
                    isinstance(mode, str)
                    and any(character in mode for character in "wax+")
                ) or (
                    isinstance(flags, int)
                    and (
                        bool(
                            flags
                            & (
                                os.O_WRONLY
                                | os.O_RDWR
                                | os.O_CREAT
                                | os.O_EXCL
                                | os.O_TRUNC
                                | os.O_APPEND
                            )
                        )
                        or bool(
                            getattr(os, "O_TMPFILE", 0)
                            and (
                                flags & getattr(os, "O_TMPFILE", 0)
                            )
                            == getattr(os, "O_TMPFILE", 0)
                        )
                    )
                )
                self._checked_path(
                    path,
                    mutation=mutation,
                    fifo_open=True,
                )

    def install(self) -> None:
        global _ACTIVE_WORKER_GUARD
        if self._installed:
            _fail("worker isolation guard cannot be reinstalled")
        self._installed = True
        sys.addaudithook(self._audit)
        self._capture_non_audited_prebind_inventory()
        self._reject_preinstall_callable_references()
        self._install_source_only_loader()
        self._install_path_guards()
        self._install_mutation_guards()
        self._install_process_and_ipc_guards()
        if self.other_stage_absence_checks != 0:
            _fail("other-stage absence guard was already used")
        self._other_stage_internal_check = True
        try:
            try:
                self._original_lstat(self.other_stage)
            except FileNotFoundError as exc:
                if exc.errno != errno.ENOENT:
                    _fail("other-stage absence check returned wrong errno")
            else:
                _fail("other worker stage exists")
            self.other_stage_absence_checks = 1
        finally:
            self._other_stage_internal_check = False
        _ACTIVE_WORKER_GUARD = self

    def bind_ledger_slot(self, slot: int) -> None:
        if type(slot) is not int or slot not in (1, 2):
            _fail("worker guard slot binding is invalid")
        if self.own_ledger is not None:
            _fail("worker guard ledger slot was already bound")
        self.own_ledger = self.ledger_paths[slot - 1]
        self.other_ledger = self.ledger_paths[1 - (slot - 1)]
        self._owned_ledger_open_state = "bound"

    def open_owned_canonical_ledger_once(
        self,
        leaf: str,
        flags: int,
        *,
        dir_fd: int,
    ) -> int:
        if (
            self._owned_ledger_open_state != "bound"
            or self._owned_ledger_open_count != 0
        ):
            self.unauthorized_write_or_ipc_events += 1
            _fail("owned worker ledger canonical-open authority differs")
        self._owned_ledger_open_state = "opening"
        self._owned_ledger_open_count = 1
        exact_flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor: int | None = None
        underlying_open_attempted = False
        try:
            if (
                self.own_ledger is None
                or type(leaf) is not str
                or leaf != self.own_ledger.name
                or type(flags) is not int
                or flags != exact_flags
                or type(dir_fd) is not int
                or dir_fd != self.results_fd
            ):
                _fail("owned worker ledger canonical-open authority differs")
            results_record = self.descriptors.get(dir_fd)
            if results_record is None or not results_record[3]:
                _fail("owned worker ledger results descriptor differs")
            results_info = os.fstat(dir_fd)
            if (int(results_info.st_dev), int(results_info.st_ino)) != (
                results_record[1],
                results_record[2],
            ):
                _fail(
                    "owned worker ledger results descriptor was substituted"
                )
            edge = self._original_stat(
                leaf,
                dir_fd=dir_fd,
                follow_symlinks=False,
            )
            unnamed = os.fstat(self.ledger_fd)
            if (
                not stat.S_ISREG(edge.st_mode)
                or stat.S_IMODE(edge.st_mode) != 0o444
                or (int(edge.st_dev), int(edge.st_ino))
                != (int(unnamed.st_dev), int(unnamed.st_ino))
            ):
                _fail("owned worker ledger pre-open edge differs")
            underlying_open_attempted = True
            descriptor = self._original_os_open(
                leaf,
                flags,
                dir_fd=dir_fd,
            )
            if descriptor == self.ledger_fd or descriptor in self.descriptors:
                _fail("owned worker ledger canonical descriptor aliases")
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or stat.S_IMODE(info.st_mode) != 0o444
                or (int(info.st_dev), int(info.st_ino))
                != (int(edge.st_dev), int(edge.st_ino))
            ):
                _fail("owned worker ledger canonical descriptor differs")
            self.descriptors[descriptor] = (
                self.own_ledger.as_posix(),
                int(info.st_dev),
                int(info.st_ino),
                False,
            )
        except BaseException:
            self._owned_ledger_open_state = "failed"
            if not underlying_open_attempted or descriptor is not None:
                self.unauthorized_write_or_ipc_events += 1
            if (
                descriptor is not None
                and descriptor != self.ledger_fd
                and descriptor not in self.descriptors
            ):
                os.close(descriptor)
            raise
        self._owned_ledger_open_state = "consumed"
        return descriptor

    def authorize_outputs(self) -> None:
        self.allowed_mutations.update(
            {
                (self.own_stage / _STAGED_CSV_NAME).as_posix(),
                (self.own_stage / _STAGED_CORE_NAME).as_posix(),
            }
        )

    def authorize_receipt(self) -> None:
        self.allowed_mutations.add(
            (self.own_stage / _STAGED_RECEIPT_NAME).as_posix()
        )

    def authorize_directory_sync(self, path: Path) -> None:
        canonical = self._checked_path(path)
        if canonical not in {
            self.results_directory.as_posix(),
            self.own_stage.as_posix(),
        }:
            _fail("worker directory durability target differs")
        if canonical == self.own_stage.as_posix() and self.stage_fd is None:
            assert self.results_fd is not None
            flags = (
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
            )
            self.stage_fd = self._original_os_open(
                self.own_stage.name, flags, dir_fd=self.results_fd
            )
            info = os.fstat(self.stage_fd)
            self.register_snapshot_descriptor(
                self.stage_fd, canonical, info, True
            )
            self.stage_entries = _directory_entries(self.stage_fd)
            self.stage_token = _descriptor_token(info)
            self.allowed_mutations.add(canonical)
        else:
            info = self._original_lstat(path)
        if not stat.S_ISDIR(info.st_mode):
            _fail("worker durability target is not a directory")
        self.allowed_directory_identities.add(
            (int(info.st_dev), int(info.st_ino))
        )

    def require_results(self, *, compare_timestamps: bool = True) -> None:
        assert self.repository_fd is not None and self.results_fd is not None
        info = os.fstat(self.results_fd)
        edge = self._original_stat(
            "results", dir_fd=self.repository_fd, follow_symlinks=False
        )
        record = self.descriptors[self.results_fd]
        if (
            self.results_timestamp_token is None
            or _directory_identity(info) != self.results_timestamp_token[:4]
            or (info.st_dev, info.st_ino) != (record[1], record[2])
            or _directory_identity(edge) != _directory_identity(info)
            or _directory_entries(self.results_fd) != self.results_entries
        ):
            _fail("worker retained results directory differs")
        if (
            compare_timestamps
            and self.results_timestamp_token is not None
            and _descriptor_token(info) != self.results_timestamp_token
        ):
            _fail("worker retained results timestamps drifted")

    def require_stage(
        self,
        expected_entries: tuple[str, ...],
        *,
        authorized_delta: bool = False,
    ) -> None:
        if self.stage_fd is None or self.stage_token is None:
            _fail("worker retained stage descriptor is absent")
        assert self.results_fd is not None
        info = os.fstat(self.stage_fd)
        edge = self._original_stat(
            self.own_stage.name,
            dir_fd=self.results_fd,
            follow_symlinks=False,
        )
        if (
            _directory_identity(info) != self.stage_token[:4]
            or _directory_identity(edge) != self.stage_token[:4]
            or _directory_entries(self.stage_fd) != expected_entries
        ):
            _fail("worker retained stage state differs")
        if authorized_delta:
            os.fchmod(self.stage_fd, 0o700)
            os.fsync(self.stage_fd)
            self.stage_entries = expected_entries
            self.stage_token = _descriptor_token(os.fstat(self.stage_fd))
        elif _descriptor_token(info) != self.stage_token:
            _fail("worker retained stage timestamps drifted")

    def counters(self) -> dict[str, int]:
        return {
            "child_process_creation_events": self.child_process_creation_events,
            "other_stage_access_events": self.other_stage_access_events,
            "other_stage_absence_checks": self.other_stage_absence_checks,
            "other_slot_ledger_access_events": (
                self.other_slot_ledger_access_events
            ),
            "unauthorized_write_or_ipc_events": (
                self.unauthorized_write_or_ipc_events
            ),
        }


def _preload_runtime_package_initializers(
    root: Path,
    preregistration: Mapping[str, Any],
) -> None:
    protocol = preregistration.get("bindings", {}).get("protocol")
    if not isinstance(protocol, list):
        _fail("protocol inventory is absent before initializer preload")
    protocol_paths = {
        str(row.get("path"))
        for row in protocol
        if isinstance(row, Mapping)
    }
    allowed = _RuntimeImportRecorder._protocol_package_initializer_paths(
        protocol_paths
    )
    package_names: set[str] = set()
    for module_name in RUNTIME_IMPORT_MODULES:
        components = module_name.split(".")[:-1]
        for length in range(1, len(components) + 1):
            package_names.add(".".join(components[:length]))
    for package_name in sorted(package_names):
        initializer = (
            PurePosixPath(*package_name.split(".")) / "__init__.py"
        ).as_posix()
        if initializer not in allowed:
            _fail(
                "runtime package initializer is not derived from a protocol "
                f"path: {initializer}"
            )
        expected = root / initializer
        if expected.is_symlink() or not expected.is_file():
            _fail(f"runtime package initializer is unsafe: {initializer}")
        module = importlib.import_module(package_name)
        observed = getattr(module, "__file__", None)
        if (
            not isinstance(observed, str)
            or Path(observed).resolve() != expected.resolve()
        ):
            _fail(
                "runtime package initializer resolved outside protocol "
                f"ancestry: {initializer}"
            )


class _RuntimeImportRecorder:
    """Authenticate and count repository-local source executions."""

    def __init__(
        self,
        *,
        root: Path,
        preregistration: Mapping[str, Any],
        runtime_closure: Sequence[Mapping[str, Any]],
    ) -> None:
        self.root = root.resolve(strict=True)
        self.expected = {
            str(row["path"]): str(row["sha256"])
            for row in runtime_closure
        }
        if len(self.expected) != len(runtime_closure):
            _fail("runtime import closure contains duplicate paths")
        protocol = preregistration.get("bindings", {}).get("protocol")
        if not isinstance(protocol, list):
            _fail("protocol inventory is absent for import recording")
        self.protocol_paths = {
            str(row.get("path"))
            for row in protocol
            if isinstance(row, Mapping)
        }
        self.allowed_initializer_paths = (
            self._protocol_package_initializer_paths(self.protocol_paths)
        )
        self.snapshot: set[str] = set()
        self.preloaded_repository_paths: set[str] = set()
        self.recorded: list[str] = []
        self._reject_mode = False
        self._installed = False
        self._active_source_paths: dict[str, int] = {}
        self._source_exec = importlib.machinery.SourceFileLoader.exec_module
        self._sourceless_exec = (
            importlib.machinery.SourcelessFileLoader.exec_module
        )

    @staticmethod
    def _protocol_package_initializer_paths(
        protocol_paths: Iterable[str],
    ) -> set[str]:
        initializers: set[str] = set()
        for text in protocol_paths:
            path = PurePosixPath(text)
            if (
                path.is_absolute()
                or any(part in ("", ".", "..") for part in path.parts)
            ):
                _fail(f"invalid protocol path for import recording: {text}")
            parent = path.parent
            while parent != PurePosixPath("."):
                initializers.add((parent / "__init__.py").as_posix())
                parent = parent.parent
        return initializers

    def _relative_repository_path(self, value: Any) -> str | None:
        if not isinstance(value, (str, bytes, os.PathLike)):
            return None
        raw = os.fspath(value)
        if isinstance(raw, bytes):
            raw = raw.decode(sys.getfilesystemencoding(), "surrogateescape")
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        try:
            absolute = Path(os.path.realpath(candidate))
            relative = absolute.relative_to(self.root)
        except (OSError, RuntimeError, ValueError):
            return None
        return relative.as_posix()

    def _repository_module_paths(self) -> set[str]:
        paths: set[str] = set()
        for module in sys.modules.values():
            path = self._relative_repository_path(
                getattr(module, "__file__", None)
            )
            if path is not None:
                if path.endswith((".pyc", ".pyo")):
                    _fail(f"repository module is sourceless bytecode: {path}")
                paths.add(path)
        return paths

    def install(self) -> None:
        if self._installed:
            _fail("runtime import recorder cannot be reinstalled")
        self._installed = True
        recorder = self

        def audit(event: str, arguments: tuple[Any, ...]) -> None:
            if event != "open" or not arguments:
                return
            path = recorder._relative_repository_path(arguments[0])
            if path is None:
                return
            if path.endswith((".pyc", ".pyo")) or "__pycache__" in PurePosixPath(
                path
            ).parts:
                _fail(
                    "repository bytecode-cache open is forbidden during "
                    f"import recording: {path}"
                )
            if path.endswith(".py") and path not in recorder._active_source_paths:
                _fail(
                    "repository source open bypassed the import recorder: "
                    f"{path}"
                )

        def source_exec(
            loader: importlib.machinery.SourceFileLoader,
            module: Any,
        ) -> Any:
            path = recorder._relative_repository_path(
                getattr(module, "__file__", None)
            )
            if path is None:
                return recorder._source_exec(loader, module)
            if recorder._reject_mode:
                _fail(f"late repository module execution is forbidden: {path}")
            if path in recorder.recorded:
                _fail(f"duplicate repository module execution: {path}")
            expected_digest = recorder.expected.get(path)
            if expected_digest is None:
                _fail(f"repository module is outside bound closure: {path}")
            absolute = recorder.root / path
            if absolute.is_symlink() or not absolute.is_file():
                _fail(f"repository module source is unsafe: {path}")
            recorder._active_source_paths[path] = (
                recorder._active_source_paths.get(path, 0) + 1
            )
            try:
                if _sha256_file(absolute) != expected_digest:
                    _fail(f"repository module source hash differs: {path}")
                result = recorder._source_exec(loader, module)
            finally:
                remaining = recorder._active_source_paths[path] - 1
                if remaining:
                    recorder._active_source_paths[path] = remaining
                else:
                    del recorder._active_source_paths[path]
            recorder.recorded.append(path)
            return result

        def sourceless_exec(
            loader: importlib.machinery.SourcelessFileLoader,
            module: Any,
        ) -> Any:
            path = recorder._relative_repository_path(
                getattr(module, "__file__", None)
            )
            if path is not None:
                _fail(
                    f"repository-local sourceless module execution is forbidden: {path}"
                )
            return recorder._sourceless_exec(loader, module)

        importlib.machinery.SourceFileLoader.get_code = _source_only_get_code
        importlib.machinery.SourceFileLoader.exec_module = source_exec
        importlib.machinery.SourcelessFileLoader.exec_module = sourceless_exec
        sys.addaudithook(audit)
        self.snapshot = set(sys.modules)
        self.preloaded_repository_paths = self._repository_module_paths()
        approved_preloaded = (
            self.protocol_paths | self.allowed_initializer_paths
        )
        for path in self.preloaded_repository_paths:
            if path not in approved_preloaded:
                _fail(f"unapproved preloaded repository module: {path}")
        for root_path in (
            "execution/gross9_rank7_clock_runtime.py",
            "training/gross9_structural_clock_primitives.py",
        ):
            if root_path in self.preloaded_repository_paths:
                _fail(f"isolated runtime root was preloaded: {root_path}")

    def reconcile(self) -> list[str]:
        current_paths = self._repository_module_paths()
        recorded = set(self.recorded)
        if len(recorded) != len(self.recorded):
            _fail("runtime import execution recorder contains duplicates")
        newly_loaded = current_paths - self.preloaded_repository_paths
        if newly_loaded != recorded:
            _fail(
                "new repository modules differ from successful execution recorder"
            )
        if not recorded.issubset(current_paths):
            _fail("recorded repository module was removed from sys.modules")
        return sorted(recorded)

    def freeze(self) -> list[str]:
        paths = self.reconcile()
        required_roots = {
            "execution/gross9_rank7_clock_runtime.py",
            "training/gross9_structural_clock_primitives.py",
        }
        if not required_roots.issubset(paths):
            _fail("isolated runtime roots were not both executed")
        self._reject_mode = True
        return paths


def _read_guarded_file(
    path: Path,
    *,
    expected_mode: int | None = None,
) -> tuple[bytes, os.stat_result]:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            _fail(f"guarded file is not regular: {path}")
        raw = bytearray()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            raw.extend(chunk)
    finally:
        os.close(descriptor)
    if len(raw) != info.st_size:
        _fail(f"guarded file changed during read: {path}")
    if expected_mode is not None and stat.S_IMODE(info.st_mode) != expected_mode:
        _fail(f"guarded file mode differs: {path}")
    return bytes(raw), info


def _sync_guarded_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive_guarded_file(
    path: Path,
    raw: bytes,
    *,
    mode: int,
    sync_directory: Path,
) -> os.stat_result:
    guard = _ACTIVE_WORKER_GUARD
    if (
        not isinstance(guard, _WorkerIsolationGuard)
        or guard.stage_fd is None
        or path.parent != guard.own_stage
        or sync_directory != guard.own_stage
    ):
        _fail("worker stage-file creation lacks the retained stage descriptor")
    baseline = guard.stage_entries
    guard.require_stage(baseline)
    if path.name in baseline:
        _fail(f"exclusive worker output already exists: {path}")
    descriptor = _openat_component(
        guard.stage_fd,
        path.name,
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        created = os.fstat(descriptor)
        guard.descriptors[descriptor] = (
            path.as_posix(),
            int(created.st_dev),
            int(created.st_ino),
            False,
        )
        expected_created = tuple(sorted((*baseline, path.name)))
        if _directory_entries(guard.stage_fd) != expected_created:
            _fail("worker stage-file create delta differs")
        guard.require_stage(expected_created, authorized_delta=True)
        _write_all(descriptor, raw)
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        info = os.fstat(descriptor)
        observed = _pread_complete(descriptor, info.st_size)
        if (
            observed != raw
            or (info.st_dev, info.st_ino)
            != (created.st_dev, created.st_ino)
            or stat.S_IMODE(info.st_mode) != mode
        ):
            _fail(f"exclusive worker output verification failed: {path}")
        os.fsync(guard.stage_fd)
        if _directory_entries(guard.stage_fd) != expected_created:
            _fail("worker stage-file stable inventory differs")
        guard.require_stage(expected_created)
    finally:
        os.close(descriptor)
    return info


def _worker_ledger_payload(
    *,
    binding: Mapping[str, Any],
    claim: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    sentinel: Mapping[str, Any],
    authority_amendments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = {
        "identity": IDENTITY,
        "protocol_version": PROTOCOL_VERSION,
        "slot": binding["slot"],
        "parent_pid": binding["parent_pid"],
        "stage_directory": binding["stage_directory"],
        "carrier_kind": "anonymous_pipe_v1",
        "carrier_device": binding["carrier_device"],
        "carrier_inode": binding["carrier_inode"],
        "token_sha256": binding["token_sha256"],
        "claim_hash": claim["claim_hash"],
        "preregistration_manifest_hash": preregistration["manifest_hash"],
        "sentinel_manifest_hash": sentinel["manifest_hash"],
        "authority_amendments": [dict(row) for row in authority_amendments],
        "status": "consumed_before_runtime_or_value_access",
    }
    if len(payload) != 14:
        _fail("worker consumption ledger schema differs")
    return payload


def _worker_metadata_final_recheck(
    guard: _WorkerIsolationGuard,
    snapshot: _SecureBoundSnapshot,
    binding: Mapping[str, Any],
    ledger_fd: int,
) -> None:
    if (
        guard.repository_fd is None
        or guard.results_fd is None
        or guard.filesystem_root_fd is None
    ):
        _fail("worker metadata anchors are absent at final recheck")
    snapshot.verify_final()
    for descriptor in (
        guard.repository_fd,
        guard.results_fd,
        guard.filesystem_root_fd,
    ):
        info = os.fstat(descriptor)
        record = guard.descriptors.get(descriptor)
        if (
            record is None
            or not stat.S_ISDIR(info.st_mode)
            or (info.st_dev, info.st_ino) != (record[1], record[2])
        ):
            _fail("worker metadata anchor final recheck differs")
    ledger_info = os.fstat(ledger_fd)
    if (
        (ledger_info.st_dev, ledger_info.st_ino)
        != (binding["ledger_device"], binding["ledger_inode"])
        or stat.S_IMODE(ledger_info.st_mode) != 0o444
    ):
        _fail("worker metadata ledger final recheck differs")
    if guard.stage_fd is None:
        _fail("worker metadata stage descriptor is absent")
    guard.require_stage(())
    guard.require_results()
    leaf = Path(str(binding["consumed_ledger_path"])).name
    try:
        guard._original_stat(
            leaf, dir_fd=guard.results_fd, follow_symlinks=False
        )
    except FileNotFoundError:
        pass
    else:
        _fail("worker canonical ledger appeared before final recheck")
    if guard.counters() != {
        "child_process_creation_events": 0,
        "other_stage_access_events": 0,
        "other_stage_absence_checks": 1,
        "other_slot_ledger_access_events": 0,
        "unauthorized_write_or_ipc_events": 0,
    }:
        _fail("worker metadata isolation counters differ at final recheck")


def _publish_worker_ledger(
    *,
    guard: _WorkerIsolationGuard,
    snapshot: _SecureBoundSnapshot,
    ledger_fd: int,
    binding: Mapping[str, Any],
    claim: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    sentinel: Mapping[str, Any],
    authority_amendments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if guard.results_fd is None or guard.ledger_fd != ledger_fd:
        _fail("worker ledger publication descriptors are not registered")
    canonical = _rooted(
        guard.root, Path(str(binding["consumed_ledger_path"]))
    )
    leaf = canonical.name
    baseline_entries = guard.results_entries
    guard.require_results()
    try:
        guard._original_stat(
            leaf, dir_fd=guard.results_fd, follow_symlinks=False
        )
    except FileNotFoundError:
        pass
    else:
        _fail("worker canonical ledger already exists")
    payload = _worker_ledger_payload(
        binding=binding,
        claim=claim,
        preregistration=preregistration,
        sentinel=sentinel,
        authority_amendments=authority_amendments,
    )
    raw = _canonical_json_bytes(payload)
    guard.allowed_mutations.add("<unnamed-worker-ledger>")
    _write_all(ledger_fd, raw)
    os.fchmod(ledger_fd, 0o444)
    os.fsync(ledger_fd)
    staged_info = os.fstat(ledger_fd)
    if (
        (staged_info.st_dev, staged_info.st_ino)
        != (binding["ledger_device"], binding["ledger_inode"])
        or not stat.S_ISREG(staged_info.st_mode)
        or stat.S_IMODE(staged_info.st_mode) != 0o444
        or staged_info.st_size != len(raw)
        or _pread_complete(ledger_fd, staged_info.st_size) != raw
    ):
        _fail("worker unnamed ledger same-FD verification differs")
    _worker_metadata_final_recheck(guard, snapshot, binding, ledger_fd)
    # Sole guarded procfd/link exception, prebound to this slot and destination.
    _link_unnamed_procfd(ledger_fd, guard.results_fd, leaf)
    canonical_fd = guard.open_owned_canonical_ledger_once(
        leaf,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        dir_fd=guard.results_fd,
    )
    try:
        canonical_info = os.fstat(canonical_fd)
        canonical_raw = _pread_complete(
            canonical_fd, canonical_info.st_size
        )
        if (
            canonical_fd == ledger_fd
            or canonical_raw != raw
            or _sha256_bytes(canonical_raw) != _sha256_bytes(raw)
            or not stat.S_ISREG(canonical_info.st_mode)
            or stat.S_IMODE(canonical_info.st_mode) != 0o444
            or canonical_info.st_size != len(raw)
            or (canonical_info.st_dev, canonical_info.st_ino)
            != (staged_info.st_dev, staged_info.st_ino)
        ):
            _fail("worker consumption ledger publication differs")
        expected_entries = tuple(sorted((*baseline_entries, leaf)))
        if _directory_entries(guard.results_fd) != expected_entries:
            _fail("worker ledger one-leaf results delta differs")
        os.fsync(guard.results_fd)
        guard.results_entries = expected_entries
        guard.results_timestamp_token = _descriptor_token(
            os.fstat(guard.results_fd)
        )
        guard.require_results()
        return {
            "payload": payload,
            "raw": raw,
            "sha256": _sha256_bytes(raw),
            "device": int(canonical_info.st_dev),
            "inode": int(canonical_info.st_ino),
            "canonical_fd": canonical_fd,
        }
    except BaseException:
        os.close(canonical_fd)
        raise


def _worker_receipt_payload(
    *,
    binding: Mapping[str, Any],
    worker_pid: int,
    ledger_sha256: str,
    rebuild_invocations_started: int,
    rebuild_invocations_completed: int,
    guard_counters: Mapping[str, int],
    csv_gzip_sha256: str,
    per_pass_core_sha256: str,
    token: bytearray,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "identity": IDENTITY,
        "protocol_version": PROTOCOL_VERSION,
        "slot": binding["slot"],
        "parent_pid": binding["parent_pid"],
        "worker_pid": worker_pid,
        "stage_directory": binding["stage_directory"],
        "consumed_ledger_path": binding["consumed_ledger_path"],
        "consumed_ledger_sha256": ledger_sha256,
        "rebuild_invocations_started": rebuild_invocations_started,
        "rebuild_invocations_completed": rebuild_invocations_completed,
        "child_process_creation_events": guard_counters[
            "child_process_creation_events"
        ],
        "other_stage_access_events": guard_counters[
            "other_stage_access_events"
        ],
        "other_stage_absence_checks": guard_counters[
            "other_stage_absence_checks"
        ],
        "other_slot_ledger_access_events": guard_counters[
            "other_slot_ledger_access_events"
        ],
        "unauthorized_write_or_ipc_events": guard_counters[
            "unauthorized_write_or_ipc_events"
        ],
        "csv_gzip_sha256": csv_gzip_sha256,
        "per_pass_core_sha256": per_pass_core_sha256,
    }
    if list(payload.values())[8:15] != [1, 1, 0, 0, 1, 0, 0]:
        _fail("worker rebuild/isolation event counters differ")
    completion = _canonical_json_bytes(payload, trailing_lf=False)
    payload["completion_hmac_sha256"] = hmac.new(
        token,
        completion,
        hashlib.sha256,
    ).hexdigest()
    payload["receipt_hash"] = _sha256_bytes(
        _canonical_json_bytes(payload, trailing_lf=False)
    )
    if len(payload) != 19:
        _fail("worker pass receipt schema differs")
    return payload


def _worker_ledger_linked_checkpoint(
    guard: _WorkerIsolationGuard,
    binding: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> None:
    guard.require_results()
    guard.require_stage(())
    canonical_fd = ledger.get("canonical_fd")
    if not isinstance(canonical_fd, int):
        _fail("worker ledger-linked checkpoint lacks canonical descriptor")
    if (
        guard._owned_ledger_open_state != "consumed"
        or guard._owned_ledger_open_count != 1
    ):
        _fail("worker ledger canonical-open state differs")
    info = os.fstat(canonical_fd)
    leaf = Path(str(binding["consumed_ledger_path"])).name
    edge = guard._original_stat(
        leaf, dir_fd=guard.results_fd, follow_symlinks=False
    )
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o444
        or (info.st_dev, info.st_ino)
        != (int(ledger["device"]), int(ledger["inode"]))
        or _descriptor_token(edge) != _descriptor_token(info)
        or info.st_size != len(ledger["raw"])
    ):
        _fail("worker ledger-linked checkpoint differs")


def _parse_timestamp(value: str) -> int:
    if not isinstance(value, str) or not _TIMESTAMP_RE.fullmatch(value):
        _fail(f"invalid UTC-second timestamp: {value!r}")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise TerminalG9CB12Failure(f"invalid timestamp: {value}") from exc
    seconds = int(parsed.timestamp())
    if datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        _fail(f"noncanonical timestamp: {value}")
    if seconds % 300:
        _fail(f"timestamp is off the five-minute grid: {value}")
    return seconds


def _timestamp(seconds: int) -> str:
    return datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _decimal(value: Any, field: str) -> Decimal:
    if isinstance(value, bool) or isinstance(value, float):
        _fail(f"{field} must be an exact decimal string or integer")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise TerminalG9CB12Failure(f"invalid decimal {field}") from exc
    if not result.is_finite():
        _fail(f"nonfinite decimal {field}")
    return result


def _empty_counters() -> dict[str, Any]:
    logical_sources = (
        "market_5m",
        "funding",
        "premium",
        "open_interest",
        "rank7_spot_premium_5m",
        "rex_taker_train",
        "rex_taker_test",
        "rex_taker_eval",
        "rex_veto_source",
        "rank7_hourly_history",
    )
    return {
        "file_access": {
            "bytes_read_by_logical_source": {
                name: 0 for name in logical_sources
            },
            "source_files_opened": 0,
            "model_files_opened": 0,
            "runtime_modules_imported": 0,
        },
        "rows_decoded": {
            name: 0 for name in logical_sources
        },
        "rows_used": {
            "causal_feature_rows_by_source": {
                name: 0 for name in logical_sources
            },
            "prediction_rows_scored": 0,
            "outcome_dependent_ohlc_rows_examined": 0,
            **{name: 0 for name in RANK7_ROWS_USED_COUNTERS},
        },
        "per_sleeve": {
            row["name"]: {
                "signal_rows_evaluated": 0,
                "intervals_emitted": 0,
                "long_intervals": 0,
                "short_intervals": 0,
                "fixed_horizon_exits": 0,
                "take_exits": 0,
                "stop_exits": 0,
                "outcome_dependent_ohlc_rows_examined": 0,
            }
            for row in SLEEVES
        },
    }


def _barrier_exit(
    bars: Sequence[Mapping[str, Any]],
    entry_index: int,
    side: int,
    *,
    hold_bars: int,
    take_bps: int | None,
    stop_bps: int | None,
    counters: dict[str, Any],
    sleeve: str,
) -> tuple[int, str]:
    entry = _decimal(bars[entry_index]["open"], "entry open")
    take = None
    stop = None
    if take_bps is not None:
        take = entry * (Decimal(1) + Decimal(side * take_bps) / Decimal(10000))
    if stop_bps is not None:
        stop = entry * (Decimal(1) - Decimal(side * stop_bps) / Decimal(10000))
    horizon = entry_index + hold_bars
    for index in range(entry_index, horizon):
        high = _decimal(bars[index]["high"], "bar high")
        low = _decimal(bars[index]["low"], "bar low")
        counters["rows_used"]["outcome_dependent_ohlc_rows_examined"] += 1
        counters["per_sleeve"][sleeve]["outcome_dependent_ohlc_rows_examined"] += 1
        stop_hit = stop is not None and (low <= stop if side == 1 else high >= stop)
        take_hit = take is not None and (high >= take if side == 1 else low <= take)
        if stop_hit:
            return index, "stop"
        if take_hit:
            return index, "take"
    return horizon, "fixed"


def reconstruct_intervals(
    bars: Sequence[Mapping[str, Any]],
    *,
    domain_start: str = DOMAIN_START,
    domain_end: str = DOMAIN_END,
    counters: dict[str, Any] | None = None,
    split_bounds: Sequence[tuple[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reconstruct structural intervals from causal generic-runtime decisions.

    ``bars`` is the narrow worker boundary: each row is one authenticated
    five-minute market row and may carry a ``decisions`` object produced from
    the authenticated generic runtime/configuration.  Tests use synthetic rows;
    production rows are created only inside a post-sentinel worker.
    """

    audit = counters if counters is not None else _empty_counters()
    start = _parse_timestamp(domain_start)
    end = _parse_timestamp(domain_end)
    if start >= end:
        _fail("empty domain")
    normalized: list[tuple[int, Mapping[str, Any]]] = []
    previous: int | None = None
    for row in bars:
        if not isinstance(row, Mapping):
            _fail("market row is not an object")
        second = _parse_timestamp(str(row.get("time_utc")))
        if previous is not None and second != previous + 300:
            _fail("market rows are not a complete unique causal grid")
        previous = second
        normalized.append((second, row))
    value_opens = [second for second, _ in normalized]
    if not value_opens:
        _fail("market rows are empty")
    if any(second % 300 for second in value_opens):
        _fail("market row is off the five-minute grid")
    if value_opens[0] > start or start not in value_opens:
        _fail("market rows lack the aligned domain start")
    if any(second >= end for second in value_opens):
        _fail("market rows contain an end-boundary value")
    if value_opens[-1] + 300 != end:
        _fail("market rows end before the terminal boundary")
    boundaries = [*value_opens, end]
    if len(boundaries) != len(value_opens) + 1:
        _fail("market boundary vector length differs")
    splits: list[tuple[int, int, list[bool]]] = []
    if split_bounds is None:
        splits.append((start, end + 300, [True] * len(value_opens)))
    else:
        for raw_split_start, raw_split_end in split_bounds:
            split_start = _parse_timestamp(raw_split_start)
            split_end = _parse_timestamp(raw_split_end)
            if not (start <= split_start < split_end):
                _fail("reference split lies outside the market domain")
            splits.append(
                (
                    split_start,
                    split_end,
                    [
                        split_start <= second < split_end
                        for second in value_opens
                    ],
                )
            )

    output: list[dict[str, Any]] = []
    for sleeve in SLEEVES:
        name = sleeve["name"]
        index_in_sleeve = 0
        for split_start, split_end, mask in splits:
            last_exit = split_start
            for signal_index, (_signal_time, row) in enumerate(normalized):
                if not mask[signal_index]:
                    continue
                decisions = row.get("decisions", {})
                if not isinstance(decisions, Mapping):
                    _fail("decisions must be an object")
                decision = decisions.get(name)
                if decision is None:
                    continue
                audit["per_sleeve"][name]["signal_rows_evaluated"] += 1
                if not isinstance(decision, Mapping):
                    _fail(f"invalid decision for {name}")
                active = decision.get("active")
                side = decision.get("side")
                if active is not True:
                    if active not in (False, None):
                        _fail(f"nonboolean active flag for {name}")
                    continue
                if isinstance(side, bool) or side not in sleeve["sides"]:
                    _fail(f"forbidden side for {name}: {side!r}")
                if name == "fresh_kimchi_fx":
                    long_gate = decision.get("long_gate")
                    short_gate = decision.get("short_gate")
                    if long_gate not in (True, False) or short_gate not in (
                        True,
                        False,
                    ):
                        _fail("fresh decision lacks exact gate booleans")
                    if long_gate == short_gate:
                        _fail("fresh active decision violates exclusive gates")
                    expected_side = 1 if long_gate else -1
                    if side != expected_side:
                        _fail("fresh side differs from exclusive gate")
                entry_index = signal_index + 1
                if entry_index >= len(normalized) or not mask[entry_index]:
                    continue
                entry_time = normalized[entry_index][0]
                if entry_time < last_exit:
                    continue
                kind = sleeve["kind"]
                if kind == "rank7":
                    source = decision.get("source")
                    if source == "funding":
                        hold, take_bps, stop_bps = 576, 400, None
                    elif source == "premium":
                        hold, take_bps, stop_bps = 144, None, 300
                    else:
                        _fail("Rank7 active decision has invalid source")
                elif kind == "barrier":
                    hold = int(sleeve["hold_bars"])
                    take_bps = int(sleeve["take_bps"])
                    stop_bps = int(sleeve["stop_bps"])
                else:
                    hold = int(sleeve["hold_bars"])
                    take_bps = stop_bps = None
                exit_position = entry_index + hold
                if exit_position > len(normalized):
                    continue
                if kind in ("rank7", "barrier"):
                    exit_position, exit_kind = _barrier_exit(
                        [item for _, item in normalized],
                        entry_index,
                        int(side),
                        hold_bars=hold,
                        take_bps=take_bps,
                        stop_bps=stop_bps,
                        counters=audit,
                        sleeve=name,
                    )
                else:
                    exit_kind = "fixed"
                if not _structural_exit_is_eligible(
                    mask,
                    boundaries,
                    exit_position,
                    exit_kind,
                    split_start,
                    split_end,
                ):
                    continue
                boundary_position = (
                    exit_position + 1
                    if exit_kind in ("take", "stop")
                    else exit_position
                )
                exit_time = boundaries[boundary_position]
                if not (
                    split_start
                    <= entry_time
                    < exit_time
                    <= min(split_end, end)
                ):
                    _fail(f"interval outside split for {name}")
                output.append(
                    {
                        "identity": IDENTITY,
                        "sleeve": name,
                        "sleeve_order": int(sleeve["order"]),
                        "configured_weight": sleeve["weight"],
                        "interval_index": index_in_sleeve,
                        "entry_time_utc": _timestamp(entry_time),
                        "exit_time_utc": _timestamp(exit_time),
                        "side": int(side),
                    }
                )
                last_exit = exit_time
                index_in_sleeve += 1
                per = audit["per_sleeve"][name]
                per["intervals_emitted"] += 1
                per["long_intervals" if side == 1 else "short_intervals"] += 1
                per[
                    {
                        "fixed": "fixed_horizon_exits",
                        "take": "take_exits",
                        "stop": "stop_exits",
                    }[exit_kind]
                ] += 1
    output.sort(key=lambda row: (row["sleeve_order"], row["interval_index"]))
    return output, audit


def serialize_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=CSV_COLUMNS,
        delimiter=",",
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row[key] for key in CSV_COLUMNS})
    raw = stream.getvalue().encode("utf-8")
    if raw.startswith(b"\xef\xbb\xbf") or not raw.endswith(b"\n") or b"\r" in raw:
        _fail("CSV serialization contract failure")
    return raw


def compress_csv(csv_bytes: bytes) -> bytes:
    stream = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        compresslevel=9,
        fileobj=stream,
        mtime=0,
    ) as handle:
        handle.write(csv_bytes)
    raw = bytearray(stream.getvalue())
    if len(raw) < 10:
        _fail("gzip output is truncated")
    raw[9] = 255
    result = bytes(raw)
    if not result.startswith(GZIP_PREFIX):
        _fail("gzip header contract failure")
    return result


def validate_csv_gzip(raw: bytes, *, require_all_sleeves: bool = True) -> list[dict[str, Any]]:
    if not raw.startswith(GZIP_PREFIX):
        _fail("gzip prefix mismatch")
    try:
        decompressed = gzip.decompress(raw)
    except (OSError, EOFError) as exc:
        raise TerminalG9CB12Failure("invalid gzip stream") from exc
    if compress_csv(decompressed) != raw:
        _fail("gzip bytes are not canonical")
    try:
        text = decompressed.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TerminalG9CB12Failure("CSV is not UTF-8") from exc
    if "\r" in text or not text.endswith("\n") or "\n\n" in text:
        _fail("CSV line-ending contract failure")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
        _fail("CSV schema mismatch")
    rows: list[dict[str, Any]] = []
    prior: dict[str, tuple[int, int]] = {}
    counts = {row["name"]: 0 for row in SLEEVES}
    for raw_row in reader:
        if set(raw_row) != set(CSV_COLUMNS) or None in raw_row.values():
            _fail("CSV row shape mismatch")
        name = raw_row["sleeve"]
        sleeve = SLEEVE_BY_NAME.get(name)
        if sleeve is None:
            _fail("unknown sleeve")
        side_text = raw_row["side"]
        if side_text not in ("1", "-1"):
            _fail("forbidden CSV side")
        entry = _parse_timestamp(raw_row["entry_time_utc"])
        exit_ = _parse_timestamp(raw_row["exit_time_utc"])
        index = int(raw_row["interval_index"])
        if (
            raw_row["identity"] != IDENTITY
            or raw_row["sleeve_order"] != str(sleeve["order"])
            or raw_row["configured_weight"] != sleeve["weight"]
            or index != counts[name]
            or int(side_text) not in sleeve["sides"]
            or not (_parse_timestamp(DOMAIN_START) <= entry < exit_ <= _parse_timestamp(DOMAIN_END))
        ):
            _fail("CSV frozen field contract mismatch")
        if name in prior and (entry < prior[name][1] or entry <= prior[name][0]):
            _fail("CSV per-sleeve order or non-overlap mismatch")
        prior[name] = (entry, exit_)
        counts[name] += 1
        rows.append(dict(raw_row))
    expected_order = sorted(
        rows,
        key=lambda row: (
            int(row["sleeve_order"]),
            row["entry_time_utc"],
            row["exit_time_utc"],
            int(row["side"]),
        ),
    )
    if rows != expected_order:
        _fail("CSV canonical ordering disagreement")
    if require_all_sleeves and any(count == 0 for count in counts.values()):
        _fail("one or more required sleeves are empty")
    return rows


def _prohibited_assertions() -> dict[str, Any]:
    return {
        "pre2025_anchor_value_rows_opened": 0,
        "candidate_rows_opened": 0,
        "comparator_clock_rows_opened": 0,
        "portfolio_return_values_computed": 0,
        "portfolio_pnl_values_computed": 0,
        "funding_cash_values_computed": 0,
        "cagr_values_computed": 0,
        "mdd_values_computed": 0,
        "economic_rank_values_computed": 0,
        "candidate_metric_values_computed": 0,
        "overlap_metric_values_computed": 0,
    }


def _validate_prohibited_output_placement(payload: Mapping[str, Any]) -> None:
    """Permit prohibited-computation keys only in the canonical zero assertion."""

    expected = _prohibited_assertions()
    allowed_path = ("evidence_boundary", "prohibited_output_counters")
    found = False

    def walk(value: Any, path: tuple[str, ...]) -> None:
        nonlocal found
        if path == allowed_path:
            if (
                not isinstance(value, Mapping)
                or set(value) != set(expected)
                or any(type(item) is not int or item != 0 for item in value.values())
            ):
                _fail("prohibited-output assertion differs from the zero contract")
            found = True
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                key_text = str(key)
                if key_text in expected:
                    _fail(
                        "prohibited-output key appears outside the canonical "
                        f"zero assertion: {'.'.join(path + (key_text,))}"
                    )
                walk(item, path + (key_text,))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, path + (str(index),))

    walk(payload, ())
    if not found:
        _fail("canonical prohibited-output zero assertion is absent")


def build_core(
    csv_gzip: bytes,
    counters: Mapping[str, Any],
    provenance: Mapping[str, Any],
    claim_binding: Mapping[str, Any],
    sentinel_binding: Mapping[str, Any],
    authority_amendments: Sequence[Mapping[str, Any]],
    parent_authentication: Mapping[str, Any] | None = None,
    *,
    expected_authority_amendments: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    csv_bytes = gzip.decompress(csv_gzip)
    amendment_rows = [dict(row) for row in authority_amendments]
    expected_amendments = (
        _expected_authority_amendment_bindings()
        if expected_authority_amendments is None
        else [dict(row) for row in expected_authority_amendments]
    )
    if amendment_rows != expected_amendments:
        _fail("core authority amendment bindings mismatch")
    parent_authentication_object = dict(parent_authentication or {})
    parent_authentication_sha256 = _sha256_bytes(
        _canonical_json_bytes(
            parent_authentication_object,
            trailing_lf=False,
        )
    )
    core = _with_hash(
        {
            "identity": IDENTITY,
            "protocol_version": PROTOCOL_VERSION,
            "domain": {"start": DOMAIN_START, "end_exclusive": DOMAIN_END},
            "csv_schema": list(CSV_COLUMNS),
            "sleeves": [
                {**row, "sides": list(row["sides"])}
                for row in SLEEVES
            ],
            "authority_amendments": amendment_rows,
            "provenance": dict(provenance),
            "parent_authentication": parent_authentication_object,
            "parent_authentication_sha256": parent_authentication_sha256,
            "access_claim": dict(claim_binding),
            "attempt_consumed": dict(sentinel_binding),
            "access_counters": dict(counters),
            "rank7_reconstruction": {
                "bundle_parity_status": (
                    "passed"
                    if counters["rows_used"][
                        "rank7_bundle_parity_rows_compared"
                    ]
                    > 0
                    else "not_applicable"
                ),
                "bundle_parity_rows_compared": counters["rows_used"][
                    "rank7_bundle_parity_rows_compared"
                ],
                "model_files_opened": counters["file_access"][
                    "model_files_opened"
                ],
                "hourly_history_rows_decoded": counters["rows_decoded"][
                    "rank7_hourly_history"
                ],
            },
            "evidence_boundary": {
                "candidate_identity_present": False,
                "candidate_artifacts_opened": False,
                "comparator_clocks_preseen_by_research_program": True,
                "prohibited_output_counters": _prohibited_assertions(),
            },
            "csv_byte_length": len(csv_bytes),
            "csv_sha256": _sha256_bytes(csv_bytes),
            "csv_gzip_byte_length": len(csv_gzip),
            "csv_gzip_sha256": _sha256_bytes(csv_gzip),
        },
        "manifest_hash",
    )
    _validate_prohibited_output_placement(core)
    return core


def _read_synthetic_worker_input(
    path: Path,
    *,
    path_text: str,
    raw_cache: Mapping[str, tuple[bytes, os.stat_result]],
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    payload, _, _ = _read_canonical_object(
        path,
        path_text=path_text,
        raw_cache=raw_cache,
    )
    bars = payload.get("bars")
    if not isinstance(bars, list):
        _fail("synthetic worker input has no bars")
    counters = _empty_counters()
    supplied = payload.get("physical_counters")
    if isinstance(supplied, Mapping):
        for section in ("file_access", "rows_decoded", "rows_used"):
            if isinstance(supplied.get(section), Mapping):
                counters[section].update(supplied[section])
    domain_end = payload.get("domain_end", DOMAIN_END)
    if not isinstance(domain_end, str):
        _fail("synthetic worker domain end is invalid")
    return [dict(row) for row in bars], counters, domain_end


def _import_authenticated_modules(repository_root: str) -> dict[str, Any]:
    """Import exactly the preregistered generic roots after the sentinel."""

    imported: dict[str, Any] = {}
    for name in GENERIC_RUNTIME_MODULES:
        module = importlib.import_module(name)
        imported[name] = module
        expected = (
            Path(repository_root) / f"{name.replace('.', '/')}.py"
        ).resolve()
        observed_file = getattr(module, "__file__", None)
        if (
            not isinstance(observed_file, str)
            or Path(observed_file).resolve() != expected
        ):
            _fail(f"generic import resolved outside authenticated root: {name}")
    return imported


def _authority_path(
    preregistration: Mapping[str, Any], name: str
) -> str:
    bindings = preregistration.get("bindings")
    inventory = (
        bindings.get("direct_authority") if isinstance(bindings, Mapping) else None
    )
    if not isinstance(inventory, list):
        _fail("direct authority inventory is absent")
    matches = [
        row
        for row in inventory
        if isinstance(row, Mapping) and row.get("name") == name
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("path"), str):
        _fail(f"direct authority binding is absent: {name}")
    return str(matches[0]["path"])


def _source_paths(
    preregistration: Mapping[str, Any],
) -> dict[str, str]:
    bindings = preregistration.get("bindings")
    inventory = (
        bindings.get("source_manifest_ordered_inventory")
        if isinstance(bindings, Mapping)
        else None
    )
    if not isinstance(inventory, list):
        _fail("source inventory is absent")
    result: dict[str, str] = {}
    for row in inventory:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("name"), str)
            or not isinstance(row.get("logical_path"), str)
        ):
            _fail("invalid source inventory row")
        result[str(row["name"])] = str(row["logical_path"])
    required = {
        "market_5m",
        "funding",
        "premium",
        "open_interest",
        "rank7_spot_premium_5m",
        "rex_taker_train",
        "rex_taker_test",
        "rex_taker_eval",
        "rex_veto_source",
    }
    if set(result) != required:
        _fail("source inventory names differ")
    return result


def _rank7_declared_bundle_paths(
    preregistration: Mapping[str, Any],
) -> tuple[str, tuple[str, ...]]:
    bindings = preregistration.get("bindings")
    bundle = bindings.get("rank7_bundle") if isinstance(bindings, Mapping) else None
    declared = bundle.get("declared_files") if isinstance(bundle, Mapping) else None
    if not isinstance(declared, list):
        _fail("Rank7 declared-file inventory is absent")
    paths = [
        str(row.get("path"))
        for row in declared
        if isinstance(row, Mapping) and isinstance(row.get("path"), str)
    ]
    history = [path for path in paths if path.endswith(".csv.gz")]
    models = [path for path in paths if path.endswith(".npz")]
    if (
        len(paths) != 6
        or len(history) != 1
        or len(models) != 5
        or len(paths) != len(set(paths))
    ):
        _fail("Rank7 declared-file inventory shape differs")
    return history[0], tuple(models)


def _install_counted_rank7_runtime(
    rank7_runtime: Any,
    root: Path,
    preregistration: Mapping[str, Any],
    counters: dict[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    """Instrument actual Rank7 model opens and portable-model predictions."""

    _history_path, model_paths = _rank7_declared_bundle_paths(preregistration)
    allowed: dict[str, str] = {}
    for logical_path in model_paths:
        candidate = Path(logical_path)
        candidate = candidate if candidate.is_absolute() else root / candidate
        resolved = os.fspath(candidate.resolve(strict=False))
        if resolved in allowed:
            _fail("Rank7 declared model paths alias one another")
        allowed[resolved] = logical_path

    numpy_module = getattr(rank7_runtime, "np", None)
    model_type = getattr(rank7_runtime, "FrozenExtraTreesModel", None)
    original_load = getattr(numpy_module, "load", None)
    original_predict = getattr(model_type, "predict", None)
    if not callable(original_load) or not callable(original_predict):
        _fail("Rank7 runtime lacks instrumentable model operations")
    state: dict[str, Any] = {
        "allowed": allowed,
        "opened": [],
    }

    def counted_load(path: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            candidate = Path(os.fspath(path))
        except TypeError as exc:
            raise TerminalG9CB12Failure(
                "Rank7 model open did not use a filesystem path"
            ) from exc
        candidate = candidate if candidate.is_absolute() else root / candidate
        resolved = os.fspath(candidate.resolve(strict=False))
        logical_path = allowed.get(resolved)
        if logical_path is None:
            _fail(f"Rank7 model open is outside the declared bundle: {path}")
        if logical_path in state["opened"]:
            _fail(f"Rank7 model was opened more than once: {logical_path}")
        result = original_load(path, *args, **kwargs)
        state["opened"].append(logical_path)
        counters["file_access"]["model_files_opened"] += 1
        return result

    def counted_predict(model: Any, matrix: Any) -> Any:
        values = numpy_module.asarray(matrix)
        if values.ndim == 1:
            rows = 1
        elif values.ndim == 2:
            rows = int(values.shape[0])
        else:
            rows = 0
        result = original_predict(model, matrix)
        counters["rows_used"]["prediction_rows_scored"] += rows
        return result

    numpy_module.load = counted_load
    try:
        model_type.predict = counted_predict
    except BaseException:
        numpy_module.load = original_load
        raise
    return original_load, original_predict, state


def _restore_counted_rank7_runtime(
    rank7_runtime: Any,
    original_load: Any,
    original_predict: Any,
) -> None:
    rank7_runtime.np.load = original_load
    rank7_runtime.FrozenExtraTreesModel.predict = original_predict


def _validate_counted_rank7_runtime(state: Mapping[str, Any]) -> None:
    allowed = state.get("allowed")
    opened = state.get("opened")
    if not isinstance(allowed, Mapping) or not isinstance(opened, list):
        _fail("Rank7 runtime operation state is invalid")
    if len(opened) != 5 or set(opened) != set(allowed.values()):
        _fail("Rank7 runtime did not open each declared model exactly once")


def _load_worker_json(root: Path, path: str) -> dict[str, Any]:
    candidate = Path(path)
    candidate = candidate if candidate.is_absolute() else root / candidate
    if candidate.is_symlink() or not candidate.is_file():
        _fail(f"worker JSON input is absent or unsafe: {path}")
    try:
        value = json.loads(
            candidate.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TerminalG9CB12Failure(f"worker JSON input is invalid: {path}") from exc
    if not isinstance(value, dict):
        _fail(f"worker JSON input is not an object: {path}")
    return value


def _market_seconds(market: Any) -> tuple[list[int], list[Any]]:
    try:
        values = list(market["date"])
    except (KeyError, TypeError) as exc:
        raise TerminalG9CB12Failure("generic market lacks date rows") from exc
    seconds: list[int] = []
    for value in values:
        if hasattr(value, "to_pydatetime"):
            value = value.to_pydatetime()
        if isinstance(value, datetime):
            parsed = value
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            else:
                parsed = parsed.astimezone(timezone.utc)
            text = parsed.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            text = str(value).replace(" ", "T")
            if text.endswith("+00:00"):
                text = text[:-6] + "Z"
            elif not text.endswith("Z"):
                text += "Z"
        seconds.append(_parse_timestamp(text))
    if any(right != left + 300 for left, right in zip(seconds, seconds[1:])):
        _fail("generic market is not a complete unique five-minute grid")
    return seconds, values


def _market_value_opens_and_boundaries(
    market: Any,
    *,
    domain_start: str = DOMAIN_START,
    domain_end: str = DOMAIN_END,
) -> tuple[list[int], list[int]]:
    """Validate physical value opens and derive the geometry-only boundary."""

    value_opens, _ = _market_seconds(market)
    start = _parse_timestamp(domain_start)
    end = _parse_timestamp(domain_end)
    if not value_opens:
        _fail("generic market has no value opens")
    if any(second % 300 for second in value_opens):
        _fail("generic market value open is off the five-minute grid")
    if value_opens[0] > start or start not in value_opens:
        _fail("generic market lacks the aligned domain-start value open")
    if any(second >= end for second in value_opens):
        _fail("generic market contains a domain-end value row")
    if value_opens[-1] + 300 != end:
        _fail("generic market ends before the canonical terminal boundary")
    boundaries = [*value_opens, end]
    if len(boundaries) != len(value_opens) + 1:
        _fail("generic market boundary vector length differs")
    return value_opens, boundaries


def _structural_exit_is_eligible(
    mask: Sequence[Any],
    boundaries: Sequence[int],
    exit_position: int,
    exit_kind: str,
    split_start: int,
    split_end: int,
) -> bool:
    """Apply physical-row eligibility and the sole virtual terminal rule."""

    n = len(mask)
    if len(boundaries) != n + 1:
        _fail("split mask and boundary vector lengths differ")
    position = int(exit_position)
    if position < 0 or position > n:
        _fail("structural exit index is outside the boundary vector")
    if exit_kind == "fixed":
        if position < n:
            return bool(mask[position])
        return split_start <= int(boundaries[n]) < split_end
    if position >= n:
        _fail("barrier hit index lacks an occupied value row")
    return bool(mask[position])


def _generic_time_second(value: Any, pandas_module: Any) -> int:
    try:
        parsed = pandas_module.Timestamp(
            pandas_module.to_datetime(value, utc=True, errors="raise")
        )
    except (TypeError, ValueError) as exc:
        raise TerminalG9CB12Failure("generic source timestamp is invalid") from exc
    return int(parsed.timestamp())


def _read_jsonl_rows(
    root: Path,
    path: str,
    counters: dict[str, Any],
    logical_name: str,
) -> list[dict[str, Any]]:
    candidate = Path(path)
    candidate = candidate if candidate.is_absolute() else root / candidate
    if candidate.is_symlink() or not candidate.is_file():
        _fail(f"JSONL source is absent or unsafe: {path}")
    descriptor = os.open(
        candidate,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    counters["file_access"]["source_files_opened"] += 1
    chunks: list[bytes] = []
    try:
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            counters["file_access"]["bytes_read_by_logical_source"][
                logical_name
            ] += len(chunk)
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line:
            continue
        try:
            row = json.loads(line.decode("utf-8"), object_pairs_hook=_unique_object)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TerminalG9CB12Failure(f"invalid JSONL row: {path}") from exc
        if not isinstance(row, dict):
            _fail(f"JSONL row is not an object: {path}")
        if "_g9cb_parser_ordinal" in row:
            _fail(f"JSONL row collides with parser ordinal key: {path}")
        row["_g9cb_parser_ordinal"] = counters["rows_decoded"][
            logical_name
        ]
        rows.append(row)
        counters["rows_decoded"][logical_name] += 1
    return rows


def _ordered_rex_source_rows(
    decoded: Sequence[dict[str, Any]],
    sleeve_name: str,
) -> list[dict[str, Any]]:
    source_keys = [
        (int(row.get("signal_pos", -1)), str(row.get("date")))
        for row in decoded
    ]
    if (
        len(source_keys) != len(set(source_keys))
        or len({position for position, _ in source_keys}) != len(source_keys)
        or len({date for _, date in source_keys}) != len(source_keys)
    ):
        _fail(f"{sleeve_name} source rows are duplicate or ambiguous")
    if sleeve_name == "rex_taker_low_range_position":
        return sorted(
            decoded,
            key=lambda row: int(row.get("signal_pos", -1)),
        )
    if sleeve_name != "cand_rex_veto_7":
        _fail(f"unknown REX source sleeve: {sleeve_name}")
    ordered = list(decoded)
    positions = [int(row.get("signal_pos", -1)) for row in ordered]
    if positions != sorted(positions):
        _fail("cand_rex_veto_7 source rows are not monotonic")
    return ordered


class _CountingRaw(io.RawIOBase):
    """Binary source wrapper that reports bytes actually returned to a decoder."""

    def __init__(self, path: Path, on_read: Any) -> None:
        super().__init__()
        self._handle = path.open("rb", buffering=0)
        self._on_read = on_read
        self.name = os.fspath(path)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        count = self._handle.readinto(buffer)
        if count:
            self._on_read(int(count))
        return int(count or 0)

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        return int(self._handle.seek(offset, whence))

    def tell(self) -> int:
        return int(self._handle.tell())

    def fileno(self) -> int:
        return int(self._handle.fileno())

    def close(self) -> None:
        if not self.closed:
            self._handle.close()
        super().close()


class _CausalRowTracker:
    """Mark parser-return ordinals at the exact causal handoff boundary."""

    def __init__(self, counters: dict[str, Any]) -> None:
        self._counters = counters
        self._seen = {
            name: set()
            for name in counters["rows_decoded"]
        }

    def frame_ordinals(self, frame: Any) -> tuple[int, ...]:
        attrs = getattr(frame, "attrs", {})
        values = attrs.get("_g9cb_parser_ordinals")
        if values is None:
            _fail("causal frame lacks parser-return ordinals")
        result = tuple(int(value) for value in values)
        if len(result) != len(frame) or len(result) != len(set(result)):
            _fail("causal frame parser ordinals are invalid")
        return result

    def handoff(
        self,
        logical_name: str,
        ordinals: Iterable[int],
    ) -> None:
        if logical_name not in self._seen:
            _fail(f"unknown causal logical source: {logical_name}")
        seen = self._seen[logical_name]
        before = len(seen)
        for ordinal in ordinals:
            value = int(ordinal)
            if value < 0 or value >= self._counters["rows_decoded"][logical_name]:
                _fail(f"causal source ordinal is out of range: {logical_name}")
            seen.add(value)
        added = len(seen) - before
        self._counters["rows_used"]["causal_feature_rows_by_source"][
            logical_name
        ] += added
        if (
            self._counters["rows_used"]["causal_feature_rows_by_source"][
                logical_name
            ]
            != len(seen)
        ):
            _fail(f"causal source bitset cardinality differs: {logical_name}")

    def handoff_frame(self, logical_name: str, frame: Any) -> None:
        self.handoff(logical_name, self.frame_ordinals(frame))

    def validate(self) -> None:
        observed = self._counters["rows_used"][
            "causal_feature_rows_by_source"
        ]
        if list(observed) != list(self._seen):
            _fail("causal source counter key order differs")
        for name, values in self._seen.items():
            if observed[name] != len(values):
                _fail(f"causal source counter differs: {name}")


def _install_counted_csv_reader(
    pandas_module: Any,
    root: Path,
    sources: Mapping[str, str],
    counters: dict[str, Any],
) -> Any:
    """Count every authenticated CSV decode performed by generic modules."""

    aliases: dict[str, str] = {}
    for logical_name, path_text in sources.items():
        candidate = Path(path_text)
        candidate = candidate if candidate.is_absolute() else root / candidate
        for alias in (candidate.absolute(), candidate.resolve(strict=False)):
            key = os.fspath(alias)
            prior = aliases.get(key)
            if prior is not None and prior != logical_name:
                _fail("two logical sources resolve to the same CSV path")
            aliases[key] = logical_name

    original = pandas_module.read_csv
    def counted_read_csv(source: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            source_path = Path(os.fspath(source))
        except TypeError as exc:
            raise TerminalG9CB12Failure(
                "generic CSV read did not use an authenticated path"
            ) from exc
        candidate = (
            source_path
            if source_path.is_absolute()
            else root / source_path
        )
        logical_name = aliases.get(os.fspath(candidate.absolute()))
        if logical_name is None:
            logical_name = aliases.get(
                os.fspath(candidate.resolve(strict=False))
            )
        if logical_name is None:
            _fail(f"generic CSV read is outside the source closure: {source}")
        if candidate.is_symlink() or not candidate.is_file():
            _fail(f"generic CSV source is absent or unsafe: {source}")
        byte_counters = counters["file_access"]["bytes_read_by_logical_source"]

        def count_bytes(count: int) -> None:
            byte_counters[logical_name] = (
                byte_counters.get(logical_name, 0) + int(count)
            )

        raw = _CountingRaw(candidate, count_bytes)
        counters["file_access"]["source_files_opened"] += 1
        call_kwargs = dict(kwargs)
        if call_kwargs.get("compression", "infer") == "infer":
            call_kwargs["compression"] = (
                "gzip" if candidate.name.endswith(".gz") else None
            )
        try:
            with io.BufferedReader(raw) as buffered:
                frame = original(buffered, *args, **call_kwargs)
        finally:
            raw.close()
        try:
            decoded_rows = len(frame)
        except TypeError as exc:
            raise TerminalG9CB12Failure(
                "chunked or streaming CSV decode is forbidden"
            ) from exc
        first_ordinal = counters["rows_decoded"][logical_name]
        counters["rows_decoded"][logical_name] += int(decoded_rows)
        if logical_name == "market_5m":
            try:
                raw_market_dates = pandas_module.to_datetime(
                    frame["date"],
                    utc=True,
                    errors="raise",
                    format="mixed",
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise TerminalG9CB12Failure(
                    "raw market timestamps are invalid"
                ) from exc
            if bool(
                (
                    raw_market_dates
                    >= pandas_module.Timestamp(DOMAIN_END)
                ).any()
            ):
                _fail("raw market contains a physical domain-end value row")
        try:
            frame.attrs["_g9cb_parser_ordinals"] = tuple(
                range(first_ordinal, first_ordinal + int(decoded_rows))
            )
            frame.attrs["_g9cb_logical_source"] = logical_name
        except (AttributeError, TypeError) as exc:
            raise TerminalG9CB12Failure(
                "decoded CSV frame cannot carry parser ordinals"
            ) from exc
        return frame

    pandas_module.read_csv = counted_read_csv
    return original


def _load_rank7_funding(
    path: str,
    cutoff: str,
    pandas_module: Any,
    causal_rows: _CausalRowTracker,
) -> Any:
    frame = pandas_module.read_csv(path, compression="infer")
    required = {"date", "funding_rate"}
    if not required.issubset(frame.columns):
        _fail("Rank7 funding source lacks required columns")
    ordinal_column = "_g9cb_rank7_parser_ordinal"
    if ordinal_column in frame.columns:
        _fail("Rank7 funding source collides with parser ordinal column")
    parser_ordinals = causal_rows.frame_ordinals(frame)
    frame = frame[["date", "funding_rate"]].copy()
    frame[ordinal_column] = parser_ordinals
    frame["date"] = pandas_module.to_datetime(
        frame["date"],
        utc=True,
        errors="raise",
        format="mixed",
    ).dt.tz_convert(None)
    frame["funding_rate"] = pandas_module.to_numeric(
        frame["funding_rate"],
        errors="raise",
    )
    frame = frame[frame["date"] < pandas_module.Timestamp(cutoff)]
    frame = (
        frame.sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    if frame["date"].duplicated().any():
        _fail("Rank7 funding source remains duplicate")
    retained_ordinals = tuple(
        int(value) for value in frame.pop(ordinal_column).tolist()
    )
    frame.attrs["_g9cb_parser_ordinals"] = retained_ordinals
    frame.attrs["_g9cb_logical_source"] = "funding"
    causal_rows.handoff_frame("funding", frame)
    return frame


_RANK7_SPOT_PREMIUM_PROJECTION = (
    "date",
    "spot_close",
    "spot_rows",
    "premium_index_1m_close",
    "premium_rows",
)


def _attach_rank7_spot_premium_exact(
    market: Any,
    raw_projection: Any,
    pandas_module: Any,
    numpy_module: Any,
) -> Any:
    """Exact UTC-naive one-to-one left join; no fill or time remapping."""

    missing = [
        column
        for column in _RANK7_SPOT_PREMIUM_PROJECTION
        if column not in raw_projection.columns
    ]
    if missing:
        _fail(f"Rank7 spot/premium projection columns differ: {missing!r}")
    if any(
        column in market.columns
        for column in _RANK7_SPOT_PREMIUM_PROJECTION[1:]
    ):
        _fail("market already contains Rank7 spot/premium projection columns")
    projection = raw_projection.loc[
        :, list(_RANK7_SPOT_PREMIUM_PROJECTION)
    ].copy()
    try:
        projection_dates = pandas_module.to_datetime(
            projection["date"],
            utc=True,
            errors="raise",
            format="mixed",
        ).dt.tz_convert(None)
        market_dates = pandas_module.to_datetime(
            market["date"],
            utc=True,
            errors="raise",
            format="mixed",
        ).dt.tz_convert(None)
    except (KeyError, TypeError, ValueError) as exc:
        raise TerminalG9CB12Failure(
            "Rank7 spot/premium timestamps are invalid"
        ) from exc
    if (
        projection_dates.isna().any()
        or projection_dates.duplicated().any()
        or not projection_dates.is_monotonic_increasing
        or bool(
            (
                projection_dates.astype("int64")
                % (300 * 1_000_000_000)
            ).any()
        )
    ):
        _fail("Rank7 spot/premium timestamp structure differs")
    if market_dates.isna().any() or market_dates.duplicated().any():
        _fail("market timestamps are invalid for Rank7 projection join")
    if (
        not market_dates.is_monotonic_increasing
        or bool(
            (
                market_dates.astype("int64")
                % (300 * 1_000_000_000)
            ).any()
        )
        or len(projection_dates) != len(market_dates)
        or not projection_dates.reset_index(drop=True).equals(
            market_dates.reset_index(drop=True)
        )
    ):
        _fail("Rank7 spot/premium timestamp vector differs from market")
    projection["date"] = projection_dates
    market_copy = market.copy()
    market_copy["date"] = market_dates
    for column in _RANK7_SPOT_PREMIUM_PROJECTION[1:]:
        projection[column] = pandas_module.to_numeric(
            projection[column], errors="coerce"
        )
        values = projection[column].to_numpy(dtype="float64")
        if projection[column].isna().any() or not numpy_module.isfinite(
            values
        ).all():
            _fail(f"Rank7 spot/premium values are invalid: {column}")
    if (
        not projection["spot_close"].gt(0).all()
        or not projection["spot_rows"].eq(5).all()
        or not projection["premium_rows"].eq(5).all()
    ):
        _fail("Rank7 spot/premium row validity differs")
    joined = market_copy.merge(
        projection,
        on="date",
        how="left",
        sort=False,
        validate="one_to_one",
    )
    if (
        len(joined) != len(market_copy)
        or not joined["date"].equals(market_copy["date"])
        or joined[list(_RANK7_SPOT_PREMIUM_PROJECTION[1:])]
        .isna()
        .any()
        .any()
    ):
        _fail("Rank7 spot/premium exact market coverage differs")
    return joined


class _StructuralTrade:
    """Only the geometry needed by generic schedule walkers."""

    __slots__ = (
        "signal_position",
        "entry_position",
        "exit_position",
        "side",
        "exit_kind",
    )

    def __init__(
        self,
        *,
        signal_position: int,
        entry_position: int,
        exit_position: int,
        side: int,
        exit_kind: str,
    ) -> None:
        self.signal_position = int(signal_position)
        self.entry_position = int(entry_position)
        self.exit_position = int(exit_position)
        self.side = int(side)
        self.exit_kind = str(exit_kind)


class _StructuralTradeEngine:
    """OHLC-only, stop-before-take structural replay with no economics."""

    def __init__(
        self,
        market: Any,
        counters: dict[str, Any],
        sleeve: str,
    ) -> None:
        self._open = market["open"].to_numpy(float)
        self._high = market["high"].to_numpy(float)
        self._low = market["low"].to_numpy(float)
        if not (len(self._open) == len(self._high) == len(self._low)):
            _fail(f"{sleeve} structural OHLC arrays differ in length")
        self._counters = counters
        self._sleeve = sleeve
        self._cache: dict[
            tuple[int, int, int, int, int], _StructuralTrade | None
        ] = {}

    def trade_at(
        self,
        signal: int,
        side: int,
        hold: int,
        take_bps: int,
        stop_bps: int,
    ) -> _StructuralTrade | None:
        key = (
            int(signal),
            int(side),
            int(hold),
            int(take_bps),
            int(stop_bps),
        )
        if key in self._cache:
            return self._cache[key]
        signal, side, hold, take_bps, stop_bps = key
        if side not in (-1, 1) or hold <= 0:
            _fail(f"{self._sleeve} structural replay contract is invalid")
        if take_bps < 0 or stop_bps < 0:
            _fail(f"{self._sleeve} structural barrier is negative")
        entry = signal + 1
        horizon = entry + hold
        if signal < 0 or entry < 0 or horizon > len(self._open):
            self._cache[key] = None
            return None
        take_enabled = take_bps < 1_000_000
        stop_enabled = stop_bps < 1_000_000
        entry_price = float(self._open[entry])
        if not entry_price > 0.0:
            _fail(f"{self._sleeve} structural entry price is invalid")
        take = take_bps / 10_000.0
        stop = stop_bps / 10_000.0
        exit_position = horizon
        exit_kind = "fixed"
        for position in range(entry, horizon):
            high = float(self._high[position])
            low = float(self._low[position])
            self._counters["rows_used"][
                "outcome_dependent_ohlc_rows_examined"
            ] += 1
            self._counters["per_sleeve"][self._sleeve][
                "outcome_dependent_ohlc_rows_examined"
            ] += 1
            if not (high > 0.0 and low > 0.0 and high >= low):
                _fail(f"{self._sleeve} structural OHLC row is invalid")
            if side == 1:
                stop_hit = stop_enabled and low <= entry_price * (1.0 - stop)
                take_hit = take_enabled and high >= entry_price * (1.0 + take)
            else:
                stop_hit = stop_enabled and high >= entry_price * (1.0 + stop)
                take_hit = take_enabled and low <= entry_price * (1.0 - take)
            if stop_hit:
                exit_position = position
                exit_kind = "stop"
                break
            if take_hit:
                exit_position = position
                exit_kind = "take"
                break
        trade = _StructuralTrade(
            signal_position=signal,
            entry_position=entry,
            exit_position=exit_position,
            side=side,
            exit_kind=exit_kind,
        )
        self._cache[key] = trade
        return trade


class _Rank7LabelTrade:
    """The exact four factor labels plus geometry authorized by G9CB-1A."""

    __slots__ = (
        "signal_position",
        "entry_position",
        "exit_position",
        "side",
        "price_factor",
        "funding_factor",
        "funding_debit_factor",
        "adverse_price_factor",
    )

    def __init__(
        self,
        *,
        signal_position: int,
        entry_position: int,
        exit_position: int,
        side: int,
        price_factor: float,
        funding_factor: float,
        funding_debit_factor: float,
        adverse_price_factor: float,
    ) -> None:
        self.signal_position = int(signal_position)
        self.entry_position = int(entry_position)
        self.exit_position = int(exit_position)
        self.side = int(side)
        self.price_factor = float(price_factor)
        self.funding_factor = float(funding_factor)
        self.funding_debit_factor = float(funding_debit_factor)
        self.adverse_price_factor = float(adverse_price_factor)


class _Rank7LabelEngine:
    """Narrow exact-label replay with no return, PnL, or favorable-path fields."""

    def __init__(
        self,
        market: Any,
        funding: Any,
        counters: dict[str, Any],
        numpy_module: Any,
    ) -> None:
        np = numpy_module
        self._np = np
        self._open = market["open"].to_numpy(float)
        self._high = market["high"].to_numpy(float)
        self._low = market["low"].to_numpy(float)
        self._dates = (
            market["date"]
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
        )
        self._funding_times = (
            funding["date"]
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
        )
        self._funding_rates = funding["funding_rate"].to_numpy(float)
        if not (
            len(self._open)
            == len(self._high)
            == len(self._low)
            == len(self._dates)
        ):
            _fail("Rank7 label market arrays differ in length")
        if len(self._funding_times) != len(self._funding_rates):
            _fail("Rank7 label funding arrays differ in length")
        if len(self._funding_times) and (
            np.any(np.diff(self._funding_times) <= 0)
            or not np.isfinite(self._funding_rates).all()
        ):
            _fail("Rank7 label funding rows are invalid")
        self._counters = counters
        self._cache: dict[
            tuple[int, int, int, int, int], _Rank7LabelTrade | None
        ] = {}

    def trade_at(
        self,
        signal: int,
        side: int,
        hold: int,
        take_bps: int,
        stop_bps: int,
    ) -> _Rank7LabelTrade | None:
        key = (
            int(signal),
            int(side),
            int(hold),
            int(take_bps),
            int(stop_bps),
        )
        if key in self._cache:
            return self._cache[key]
        signal, side, hold, take_bps, stop_bps = key
        if side not in (-1, 1) or hold <= 0:
            _fail("Rank7 label replay geometry is invalid")
        if take_bps < 0 or stop_bps < 0:
            _fail("Rank7 label replay barrier is negative")
        entry = signal + 1
        cap = entry + hold
        if signal < 0 or entry < 0 or cap >= len(self._open):
            self._cache[key] = None
            return None

        entry_price = float(self._open[entry])
        cap_price = float(self._open[cap])
        if not (entry_price > 0.0 and cap_price > 0.0):
            _fail("Rank7 label entry or cap price is invalid")
        leverage = float(RANK7_LABEL_EXECUTION["leverage"])
        take = take_bps / 10_000.0
        stop = stop_bps / 10_000.0
        take_enabled = take_bps < 1_000_000
        stop_enabled = stop_bps < 1_000_000
        exit_position = cap
        exit_price_ratio = cap_price / entry_price
        adverse_price = entry_price
        for position in range(entry, cap):
            high = float(self._high[position])
            low = float(self._low[position])
            self._counters["rows_used"][
                "outcome_dependent_ohlc_rows_examined"
            ] += 1
            self._counters["per_sleeve"]["frozen_annual_rank7"][
                "outcome_dependent_ohlc_rows_examined"
            ] += 1
            if not (high > 0.0 and low > 0.0 and high >= low):
                _fail("Rank7 label OHLC row is invalid")
            if side == 1:
                adverse_price = min(adverse_price, low)
                stop_hit = stop_enabled and low <= entry_price * (1.0 - stop)
                take_hit = take_enabled and high >= entry_price * (1.0 + take)
            else:
                adverse_price = max(adverse_price, high)
                stop_hit = stop_enabled and high >= entry_price * (1.0 + stop)
                take_hit = take_enabled and low <= entry_price * (1.0 - take)
            if stop_hit:
                exit_position = position
                exit_price_ratio = 1.0 - side * stop
                adverse_price = entry_price * (1.0 - side * stop)
                break
            if take_hit:
                exit_position = position
                exit_price_ratio = 1.0 + side * take
                break

        price_factor = max(
            0.0,
            1.0 + leverage * side * (exit_price_ratio - 1.0),
        )
        adverse_price_factor = max(
            0.0,
            1.0
            + leverage * side * (adverse_price / entry_price - 1.0),
        )
        entry_ns = int(self._dates[entry])
        exit_ns = int(self._dates[exit_position])
        left = int(
            self._np.searchsorted(self._funding_times, entry_ns, side="left")
        )
        right = int(
            self._np.searchsorted(self._funding_times, exit_ns, side="right")
        )
        factors = (
            1.0
            - leverage
            * side
            * self._funding_rates[left:right]
        )
        if (
            not self._np.isfinite(factors).all()
            or (factors <= 0.0).any()
        ):
            _fail("Rank7 label realized funding factor is invalid")
        funding_factor = (
            float(self._np.prod(factors, dtype=float))
            if len(factors)
            else 1.0
        )
        funding_debit_factor = (
            float(
                self._np.prod(
                    self._np.minimum(factors, 1.0),
                    dtype=float,
                )
            )
            if len(factors)
            else 1.0
        )
        trade = _Rank7LabelTrade(
            signal_position=signal,
            entry_position=entry,
            exit_position=exit_position,
            side=side,
            price_factor=price_factor,
            funding_factor=funding_factor,
            funding_debit_factor=funding_debit_factor,
            adverse_price_factor=adverse_price_factor,
        )
        self._cache[key] = trade
        return trade


def _naive_utc_timestamp(value: Any, pandas_module: Any) -> Any:
    timestamp = pandas_module.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp
    return timestamp.tz_convert("UTC").tz_localize(None)


def _rank7_bundle_activation_with_parity(
    rank7_runtime: Any,
    bundle: Any,
    context: Mapping[str, Any],
    historical_active: Any,
    counters: dict[str, Any],
    numpy_module: Any,
    pandas_module: Any,
) -> Any:
    """Score the bundle-valid anchors and prove full-window activation parity."""

    np = numpy_module
    pd = pandas_module
    dates = pd.Series(pd.to_datetime(context["dates"], utc=True)).dt.tz_convert(
        None
    )
    matrix = np.asarray(context["matrix"], dtype=float)
    anchors = np.asarray(context["anchors"], dtype=bool)
    funding_leg = np.asarray(context["funding_leg"], dtype=bool)
    premium_leg = np.asarray(context["premium_leg"], dtype=bool)
    historical = np.asarray(historical_active, dtype=bool)
    row_count = len(dates)
    expected_shape = (row_count,)
    if (
        matrix.shape[0] != row_count
        or anchors.shape != expected_shape
        or funding_leg.shape != expected_shape
        or premium_leg.shape != expected_shape
        or historical.shape != expected_shape
    ):
        _fail("Rank7 parity arrays differ in length")

    start = max(
        _naive_utc_timestamp(DOMAIN_START, pd),
        _naive_utc_timestamp(bundle.valid_from, pd),
    )
    end = min(
        _naive_utc_timestamp(DOMAIN_END, pd),
        _naive_utc_timestamp(bundle.valid_until, pd),
    )
    if not start < end:
        _fail("Rank7 bundle has no valid overlap with the canonical domain")
    valid = np.asarray((dates >= start) & (dates < end), dtype=bool)
    if not valid.any():
        _fail("Rank7 bundle-valid domain has no market rows")

    bundle_active = np.zeros(row_count, dtype=bool)
    scored_sources: dict[int, str | None] = {}
    if len(tuple(bundle.models)) != 5:
        _fail("Rank7 loaded model count differs")
    for raw_signal in np.flatnonzero(anchors & valid):
        signal = int(raw_signal)
        decision = rank7_runtime.score_rank7_row(
            bundle,
            matrix[signal],
            decision_ts=dates.iloc[signal],
            is_anchor=True,
        )
        counters["rows_used"]["rank7_bundle_activation_rows_scored"] += 1
        expected_source = (
            "funding"
            if bool(funding_leg[signal])
            else "premium"
            if bool(premium_leg[signal])
            else None
        )
        observed_source = getattr(decision, "source", None)
        if observed_source != expected_source:
            _fail("Rank7 bundle source identity differs from annual refit")
        scored_sources[signal] = observed_source
        bundle_active[signal] = bool(getattr(decision, "active", False))

    for position, is_valid in enumerate(valid):
        if not bool(is_valid):
            continue
        matches = bool(historical[position]) == bool(bundle_active[position])
        counters["rows_used"]["rank7_bundle_parity_rows_compared"] += 1
        if not matches:
            _fail("Rank7 annual-refit and bundle activation differ")
    for raw_signal in np.flatnonzero(historical & valid):
        signal = int(raw_signal)
        expected_source = "funding" if bool(funding_leg[signal]) else "premium"
        if scored_sources.get(signal) != expected_source:
            _fail("Rank7 active source or scheduled entry differs")
    return bundle_active


def _append_direct_interval(
    intervals: dict[str, list[tuple[int, int, int, str]]],
    counters: dict[str, Any],
    sleeve: str,
    entry: int,
    exit_: int,
    side: int,
    exit_kind: str,
) -> None:
    if side not in SLEEVE_BY_NAME[sleeve]["sides"]:
        _fail(f"direct adapter emitted forbidden side for {sleeve}")
    start = _parse_timestamp(DOMAIN_START)
    end = _parse_timestamp(DOMAIN_END)
    if not (start <= entry < exit_ <= end):
        return
    rows = intervals[sleeve]
    if rows and entry < rows[-1][1]:
        _fail(f"direct adapter emitted overlapping {sleeve} intervals")
    rows.append((entry, exit_, side, exit_kind))
    per = counters["per_sleeve"][sleeve]
    per["intervals_emitted"] += 1
    per["long_intervals" if side == 1 else "short_intervals"] += 1
    per[
        "fixed_horizon_exits"
        if exit_kind == "fixed"
        else f"{exit_kind}_exits"
    ] += 1


def _schedule_direct_fixed_interval(
    intervals: dict[str, list[tuple[int, int, int, str]]],
    counters: dict[str, Any],
    sleeve: str,
    value_opens: Sequence[int],
    boundaries: Sequence[int],
    mask: Sequence[Any],
    entry_position: int,
    exit_position: int,
    side: int,
    split_start: int,
    split_end: int,
) -> bool:
    """Schedule a fixed exit from geometry only, without reading OHLC."""

    if entry_position < 0 or entry_position >= len(value_opens):
        _fail(f"direct adapter fixed entry is outside values: {sleeve}")
    if not _structural_exit_is_eligible(
        mask,
        boundaries,
        exit_position,
        "fixed",
        split_start,
        split_end,
    ):
        return False
    _append_direct_interval(
        intervals,
        counters,
        sleeve,
        value_opens[entry_position],
        boundaries[exit_position],
        side,
        "fixed",
    )
    return True


def _materialize_direct_rows(
    intervals: Mapping[str, Sequence[tuple[int, int, int, str]]],
    counters: dict[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for sleeve in SLEEVES:
        name = sleeve["name"]
        values = list(intervals[name])
        if values != sorted(values, key=lambda row: (row[0], row[1], row[2])):
            _fail(f"direct adapter {name} rows are unsorted")
        for index, (entry, exit_, side, exit_kind) in enumerate(values):
            output.append(
                {
                    "identity": IDENTITY,
                    "sleeve": name,
                    "sleeve_order": sleeve["order"],
                    "configured_weight": sleeve["weight"],
                    "interval_index": index,
                    "entry_time_utc": _timestamp(entry),
                    "exit_time_utc": _timestamp(exit_),
                    "side": side,
                }
            )
    return output


def _direct_generic_adapter_impl(
    modules: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    root: Path,
    sources: Mapping[str, str],
    counters: dict[str, Any],
    causal_rows: _CausalRowTracker,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reconstruct all five clocks from the two isolated runtime roots."""

    primitives = modules["training.gross9_structural_clock_primitives"]
    rank7_runtime = modules["execution.gross9_rank7_clock_runtime"]
    np = primitives.np
    pd = primitives.pd
    configs = {
        "cand_rex_veto_7": _load_worker_json(
            root, _authority_path(preregistration, "rex_veto_config")
        ),
        "fresh_kimchi_fx": _load_worker_json(
            root, _authority_path(preregistration, "fresh_kimchi_config")
        ),
        "frozen_annual_rank7": _load_worker_json(
            root, _authority_path(preregistration, "rank7_config")
        ),
        "markov_transition_long": _load_worker_json(
            root, _authority_path(preregistration, "markov_config")
        ),
        "rex_taker_low_range_position": _load_worker_json(
            root, _authority_path(preregistration, "rex_taker_config")
        ),
    }

    original_normalise = primitives.normalise_market
    original_attach_aux = primitives.attach_binance_um_aux_frames

    def counted_normalise(frame: Any, *args: Any, **kwargs: Any) -> Any:
        if getattr(frame, "attrs", {}).get("_g9cb_logical_source") == "market_5m":
            causal_rows.handoff_frame("market_5m", frame)
        return original_normalise(frame, *args, **kwargs)

    def counted_attach_aux(
        market_frame: Any,
        *args: Any,
        funding_frame: Any = None,
        premium_frame: Any = None,
        **kwargs: Any,
    ) -> Any:
        if funding_frame is not None:
            causal_rows.handoff_frame("funding", funding_frame)
        if premium_frame is not None:
            causal_rows.handoff_frame("premium", premium_frame)
        return original_attach_aux(
            market_frame,
            *args,
            funding_frame=funding_frame,
            premium_frame=premium_frame,
            **kwargs,
        )

    primitives.normalise_market = counted_normalise
    primitives.attach_binance_um_aux_frames = counted_attach_aux
    try:
        market = primitives.load_market(
            sources["market_5m"],
            funding_path=sources["funding"],
            premium_path=sources["premium"],
            exclude_from=pd.Timestamp(DOMAIN_END).tz_localize(None),
        )
    finally:
        primitives.normalise_market = original_normalise
        primitives.attach_binance_um_aux_frames = original_attach_aux

    open_interest = pd.read_csv(sources["open_interest"], compression="infer")
    causal_rows.handoff_frame("open_interest", open_interest)
    market = primitives.attach_open_interest(market, open_interest)
    rank7_spot_premium = pd.read_csv(
        sources["rank7_spot_premium_5m"],
        compression="infer",
    )
    causal_rows.handoff_frame(
        "rank7_spot_premium_5m", rank7_spot_premium
    )
    market = _attach_rank7_spot_premium_exact(
        market,
        rank7_spot_premium,
        pd,
        np,
    )
    value_opens, boundaries = _market_value_opens_and_boundaries(market)
    dates = pd.to_datetime(market["date"])
    masks = {
        name: np.asarray(
            (dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end)),
            dtype=bool,
        )
        for name, start, end in SPLIT_BOUNDS
    }
    split_seconds = {
        name: (_parse_timestamp(start), _parse_timestamp(end))
        for name, start, end in SPLIT_BOUNDS
    }
    intervals: dict[str, list[tuple[int, int, int, str]]] = {
        sleeve["name"]: [] for sleeve in SLEEVES
    }

    causal_rows.handoff("market_5m", range(counters["rows_decoded"]["market_5m"]))
    base_features = primitives.build_market_feature_frame(market, window_size=144)
    causal_rows.handoff("market_5m", range(counters["rows_decoded"]["market_5m"]))
    interest_features = primitives.build_interest_features(market, base_features)
    features = pd.concat([base_features, interest_features], axis=1).loc[
        :, lambda frame: ~frame.columns.duplicated(keep="last")
    ]

    markov_cfg = configs["markov_transition_long"]
    clauses = markov_cfg.get("gate_clauses")
    if not isinstance(clauses, list) or not clauses:
        _fail("bound Markov config has no gate clauses")
    setup = np.zeros(len(market), dtype=bool)
    for clause in clauses:
        if not isinstance(clause, list):
            _fail("bound Markov gate clause is invalid")
        causal_rows.handoff("market_5m", range(counters["rows_decoded"]["market_5m"]))
        setup |= primitives.gate_mask(features, clause, fallback=market)
    state_model = markov_cfg.get("state_model")
    if not isinstance(state_model, Mapping):
        _fail("bound Markov state_model is absent")
    causal_rows.handoff("market_5m", range(counters["rows_decoded"]["market_5m"]))
    markov_active = primitives.markov_active(market, setup, state_model)
    markov_hold = int(markov_cfg["hold_bars"])
    markov_stride = int(markov_cfg["stride_bars"])
    markov_offset = int(markov_cfg["stride_offset_bars"])
    if (
        int(markov_cfg.get("entry_delay_bars", -1)) != 1
        or 143 % markov_stride != markov_offset
    ):
        _fail("bound Markov stride offset differs from canonical warm-up")
    markov_positions = np.arange(
        143,
        max(0, len(market) - markov_hold),
        markov_stride,
        dtype=np.int64,
    )
    for split_name, mask in masks.items():
        next_allowed = 0
        for raw_position in markov_positions:
            position = int(raw_position)
            if not bool(mask[position]):
                continue
            counters["per_sleeve"]["markov_transition_long"][
                "signal_rows_evaluated"
            ] += 1
            if not bool(markov_active[position]) or position < next_allowed:
                continue
            entry = position + 1
            exit_position = entry + markov_hold
            split_start, split_end = split_seconds[split_name]
            if not _schedule_direct_fixed_interval(
                intervals,
                counters,
                "markov_transition_long",
                value_opens,
                boundaries,
                mask,
                entry,
                exit_position,
                1,
                split_start,
                split_end,
            ):
                continue
            next_allowed = exit_position + 1

    causal_rows.handoff("market_5m", range(counters["rows_decoded"]["market_5m"]))
    rex_features = primitives.build_light_rex_features(market)
    rex_specs = (
        (
            "cand_rex_veto_7",
            ("rex_veto_source",),
            "base_event",
            "base_side",
        ),
        (
            "rex_taker_low_range_position",
            ("rex_taker_train", "rex_taker_test", "rex_taker_eval"),
            "action",
            "side",
        ),
    )
    for sleeve_name, logical_sources, side_container, side_key in rex_specs:
        config = configs[sleeve_name]
        hold = int(config["hold_bars"])
        if (
            hold != 144
            or int(config.get("stride_bars", -1)) != 24
            or int(config.get("stride_offset_bars", -1)) != 11
            or int(config.get("entry_delay_bars", -1)) != 1
        ):
            _fail(f"{sleeve_name} bound structural contract differs")
        decoded: list[dict[str, Any]] = []
        row_sources: dict[int, str] = {}
        for logical_name in logical_sources:
            source_rows = _read_jsonl_rows(
                root,
                sources[logical_name],
                counters,
                logical_name,
            )
            decoded.extend(source_rows)
            row_sources.update({id(row): logical_name for row in source_rows})
        ordered = _ordered_rex_source_rows(decoded, sleeve_name)
        for split_name, mask in masks.items():
            next_allowed = 0
            for source_row in ordered:
                position = int(source_row.get("signal_pos", -1))
                if position < 0 or position >= len(market) or not bool(mask[position]):
                    continue
                counters["per_sleeve"][sleeve_name][
                    "signal_rows_evaluated"
                ] += 1
                if position < next_allowed:
                    continue
                source_second = _generic_time_second(source_row["date"], pd)
                if (
                    source_second % 300
                    or source_second != value_opens[position]
                ):
                    _fail(f"{sleeve_name} source row is off the market grid")
                logical_name = row_sources[id(source_row)]
                causal_rows.handoff(
                    logical_name,
                    (int(source_row["_g9cb_parser_ordinal"]),),
                )
                gate_match = (
                    primitives.rex_veto_gate_match(
                        config["gates"], rex_features, source_row
                    )
                    if sleeve_name == "cand_rex_veto_7"
                    else primitives.rex_taker_gate_match(
                        source_row, config["gates"]
                    )
                )
                if not gate_match:
                    continue
                side_text = str(
                    (source_row.get(side_container) or {}).get(side_key, "")
                ).lower()
                if side_text not in ("long", "short"):
                    continue
                side = 1 if side_text == "long" else -1
                entry = position + 1
                exit_position = entry + hold
                split_start, split_end = split_seconds[split_name]
                if not _schedule_direct_fixed_interval(
                    intervals,
                    counters,
                    sleeve_name,
                    value_opens,
                    boundaries,
                    mask,
                    entry,
                    exit_position,
                    side,
                    split_start,
                    split_end,
                ):
                    continue
                next_allowed = exit_position + 1

    fresh_cfg = configs["fresh_kimchi_fx"]
    expected_fresh_geometry = {
        "hold_bars": 288,
        "stride_bars": 6,
        "entry_delay_bars": 1,
        "take_bps": 400,
        "stop_bps": 250,
    }
    for key, value in expected_fresh_geometry.items():
        if int(fresh_cfg.get(key, -1)) != value:
            _fail(f"Fresh config structural mismatch: {key}")
    if 143 % int(fresh_cfg["stride_bars"]) != int(
        fresh_cfg["stride_offset_bars"]
    ):
        _fail("Fresh stride offset differs from canonical warm-up")
    causal_rows.handoff("market_5m", range(counters["rows_decoded"]["market_5m"]))
    fresh_base = primitives.build_market_feature_frame(market, window_size=144)
    causal_rows.handoff("market_5m", range(counters["rows_decoded"]["market_5m"]))
    fresh_interest = primitives.build_interest_features(market, fresh_base)
    fresh_features = pd.concat([fresh_base, fresh_interest], axis=1)
    causal_rows.handoff("market_5m", range(counters["rows_decoded"]["market_5m"]))
    fresh_features = primitives.build_bidirectional_features(market, fresh_features)
    causal_rows.handoff("market_5m", range(counters["rows_decoded"]["market_5m"]))
    fresh_features = primitives.build_kimchi_features(market, fresh_features)
    long_gates = list(fresh_cfg.get("long_gates", ()))
    short_gates = list(fresh_cfg.get("short_gates", ()))
    expected_availability_gates = {
        ("usdkrw_available", ">=", 0.5),
        ("kimchi_available", ">=", 0.5),
    }
    observed_availability_gates = {
        (str(row.get("feature")), str(row.get("op")), float(row.get("threshold")))
        for row in long_gates + short_gates
        if str(row.get("feature", "")).endswith("_available")
    }
    if observed_availability_gates != expected_availability_gates:
        _fail("Fresh config availability gates differ")
    long_conditions = [
        row for row in long_gates if not str(row["feature"]).endswith("_available")
    ]
    short_conditions = [
        row for row in short_gates if not str(row["feature"]).endswith("_available")
    ]
    causal_rows.handoff("market_5m", range(counters["rows_decoded"]["market_5m"]))
    fresh_long_active, fresh_short_active, _ = primitives.fresh_masks(
        market,
        fresh_features,
        long_conditions=long_conditions,
        short_conditions=short_conditions,
        long_availability=(
            "funding_available",
            "usdkrw_available",
            "kimchi_available",
        ),
        short_availability=("usdkrw_available",),
    )
    fresh_engine = _StructuralTradeEngine(
        market,
        counters,
        "fresh_kimchi_fx",
    )
    fresh_positions = np.arange(
        143,
        len(market) - int(fresh_cfg["hold_bars"]),
        int(fresh_cfg["stride_bars"]),
        dtype=np.int64,
    )
    fresh_active = np.logical_xor(fresh_long_active, fresh_short_active)
    for split_name, mask in masks.items():
        next_allowed = 0
        for raw_signal in fresh_positions:
            signal_position = int(raw_signal)
            if not bool(mask[signal_position]):
                continue
            counters["per_sleeve"]["fresh_kimchi_fx"][
                "signal_rows_evaluated"
            ] += 1
            if signal_position < next_allowed or not bool(
                fresh_active[signal_position]
            ):
                continue
            side = 1 if bool(fresh_long_active[signal_position]) else -1
            trade = fresh_engine.trade_at(
                signal_position,
                side,
                int(fresh_cfg["hold_bars"]),
                int(fresh_cfg["take_bps"]),
                int(fresh_cfg["stop_bps"]),
            )
            if trade is None:
                continue
            split_start, split_end = split_seconds[split_name]
            if not _structural_exit_is_eligible(
                mask,
                boundaries,
                int(trade.exit_position),
                str(trade.exit_kind),
                split_start,
                split_end,
            ):
                continue
            exit_boundary = int(trade.exit_position) + (
                0 if trade.exit_kind == "fixed" else 1
            )
            _append_direct_interval(
                intervals,
                counters,
                "fresh_kimchi_fx",
                value_opens[int(trade.entry_position)],
                boundaries[exit_boundary],
                side,
                str(trade.exit_kind),
            )
            next_allowed = int(trade.exit_position) + 1

    bundle_manifest = _load_worker_json(
        root, _authority_path(preregistration, "rank7_bundle_manifest")
    )
    if bundle_manifest.get("bundle_manifest_hash") != rank7_runtime.rank7_manifest_hash(
        bundle_manifest
    ):
        _fail("Rank7 bundle manifest internal hash differs")
    rank7_cfg = configs["frozen_annual_rank7"]
    if (
        rank7_cfg.get("side") != "LONG"
        or int(rank7_cfg.get("hold_bars", -1)) != 576
        or int(rank7_cfg.get("stride_bars", -1)) != 1
        or int(rank7_cfg.get("entry_delay_bars", -1)) != 1
        or rank7_cfg.get("bundle_manifest_hash")
        != bundle_manifest.get("bundle_manifest_hash")
    ):
        _fail("Rank7 bound config structural contract differs")
    expected_exits = {
        "funding": {
            "hold_bars": 576,
            "stop_bps": 1_000_000,
            "take_bps": 400,
        },
        "premium": {
            "hold_bars": 144,
            "stop_bps": 300,
            "take_bps": 1_000_000,
        },
    }
    if bundle_manifest.get("exits_by_source") != expected_exits:
        _fail("Rank7 bundle source-routed exits differ")
    delay_bars = int(bundle_manifest.get("delay_bars", -1))
    runtime_contract = rank7_cfg.get("runtime_contract")
    if (
        not isinstance(runtime_contract, Mapping)
        or int(runtime_contract.get("predictor_delay_bars", -1)) != delay_bars
        or int(runtime_contract.get("anchor_cooldown_bars", -1)) != 144
    ):
        _fail("Rank7 runtime/config feature geometry differs")
    bundle_path_text = rank7_cfg.get("bundle_path")
    if not isinstance(bundle_path_text, str) or not bundle_path_text:
        _fail("Rank7 bound config has no bundle path")
    bundle_root = _rooted(root, Path(bundle_path_text))
    if bundle_root.is_symlink() or not bundle_root.is_dir():
        _fail("Rank7 bundle root is absent or unsafe")
    manifest_path_text = _authority_path(
        preregistration, "rank7_bundle_manifest"
    )
    manifest_path = Path(manifest_path_text)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    if manifest_path.resolve() != (bundle_root / "manifest.json").resolve():
        _fail("Rank7 config and authenticated manifest paths differ")
    bundle = rank7_runtime.Rank7Bundle.load(bundle_root)
    if getattr(bundle, "manifest", None) != bundle_manifest:
        _fail("Rank7 loaded bundle manifest differs")
    models = tuple(getattr(bundle, "models", ()))
    hourly_history = getattr(bundle, "hourly_history", None)
    if len(models) != 5 or hourly_history is None:
        _fail("Rank7 bundle did not load five models and hourly history")
    expected_history_rows = int(
        bundle_manifest.get("hourly_history", {}).get("rows", -1)
    )
    if (
        len(hourly_history) != expected_history_rows
        or counters["rows_decoded"]["rank7_hourly_history"]
        != expected_history_rows
    ):
        _fail("Rank7 hourly-history access counter differs")
    if (
        int(runtime_contract.get("feature_columns", -1))
        != len(tuple(bundle.feature_columns))
        or int(runtime_contract.get("prediction_n_jobs", -1)) != 1
        or runtime_contract.get("source_identity_current") is not True
        or runtime_contract.get("no_overlap") is not True
        or runtime_contract.get("source_specific_barrier_exits") is not True
        or rank7_cfg.get("model_version") != bundle.model_version
    ):
        _fail("Rank7 loaded runtime contract differs from bound config")
    expected_model_params = {
        "bootstrap": False,
        "max_depth": int(RANK7_LEARNER["max_depth"]),
        "max_features": float(RANK7_LEARNER["max_features"]),
        "min_samples_leaf": int(RANK7_LEARNER["min_samples_leaf"]),
    }
    if (
        bundle_manifest.get("extra_trees_params") != expected_model_params
        or tuple(bundle_manifest.get("seeds", ()))
        != tuple(primitives.RANK7_SEEDS)
        or int(bundle_manifest.get("trees_per_seed", -1))
        != int(primitives.RANK7_TREES)
        or int(bundle_manifest.get("prediction_n_jobs", -1)) != 1
    ):
        _fail("Rank7 annual learner or deterministic scoring contract differs")
    history_ordinals = getattr(hourly_history, "attrs", {}).get(
        "_g9cb_parser_ordinals"
    )
    if history_ordinals is None:
        history_ordinals = tuple(range(expected_history_rows))
    causal_rows.handoff("rank7_hourly_history", history_ordinals)
    causal_rows.handoff("market_5m", range(counters["rows_decoded"]["market_5m"]))
    rank7_context = rank7_runtime.build_rank7_feature_context(market, bundle)
    rank7_seconds, _ = _market_seconds(rank7_context["market"])
    if rank7_seconds != value_opens:
        _fail("Rank7 generic context market grid differs")
    structural_funding = _load_rank7_funding(
        sources["funding"],
        "2026-06-02",
        pd,
        causal_rows,
    )
    rank7_label_engine = _Rank7LabelEngine(
        rank7_context["market"],
        structural_funding,
        counters,
        np,
    )
    rank7_structural_engine = _StructuralTradeEngine(
        rank7_context["market"],
        counters,
        "frozen_annual_rank7",
    )
    rank7_signals = np.flatnonzero(
        np.asarray(rank7_context["anchors"], dtype=bool)
    )
    rank7_funding_source = np.asarray(
        rank7_context["funding_leg"], dtype=bool
    )[rank7_signals]
    targets: list[tuple[float, float]] = []
    exits: list[int] = []
    for signal_position, is_funding in zip(
        rank7_signals, rank7_funding_source, strict=True
    ):
        hold, take, stop = (
            (576, 400, 1_000_000)
            if bool(is_funding)
            else (144, 1_000_000, 300)
        )
        trade = rank7_label_engine.trade_at(
            int(signal_position), 1, hold, take, stop
        )
        if trade is None:
            targets.append((np.nan, np.nan))
            exits.append(len(value_opens))
            continue
        counters["rows_used"]["rank7_training_trades_replayed"] += 1
        price_factor = float(trade.price_factor)
        funding_factor = float(trade.funding_factor)
        funding_debit_factor = float(trade.funding_debit_factor)
        adverse_price_factor = float(trade.adverse_price_factor)
        counters["rows_used"]["rank7_price_factor_values_used"] += 1
        counters["rows_used"]["rank7_funding_factor_values_used"] += 1
        counters["rows_used"][
            "rank7_funding_debit_factor_values_used"
        ] += 1
        counters["rows_used"][
            "rank7_adverse_price_factor_values_used"
        ] += 1
        fee_factor = 1.0 - RANK7_LABEL_EXECUTION["leverage"] * (
            RANK7_LABEL_EXECUTION["fee_rate"]
            + RANK7_LABEL_EXECUTION["slippage_rate"]
        )
        counters["rows_used"]["rank7_fee_factor_values_used"] += 1
        net_label = (
            fee_factor * price_factor * funding_factor * fee_factor - 1.0
        )
        adverse_label = max(
            0.0,
            1.0
            - fee_factor
            * funding_debit_factor
            * adverse_price_factor,
        )
        if not (np.isfinite(net_label) and np.isfinite(adverse_label)):
            _fail("Rank7 exact training label is nonfinite")
        counters["rows_used"]["rank7_net_labels_computed"] += 1
        counters["rows_used"]["rank7_adverse_labels_computed"] += 1
        targets.append((net_label, adverse_label))
        exits.append(int(trade.exit_position))
    feature_columns = tuple(rank7_context["feature_columns"])
    rank7_base = {
        "context": rank7_context,
        "dates": pd.to_datetime(rank7_context["dates"]),
        "engine": rank7_structural_engine,
        "signals": rank7_signals,
        "funding_source": rank7_funding_source,
        "targets": np.asarray(targets, dtype=float),
        "exit_positions": np.asarray(exits, dtype=int),
        "signal_dates": pd.to_datetime(rank7_context["dates"])
        .iloc[rank7_signals]
        .reset_index(drop=True),
        "exit_dates": pd.to_datetime(rank7_context["dates"])
        .iloc[np.minimum(exits, len(value_opens) - 1)]
        .to_numpy(),
        "width": rank7_context["matrix"][
            :, feature_columns.index("rex_2016_range_width_pct")
        ],
        "pullback": rank7_context["matrix"][
            :, feature_columns.index("htf_1d_range_pos")
        ],
    }
    learner = primitives.LearnerSpec(
        max_depth=int(RANK7_LEARNER["max_depth"]),
        min_samples_leaf=int(RANK7_LEARNER["min_samples_leaf"]),
        max_features=float(RANK7_LEARNER["max_features"]),
    )
    policy = primitives.SelectionSpec(
        funding_quantile=float(RANK7_SELECTION["funding_quantile"]),
        premium_quantile=float(RANK7_SELECTION["premium_quantile"]),
        risk_lambda=float(RANK7_SELECTION["risk_lambda"]),
        risk_quantile=float(RANK7_SELECTION["risk_quantile"]),
    )
    original_predict = primitives.deterministic_extra_trees_predict

    def counted_predict(model: Any, matrix: Any) -> Any:
        prediction = original_predict(model, matrix)
        values = np.asarray(matrix)
        rows_scored = 1 if values.ndim == 1 else int(values.shape[0])
        counters["rows_used"]["prediction_rows_scored"] += rows_scored
        return prediction

    primitives.deterministic_extra_trees_predict = counted_predict
    try:
        folds = primitives.fit_annual_rank7_folds(
            rank7_base,
            learner,
            start="2023-01-01",
            end="2026-06-02",
        )
    finally:
        primitives.deterministic_extra_trees_predict = original_predict
    active = primitives.rank7_activation(rank7_base, folds, policy)
    _rank7_bundle_activation_with_parity(
        rank7_runtime,
        bundle,
        rank7_context,
        active,
        counters,
        np,
        pd,
    )
    for split_name, mask in masks.items():
        next_allowed = 0
        for raw_signal in rank7_signals:
            signal_position = int(raw_signal)
            if not bool(mask[signal_position]):
                continue
            counters["per_sleeve"]["frozen_annual_rank7"][
                "signal_rows_evaluated"
            ] += 1
            if not bool(active[signal_position]):
                continue
            if signal_position < next_allowed:
                continue
            funding_leg = bool(rank7_context["funding_leg"][signal_position])
            hold, take, stop = (
                (576, 400, 1_000_000)
                if funding_leg
                else (144, 1_000_000, 300)
            )
            trade = rank7_structural_engine.trade_at(
                signal_position, 1, hold, take, stop
            )
            if trade is None:
                continue
            split_start, split_end = split_seconds[split_name]
            if not _structural_exit_is_eligible(
                mask,
                boundaries,
                int(trade.exit_position),
                str(trade.exit_kind),
                split_start,
                split_end,
            ):
                continue
            exit_boundary = int(trade.exit_position) + (
                0 if trade.exit_kind == "fixed" else 1
            )
            _append_direct_interval(
                intervals,
                counters,
                "frozen_annual_rank7",
                value_opens[int(trade.entry_position)],
                boundaries[exit_boundary],
                1,
                str(trade.exit_kind),
            )
            next_allowed = int(trade.exit_position) + 1

    causal_rows.validate()
    return _materialize_direct_rows(intervals, counters), counters

def _direct_generic_adapter(
    modules: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources = _source_paths(preregistration)
    counters = _empty_counters()
    history_path, _model_paths = _rank7_declared_bundle_paths(preregistration)
    counted_csv_sources = dict(sources)
    counted_csv_sources["rank7_hourly_history"] = history_path
    primitives = modules["training.gross9_structural_clock_primitives"]
    pandas_module = primitives.pd
    original_read_csv = _install_counted_csv_reader(
        pandas_module,
        root,
        counted_csv_sources,
        counters,
    )
    rank7_runtime = modules["execution.gross9_rank7_clock_runtime"]
    causal_rows = _CausalRowTracker(counters)
    try:
        original_load, original_predict, rank7_state = (
            _install_counted_rank7_runtime(
                rank7_runtime,
                root,
                preregistration,
                counters,
            )
        )
    except BaseException:
        pandas_module.read_csv = original_read_csv
        raise
    succeeded = False
    try:
        rows, counters = _direct_generic_adapter_impl(
            modules,
            preregistration,
            root,
            sources,
            counters,
            causal_rows,
        )
        succeeded = True
    finally:
        _restore_counted_rank7_runtime(
            rank7_runtime,
            original_load,
            original_predict,
        )
        pandas_module.read_csv = original_read_csv
    if succeeded:
        _validate_counted_rank7_runtime(rank7_state)
    return rows, counters


def _worker_main(
    arguments: argparse.Namespace,
    guard: _WorkerIsolationGuard,
) -> int:
    expected_parent_pid = arguments.expected_parent_pid
    capability_fd = arguments.worker_capability_fd
    ledger_fd = arguments.worker_ledger_fd
    if (
        type(capability_fd) is not int
        or type(ledger_fd) is not int
        or capability_fd == ledger_fd
    ):
        _fail("worker descriptor arguments differ")
    root = Path(arguments.repository_root).resolve(strict=True)
    if root != guard.root or guard.cwd != root:
        _fail("worker bootstrap repository root or cwd differs")
    own_stage_text = _worker_stage_path(root, Path(arguments.output_dir))
    other_stage_text = _worker_stage_path(
        root, Path(arguments.other_stage_directory)
    )
    if own_stage_text == other_stage_text:
        _fail("worker own and other stage paths are identical")
    if (
        guard.own_stage != root / own_stage_text
        or guard.other_stage != root / other_stage_text
    ):
        _fail("worker bootstrap stage strings differ")
    root = guard.root
    output_dir = guard.own_stage
    synthetic = bool(arguments.synthetic_input)
    synthetic_input_path: str | None = None
    if synthetic:
        candidate = Path(os.path.abspath(str(arguments.synthetic_input)))
        try:
            synthetic_input_path = candidate.relative_to(root).as_posix()
        except ValueError:
            _fail("synthetic worker input escapes the repository")
    token: bytearray | None = None
    capability_closed = False
    try:
        expected_worker_environment = prereg.worker_process_environment(root)
        if dict(os.environ) != expected_worker_environment:
            _fail("worker process environment differs from the exact mapping")
        if sys.dont_write_bytecode is not True:
            _fail("worker bytecode writing is not disabled")
        expected_prefix = (root / _PYCACHE_PREFIX_RELATIVE).as_posix()
        if sys.pycache_prefix != expected_prefix:
            _fail("worker bytecode-cache prefix differs")

        parent_authentication_raw = arguments.parent_auth_json.encode("ascii")
        try:
            parent_authentication = json.loads(
                parent_authentication_raw.decode("ascii"),
                object_pairs_hook=_unique_object,
            )
        except (UnicodeEncodeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TerminalG9CB12Failure(
                "worker parent authentication is invalid"
            ) from exc
        if (
            not isinstance(parent_authentication, dict)
            or parent_authentication_raw
            != _canonical_json_bytes(
                parent_authentication, trailing_lf=False
            )
        ):
            _fail("worker parent authentication bytes are not canonical")
        parent_authentication_sha256 = _sha256_bytes(
            parent_authentication_raw
        )
        expected_security_profile_raw = (
            arguments.expected_security_profile_json.encode("ascii")
        )
        try:
            expected_security_profile = json.loads(
                expected_security_profile_raw.decode("ascii"),
                object_pairs_hook=_unique_object,
            )
        except (UnicodeEncodeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TerminalG9CB12Failure(
                "worker expected security profile is invalid"
            ) from exc
        if (
            not isinstance(expected_security_profile, dict)
            or not expected_security_profile
            or expected_security_profile_raw
            != _canonical_json_bytes(
                expected_security_profile, trailing_lf=False
            )
        ):
            _fail("worker expected security profile bytes differ")

        active_metadata = _authenticate_worker_metadata_entry(
            root,
            parent_authentication,
            synthetic=synthetic,
            synthetic_input_path=synthetic_input_path,
            expected_security_profile=expected_security_profile,
        )
        sentinel = active_metadata["sentinel"]
        sentinel_raw = active_metadata["sentinel_raw"]
        claim = active_metadata["claim"]
        claim_raw = active_metadata["claim_raw"]
        preregistration = active_metadata["preregistration"]
        prereg_binding = active_metadata["preregistration_binding"]
        guarded_metadata = active_metadata["guarded_metadata"]
        metadata_snapshot = active_metadata["snapshot"]
        authority_amendments = _authority_amendment_bindings(
            preregistration,
            expected=expected_security_profile["authority_amendments"],
        )
        if claim.get("authority_amendments") != authority_amendments:
            _fail("worker claim amendment bindings differ")
        if (
            claim.get("preregistration") != prereg_binding
            or not isinstance(sentinel.get("claim_commit"), str)
        ):
            _fail("worker claim or sentinel binding differs")
        claim_binding = {
            "path": CLAIM_PATH.as_posix(),
            "sha256": _sha256_bytes(claim_raw),
            "claim_hash": claim["claim_hash"],
            "protocol_parent_commit": claim["protocol_parent_commit"],
            "claim_commit": sentinel["claim_commit"],
        }
        capabilities = _normalized_worker_capabilities(
            sentinel.get("worker_capabilities")
        )
        if sentinel != _sentinel_payload(
            claim_binding,
            prereg_binding,
            authority_amendments,
            parent_authentication_sha256,
            capabilities,
            expected_authority_amendments=expected_security_profile[
                "authority_amendments"
            ],
        ):
            _fail("worker sentinel contract differs")
        if sentinel.get("parent_authentication_sha256") != (
            parent_authentication_sha256
        ):
            _fail("worker parent authentication hash differs from sentinel")
        matches = [
            row
            for row in capabilities
            if row["stage_directory"] == own_stage_text
        ]
        if len(matches) != 1:
            _fail("worker stage is not bound to one capability slot")
        binding = matches[0]
        other_binding = capabilities[1 - (int(binding["slot"]) - 1)]
        if (
            binding["parent_pid"] != expected_parent_pid
            or other_binding["parent_pid"] != expected_parent_pid
            or other_binding["stage_directory"] != other_stage_text
        ):
            _fail("worker parent or cross-stage invocation binding differs")
        guard.bind_ledger_slot(int(binding["slot"]))
        ledger_info = os.fstat(ledger_fd)
        if (
            guard.ledger_fd != ledger_fd
            or (ledger_info.st_dev, ledger_info.st_ino)
            != (
                binding["ledger_device"],
                binding["ledger_inode"],
            )
            or stat.S_IMODE(ledger_info.st_mode) != 0o600
            or ledger_info.st_size != 0
        ):
            _fail("worker ledger descriptor does not match sentinel")

        guard.authorize_directory_sync(output_dir)
        assert guard.stage_fd is not None
        stage_info = os.fstat(guard.stage_fd)
        if (
            not stat.S_ISDIR(stage_info.st_mode)
            or stat.S_IMODE(stage_info.st_mode) != 0o700
            or _directory_entries(guard.stage_fd)
        ):
            _fail("worker own stage is not a fresh empty directory")
        if guard.counters() != {
            "child_process_creation_events": 0,
            "other_stage_access_events": 0,
            "other_stage_absence_checks": 1,
            "other_slot_ledger_access_events": 0,
            "unauthorized_write_or_ipc_events": 0,
        }:
            _fail("worker pre-capability isolation counters differ")
        try:
            guard._original_stat(
                Path(str(binding["consumed_ledger_path"])).name,
                dir_fd=guard.results_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError as exc:
            if exc.errno != errno.ENOENT:
                _fail("worker own-ledger absence check returned wrong errno")
        else:
            _fail("worker own consumption ledger already exists")

        if synthetic:
            if set(parent_authentication) != {
                "environment",
                "hashed_inputs",
                "preregistration_authentication",
                "runtime_import_closure",
            }:
                _fail("synthetic parent authentication schema differs")
            if (
                not isinstance(parent_authentication["hashed_inputs"], list)
                or not isinstance(
                    parent_authentication["runtime_import_closure"], list
                )
                or not isinstance(
                    parent_authentication["environment"], Mapping
                )
                or parent_authentication["environment"].get(
                    "worker_process_environment"
                )
                != expected_worker_environment
            ):
                _fail("synthetic parent authentication bindings differ")
            worker_closures = {
                "runtime": parent_authentication["runtime_import_closure"]
            }
            worker_authentication = dict(parent_authentication)
        else:
            if guarded_metadata is None:
                _fail("guarded worker metadata authentication is absent")
            worker_closures = guarded_metadata["closures"]
            worker_authentication = guarded_metadata["authentication"]
        if worker_authentication != parent_authentication:
            _fail("worker authentication differs from the parent seal")

        token = _consume_worker_capability(capability_fd, binding)
        capability_closed = True
        ledger = _publish_worker_ledger(
            guard=guard,
            snapshot=metadata_snapshot,
            ledger_fd=ledger_fd,
            binding=binding,
            claim=claim,
            preregistration=preregistration,
            sentinel=sentinel,
            authority_amendments=authority_amendments,
        )
        _worker_ledger_linked_checkpoint(guard, binding, ledger)

        rebuild_invocations_started = 0
        rebuild_invocations_completed = 0
        recorder: _RuntimeImportRecorder | None = None
        modules: dict[str, Any] | None = None
        if not synthetic:
            _preload_runtime_package_initializers(root, preregistration)
            recorder = _RuntimeImportRecorder(
                root=root,
                preregistration=preregistration,
                runtime_closure=worker_closures["runtime"],
            )
            recorder.install()
            modules = _import_authenticated_modules(root.as_posix())

        rebuild_invocations_started += 1
        if synthetic:
            assert synthetic_input_path is not None
            bars, counters, synthetic_domain_end = (
                _read_synthetic_worker_input(
                    root / synthetic_input_path,
                    path_text=synthetic_input_path,
                    raw_cache=active_metadata["raw_cache"],
                )
            )
            rows, counters = reconstruct_intervals(
                bars,
                counters=counters,
                domain_end=synthetic_domain_end,
            )
        else:
            if modules is None:
                _fail("official isolated modules were not imported")
            rows, counters = _direct_generic_adapter(
                modules,
                preregistration,
                root,
            )
        csv_raw = compress_csv(serialize_csv(rows))
        if recorder is None:
            imported_paths: list[str] = []
            preloaded_paths: list[str] = []
        else:
            imported_paths = recorder.freeze()
            preloaded_paths = sorted(recorder.preloaded_repository_paths)
            counters["file_access"]["runtime_modules_imported"] = len(
                imported_paths
            )
        provenance = (
            {
                "synthetic_test_only": True,
                "worker_process_environment": expected_worker_environment,
                "preloaded_repository_paths": preloaded_paths,
                "runtime_import_paths": imported_paths,
            }
            if synthetic
            else {
                "preregistration": prereg_binding,
                "runtime_import_modules": list(RUNTIME_IMPORT_MODULES),
                "worker_process_environment": expected_worker_environment,
                "preloaded_repository_paths": preloaded_paths,
                "runtime_import_paths": imported_paths,
            }
        )
        sentinel_binding = {
            "path": SENTINEL_PATH.as_posix(),
            "sha256": _sha256_bytes(sentinel_raw),
            "manifest_hash": sentinel["manifest_hash"],
        }
        core = build_core(
            csv_raw,
            counters,
            provenance,
            claim_binding,
            sentinel_binding,
            authority_amendments,
            parent_authentication,
            expected_authority_amendments=expected_security_profile[
                "authority_amendments"
            ],
        )
        core_raw = _canonical_json_bytes(core)
        rebuild_invocations_completed += 1

        if _directory_entries(guard.stage_fd):
            _fail("worker staging directory changed before output writing")
        ledger_info = os.fstat(ledger["canonical_fd"])
        if (
            (ledger_info.st_dev, ledger_info.st_ino)
            != (ledger["device"], ledger["inode"])
            or stat.S_IMODE(ledger_info.st_mode) != 0o444
        ):
            _fail("worker consumption ledger changed before output writing")

        guard.authorize_outputs()
        csv_path = output_dir / _STAGED_CSV_NAME
        core_path = output_dir / _STAGED_CORE_NAME
        receipt_path = output_dir / _STAGED_RECEIPT_NAME
        _write_exclusive_guarded_file(
            csv_path,
            csv_raw,
            mode=0o400,
            sync_directory=output_dir,
        )
        _write_exclusive_guarded_file(
            core_path,
            core_raw,
            mode=0o400,
            sync_directory=output_dir,
        )
        observed_csv = csv_raw
        observed_core = core_raw
        receipt = _worker_receipt_payload(
            binding=binding,
            worker_pid=os.getpid(),
            ledger_sha256=ledger["sha256"],
            rebuild_invocations_started=rebuild_invocations_started,
            rebuild_invocations_completed=rebuild_invocations_completed,
            guard_counters=guard.counters(),
            csv_gzip_sha256=_sha256_bytes(observed_csv),
            per_pass_core_sha256=_sha256_bytes(observed_core),
            token=token,
        )
        receipt_raw = _canonical_json_bytes(receipt)
        guard.authorize_receipt()
        _write_exclusive_guarded_file(
            receipt_path,
            receipt_raw,
            mode=0o400,
            sync_directory=output_dir,
        )
        return 0
    finally:
        if not capability_closed:
            try:
                os.close(capability_fd)
            except OSError:
                pass
        if token is not None:
            _zero_token(token)
        metadata = locals().get("metadata_snapshot")
        if isinstance(metadata, _SecureBoundSnapshot):
            metadata.close()
        published_ledger = locals().get("ledger")
        if isinstance(published_ledger, Mapping):
            canonical_descriptor = published_ledger.get("canonical_fd")
            if isinstance(canonical_descriptor, int):
                try:
                    os.close(canonical_descriptor)
                except OSError:
                    pass
        try:
            os.close(ledger_fd)
        except OSError:
            pass
        if guard.stage_fd is not None:
            try:
                os.close(guard.stage_fd)
            except OSError:
                pass
        for descriptor in (
            guard.filesystem_root_fd,
            guard.results_fd,
            guard.repository_fd,
        ):
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


def _prepare_worker(
    *,
    root: Path,
    capability: dict[str, Any],
    other_stage_directory: str,
    synthetic_input: Path | None,
    parent_authentication: Mapping[str, Any],
    expected_security_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = capability["row"]
    if "ledger_fd" not in capability:
        _fail("prepared worker ledger descriptor is absent")
    ledger_fd = int(capability["ledger_fd"])
    capability_fd = int(capability["read_fd"])
    if capability_fd < 0 or ledger_fd < 0 or capability_fd == ledger_fd:
        _fail("prepared worker descriptors are not distinct")
    parent_auth_json = _canonical_json_bytes(
        dict(parent_authentication), trailing_lf=False
    ).decode("ascii")
    expected_security_profile_json = _canonical_json_bytes(
        dict(expected_security_profile or {}), trailing_lf=False
    ).decode("ascii")
    command = [
        sys.executable,
        "-B",
        str(_rooted(root, BUILDER_PATH)),
        "--internal-worker",
        "--repository-root",
        root.resolve().as_posix(),
        "--output-dir",
        str(row["stage_directory"]),
        "--other-stage-directory",
        other_stage_directory,
        "--worker-capability-fd",
        str(capability_fd),
        "--worker-ledger-fd",
        str(ledger_fd),
        "--expected-parent-pid",
        str(row["parent_pid"]),
        "--parent-auth-json",
        parent_auth_json,
        "--expected-security-profile-json",
        expected_security_profile_json,
    ]
    if synthetic_input is not None:
        command.extend(["--synthetic-input", str(synthetic_input)])
    environment = prereg.worker_process_environment(root)
    if set(environment) != set(prereg.WORKER_PROCESS_ENVIRONMENT):
        _fail("prepared worker environment names differ")
    return {
        "command": command,
        "environment": environment,
        "cwd": root.resolve(),
        "capability": capability,
    }


def _execute_prepared_worker(invocation: dict[str, Any]) -> int:
    capability = invocation["capability"]
    if "ledger_fd" not in capability:
        _fail("worker ledger descriptor is absent")
    capability_read_fd = int(capability["read_fd"])
    ledger_fd = int(capability["ledger_fd"])
    if capability_read_fd < 0 or ledger_fd < 0 or capability_read_fd == ledger_fd:
        _fail("worker capability and ledger descriptors are not distinct")
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            invocation["command"],
            cwd=invocation["cwd"],
            env=invocation["environment"],
            close_fds=True,
            pass_fds=(capability_read_fd, ledger_fd),
        )
        try:
            os.close(capability_read_fd)
        except BaseException:
            process.terminate()
            process.wait()
            raise
        capability["read_fd"] = -1
        observation = invocation.get("ledger_observation")
        if observation is not None:
            if not isinstance(observation, Mapping):
                _fail("worker ledger observation contract differs")
            context = observation.get("context")
            stage = observation.get("stage")
            ledger = observation.get("ledger")
            advance = observation.get("advance")
            if (
                not isinstance(context, _PublicationContext)
                or not isinstance(stage, Path)
                or not isinstance(ledger, Path)
                or not callable(advance)
            ):
                _fail("worker ledger observation binding differs")
            stage_fd = context.stage_descriptors.get(stage.name)
            if stage_fd is None:
                _fail("worker ledger observation stage is absent")
            while True:
                stage_entries = _directory_entries(stage_fd)
                try:
                    ledger_info = os.stat(
                        ledger.name,
                        dir_fd=context.results_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    ledger_info = None
                if ledger_info is not None:
                    if stage_entries:
                        _fail(
                            "worker outputs appeared before parent ledger "
                            "checkpoint"
                        )
                    if (
                        not stat.S_ISREG(ledger_info.st_mode)
                        or stat.S_IMODE(ledger_info.st_mode) != 0o444
                    ):
                        _fail("observed worker ledger mode/type differs")
                    expected_entries = tuple(
                        sorted((*context.entries, ledger.name))
                    )
                    if _directory_entries(context.results_fd) != expected_entries:
                        _fail("worker ledger observation delta differs")
                    os.fsync(context.results_fd)
                    context._rebaseline(expected_entries)
                    if process.poll() is not None:
                        _fail("worker exited before ledger-linked checkpoint")
                    advance()
                    break
                if stage_entries:
                    _fail("worker outputs preceded canonical ledger link")
                if process.poll() is not None:
                    _fail("worker exited before canonical ledger link")
                os.sched_yield()
        return_code = process.wait()
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        raise
    if return_code != 0:
        _fail(
            f"fresh worker failed with PID {process.pid} and status {return_code}"
        )
    return int(process.pid)


def _validate_counter_contract(counters: Any) -> None:
    template = _empty_counters()
    if not isinstance(counters, Mapping) or set(counters) != set(template):
        _fail("worker counter sections differ")
    for section in ("file_access", "rows_decoded", "rows_used"):
        value = counters.get(section)
        if not isinstance(value, Mapping) or set(value) != set(template[section]):
            _fail(f"worker {section} counter names differ")
    per_sleeve = counters.get("per_sleeve")
    if not isinstance(per_sleeve, Mapping) or set(per_sleeve) != set(
        template["per_sleeve"]
    ):
        _fail("worker per-sleeve counter names differ")
    for sleeve, expected in template["per_sleeve"].items():
        value = per_sleeve.get(sleeve)
        if not isinstance(value, Mapping) or set(value) != set(expected):
            _fail(f"worker {sleeve} counter names differ")

    maps = (
        counters["file_access"]["bytes_read_by_logical_source"],
        counters["rows_used"]["causal_feature_rows_by_source"],
    )
    source_names = set(template["rows_decoded"])
    for value in maps:
        if not isinstance(value, Mapping) or set(value) != source_names:
            _fail("worker source counter map differs")
    for name in source_names:
        if (
            counters["rows_used"]["causal_feature_rows_by_source"][name]
            > counters["rows_decoded"][name]
        ):
            _fail(f"worker causal rows exceed decoded rows: {name}")

    def check(value: Any) -> None:
        if isinstance(value, Mapping):
            for nested in value.values():
                check(nested)
            return
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            _fail("worker counter is not a nonnegative integer")

    check(counters)


def _validate_counter_consistency(
    rows: Sequence[Mapping[str, Any]],
    counters: Mapping[str, Any],
    *,
    synthetic: bool,
    authenticated_import_path_count: int = 0,
) -> None:
    imported = counters["file_access"]["runtime_modules_imported"]
    if synthetic and imported != 0:
        _fail("worker runtime import counter differs")
    if not synthetic and not (
        len(GENERIC_RUNTIME_MODULES)
        <= imported
        <= authenticated_import_path_count
    ):
        _fail("worker runtime import counter differs")
    total_examined = sum(
        counters["per_sleeve"][sleeve["name"]][
            "outcome_dependent_ohlc_rows_examined"
        ]
        for sleeve in SLEEVES
    )
    if (
        counters["rows_used"]["outcome_dependent_ohlc_rows_examined"]
        != total_examined
    ):
        _fail("worker OHLC examination counters differ")
    rank7_rows = counters["rows_used"]
    label_count = rank7_rows["rank7_training_trades_replayed"]
    label_counters = (
        "rank7_net_labels_computed",
        "rank7_adverse_labels_computed",
        "rank7_price_factor_values_used",
        "rank7_funding_factor_values_used",
        "rank7_funding_debit_factor_values_used",
        "rank7_adverse_price_factor_values_used",
        "rank7_fee_factor_values_used",
    )
    if any(rank7_rows[name] != label_count for name in label_counters):
        _fail("worker Rank7 exact-label counters differ")
    if synthetic:
        if (
            counters["file_access"]["model_files_opened"] != 0
            or counters["rows_decoded"]["rank7_hourly_history"] != 0
            or any(rank7_rows[name] != 0 for name in RANK7_ROWS_USED_COUNTERS)
        ):
            _fail("synthetic worker reports Rank7 bundle access")
    else:
        bundle_scored = rank7_rows["rank7_bundle_activation_rows_scored"]
        parity_rows = rank7_rows["rank7_bundle_parity_rows_compared"]
        if (
            counters["file_access"]["model_files_opened"] != 5
            or counters["rows_decoded"]["rank7_hourly_history"] <= 0
            or label_count <= 0
            or bundle_scored <= 0
            or parity_rows < bundle_scored
            or rank7_rows["prediction_rows_scored"] < bundle_scored * 5
        ):
            _fail("worker Rank7 bundle or parity counters differ")

    for sleeve in SLEEVES:
        name = sleeve["name"]
        selected = [row for row in rows if row["sleeve"] == name]
        per = counters["per_sleeve"][name]
        long_count = sum(row["side"] == "1" for row in selected)
        short_count = sum(row["side"] == "-1" for row in selected)
        if (
            per["intervals_emitted"] != len(selected)
            or per["long_intervals"] != long_count
            or per["short_intervals"] != short_count
            or per["long_intervals"] + per["short_intervals"] != len(selected)
            or per["fixed_horizon_exits"]
            + per["take_exits"]
            + per["stop_exits"]
            != len(selected)
        ):
            _fail(f"worker interval counters differ for {name}")
        if sleeve["kind"] == "fixed" and (
            per["fixed_horizon_exits"] != len(selected)
            or per["take_exits"] != 0
            or per["stop_exits"] != 0
        ):
            _fail(f"worker fixed-exit counters differ for {name}")


def _validate_worker_provenance(
    provenance: Any,
    *,
    synthetic: bool,
    prereg_binding: Mapping[str, Any],
    parent_authentication: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(provenance, Mapping):
        _fail("worker provenance is absent")
    expected_environment = parent_authentication["environment"][
        "worker_process_environment"
    ]
    expected_keys = (
        {
            "synthetic_test_only",
            "worker_process_environment",
            "preloaded_repository_paths",
            "runtime_import_paths",
        }
        if synthetic
        else {
            "preregistration",
            "runtime_import_modules",
            "worker_process_environment",
            "preloaded_repository_paths",
            "runtime_import_paths",
        }
    )
    if set(provenance) != expected_keys:
        _fail("worker provenance schema differs")
    if provenance.get("worker_process_environment") != expected_environment:
        _fail("worker provenance environment differs")
    runtime_paths = provenance.get("runtime_import_paths")
    preloaded_paths = provenance.get("preloaded_repository_paths")
    if (
        not isinstance(runtime_paths, list)
        or runtime_paths != sorted(set(runtime_paths))
        or not all(isinstance(path, str) for path in runtime_paths)
        or not isinstance(preloaded_paths, list)
        or preloaded_paths != sorted(set(preloaded_paths))
        or not all(isinstance(path, str) for path in preloaded_paths)
    ):
        _fail("worker import provenance paths differ")
    if synthetic:
        if provenance.get("synthetic_test_only") is not True or runtime_paths:
            _fail("synthetic worker provenance differs")
    else:
        if (
            provenance.get("preregistration") != dict(prereg_binding)
            or provenance.get("runtime_import_modules")
            != list(RUNTIME_IMPORT_MODULES)
        ):
            _fail("official worker provenance bindings differ")
        closure_paths = {
            str(row["path"])
            for row in parent_authentication["runtime_import_closure"]
        }
        required_roots = {
            f"{module.replace('.', '/')}.py"
            for module in RUNTIME_IMPORT_MODULES
        }
        if (
            not required_roots.issubset(runtime_paths)
            or not set(runtime_paths).issubset(closure_paths)
            or required_roots & set(preloaded_paths)
        ):
            _fail("official worker runtime execution inventory differs")
    return dict(provenance)


def _validate_worker_ledger_and_receipt(
    *,
    root: Path,
    capability: dict[str, Any],
    observed_worker_pid: int,
    claim: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    sentinel: Mapping[str, Any],
    authority_amendments: Sequence[Mapping[str, Any]],
    publication_context: _PublicationContext | None = None,
) -> dict[str, Any]:
    binding = capability["row"]
    stage = capability["stage_path"]
    retained_fd = int(capability["ledger_fd"])
    ledger_info = os.fstat(retained_fd)
    ledger_chunks: list[bytes] = []
    ledger_offset = 0
    while ledger_offset < ledger_info.st_size:
        chunk = os.pread(
            retained_fd,
            min(1024 * 1024, ledger_info.st_size - ledger_offset),
            ledger_offset,
        )
        if not chunk:
            break
        ledger_chunks.append(chunk)
        ledger_offset += len(chunk)
    ledger_raw = b"".join(ledger_chunks)
    if (
        stat.S_IMODE(ledger_info.st_mode) != 0o444
        or (ledger_info.st_dev, ledger_info.st_ino)
        != (binding["ledger_device"], binding["ledger_inode"])
    ):
        _fail("worker consumption ledger mode differs")
    owned_context = publication_context is None
    if publication_context is None:
        publication_context = _PublicationContext(root)
    canonical_fd = os.open(
        Path(str(binding["consumed_ledger_path"])).name,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        dir_fd=publication_context.results_fd,
    )
    try:
        canonical_info = os.fstat(canonical_fd)
        if (
            (canonical_info.st_dev, canonical_info.st_ino)
            != (ledger_info.st_dev, ledger_info.st_ino)
            or _pread_complete(canonical_fd, canonical_info.st_size)
            != ledger_raw
        ):
            _fail("parent retained worker-ledger validation differs")
    finally:
        os.close(canonical_fd)
        if owned_context:
            publication_context.close()
    if not owned_context and Path(
        str(binding["consumed_ledger_path"])
    ).name not in publication_context.entries:
        _fail("parent ledger checkpoint was not advanced at worker time")
    try:
        ledger_payload = json.loads(
            ledger_raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TerminalG9CB12Failure(
            "worker consumption ledger is invalid"
        ) from exc
    expected_ledger = _worker_ledger_payload(
        binding=binding,
        claim=claim,
        preregistration=preregistration,
        sentinel=sentinel,
        authority_amendments=authority_amendments,
    )
    if (
        ledger_payload != expected_ledger
        or ledger_raw != _canonical_json_bytes(expected_ledger)
    ):
        _fail("worker consumption ledger bytes differ")

    stage_fd = publication_context.stage_descriptors.get(stage.name)
    if stage_fd is None:
        _fail("worker stage descriptor is absent after successful exit")
    expected_names = {
        _STAGED_CSV_NAME,
        _STAGED_CORE_NAME,
        _STAGED_RECEIPT_NAME,
    }
    if set(_directory_entries(stage_fd)) != expected_names:
        _fail("worker stage output inventory differs")

    def read_stage_leaf(name: str) -> tuple[bytes, os.stat_result]:
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=stage_fd,
        )
        try:
            before = os.fstat(descriptor)
            raw = _pread_complete(descriptor, before.st_size)
            after = os.fstat(descriptor)
            if (
                _descriptor_token(before) != _descriptor_token(after)
                or not stat.S_ISREG(after.st_mode)
                or stat.S_IMODE(after.st_mode) != 0o400
            ):
                _fail(f"worker staged file differs: {name}")
            return raw, after
        finally:
            os.close(descriptor)

    csv_raw, _ = read_stage_leaf(_STAGED_CSV_NAME)
    core_raw, _ = read_stage_leaf(_STAGED_CORE_NAME)
    receipt_raw, receipt_info = read_stage_leaf(_STAGED_RECEIPT_NAME)
    try:
        receipt = json.loads(
            receipt_raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TerminalG9CB12Failure("worker receipt is invalid") from exc
    if (
        not isinstance(receipt, dict)
        or receipt_raw != _canonical_json_bytes(receipt)
    ):
        _fail("worker receipt bytes are not canonical")
    expected_receipt = _worker_receipt_payload(
        binding=binding,
        worker_pid=observed_worker_pid,
        ledger_sha256=_sha256_bytes(ledger_raw),
        rebuild_invocations_started=1,
        rebuild_invocations_completed=1,
        guard_counters={
            "child_process_creation_events": 0,
            "other_stage_access_events": 0,
            "other_stage_absence_checks": 1,
            "other_slot_ledger_access_events": 0,
            "unauthorized_write_or_ipc_events": 0,
        },
        csv_gzip_sha256=_sha256_bytes(csv_raw),
        per_pass_core_sha256=_sha256_bytes(core_raw),
        token=capability["token"],
    )
    if receipt != expected_receipt:
        _fail("worker receipt authentication differs")
    if (
        observed_worker_pid <= 0
        or observed_worker_pid == binding["parent_pid"]
        or receipt["worker_pid"] != observed_worker_pid
        or receipt["stage_directory"] != binding["stage_directory"]
    ):
        _fail("worker receipt PID or stage binding differs")
    os.close(retained_fd)
    capability["ledger_fd"] = -1
    return {
        "csv_raw": csv_raw,
        "core_raw": core_raw,
        "receipt_raw": receipt_raw,
        "receipt": receipt,
        "receipt_info": receipt_info,
        "ledger_raw": ledger_raw,
        "ledger_payload": ledger_payload,
        "consumption": {
            "slot": binding["slot"],
            "parent_pid": binding["parent_pid"],
            "path": binding["consumed_ledger_path"],
            "sha256": _sha256_bytes(ledger_raw),
            "carrier_kind": binding["carrier_kind"],
            "carrier_device": binding["carrier_device"],
            "carrier_inode": binding["carrier_inode"],
            "token_sha256": binding["token_sha256"],
        },
        "rebuild_receipt": {
            **receipt,
            "pass_receipt_sha256": _sha256_bytes(receipt_raw),
        },
    }


def _cleanup_successful_stage(
    stage: Path,
    results_directory: Path,
    *,
    publication_context: _PublicationContext | None = None,
) -> None:
    owned_context = publication_context is None
    context = publication_context or _PublicationContext(
        results_directory, directory_relative="."
    )
    stage_fd = context.stage_descriptors.get(stage.name, -1)
    if stage_fd < 0:
        _fail("worker stage cleanup lacks its retained descriptor")
    try:
        remaining = list(
            (_STAGED_CSV_NAME, _STAGED_CORE_NAME, _STAGED_RECEIPT_NAME)
        )
        if _descriptor_token(os.fstat(stage_fd)) != context.stage_tokens[
            stage.name
        ]:
            context.require_stage(
                stage,
                tuple(sorted(remaining)),
                authorized_delta=True,
            )
        for name in tuple(remaining):
            context.require_stage(stage, tuple(sorted(remaining)))
            info = os.stat(name, dir_fd=stage_fd, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                _fail(f"worker stage cleanup target differs: {stage / name}")
            os.unlink(name, dir_fd=stage_fd)
            remaining.remove(name)
            if _directory_entries(stage_fd) != tuple(sorted(remaining)):
                _fail("worker stage cleanup one-leaf delta differs")
            os.fsync(stage_fd)
            context.require_stage(
                stage,
                tuple(sorted(remaining)),
                authorized_delta=True,
            )
        if _directory_entries(stage_fd):
            _fail("worker stage is not empty before rmdir")
        context.require_stage(stage, ())
        os.fsync(stage_fd)
    finally:
        os.close(stage_fd)
    baseline = context.entries
    context._require_bound_results()
    os.rmdir(stage.name, dir_fd=context.results_fd)
    expected = tuple(name for name in baseline if name != stage.name)
    if _directory_entries(context.results_fd) != expected:
        _fail("worker stage rmdir one-leaf delta differs")
    os.fsync(context.results_fd)
    context._rebaseline(expected)
    del context.stage_descriptors[stage.name]
    del context.stage_tokens[stage.name]
    del context.stage_entries[stage.name]
    if owned_context:
        context.close()


def _create_stage_directory(
    stage: Path,
    results_directory: Path,
    *,
    publication_context: _PublicationContext | None = None,
) -> None:
    owned_context = publication_context is None
    context = publication_context or _PublicationContext(
        results_directory, directory_relative="."
    )
    context._require_bound_results()
    if stage.name in context.entries:
        _fail(f"reserved worker stage already exists: {stage}")
    os.mkdir(stage.name, 0o700, dir_fd=context.results_fd)
    stage_fd = os.open(
        stage.name,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        dir_fd=context.results_fd,
    )
    try:
        info = os.fstat(stage_fd)
        if (
            not stat.S_ISDIR(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o700
            or _directory_entries(stage_fd)
        ):
            _fail("worker stage directory mode or inventory differs")
    except BaseException:
        os.close(stage_fd)
        raise
    context.stage_descriptors[stage.name] = stage_fd
    context.stage_tokens[stage.name] = _descriptor_token(info)
    context.stage_entries[stage.name] = ()
    expected = tuple(sorted((*context.entries, stage.name)))
    if _directory_entries(context.results_fd) != expected:
        _fail("worker stage mkdir one-leaf delta differs")
    os.fsync(context.results_fd)
    context._rebaseline(expected)
    if owned_context:
        context.close()


def _validate_worker_product(
    csv_raw: bytes,
    core_raw: bytes,
    *,
    synthetic: bool,
    prereg_binding: Mapping[str, Any],
    parent_authentication: Mapping[str, Any],
    claim_binding: Mapping[str, Any],
    sentinel_binding: Mapping[str, Any],
    authority_amendments: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = validate_csv_gzip(csv_raw, require_all_sleeves=True)
    try:
        core = json.loads(
            core_raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TerminalG9CB12Failure("worker core is invalid JSON") from exc
    if not isinstance(core, dict) or core_raw != _canonical_json_bytes(core):
        _fail("worker core bytes are not canonical")
    _validate_prohibited_output_placement(core)
    if core.get("authority_amendments") != [
        dict(row) for row in authority_amendments
    ]:
        _fail("worker core authority amendment bindings differ")
    expected_parent_hash = _sha256_bytes(
        _canonical_json_bytes(
            dict(parent_authentication), trailing_lf=False
        )
    )
    if (
        core.get("parent_authentication") != dict(parent_authentication)
        or core.get("parent_authentication_sha256") != expected_parent_hash
    ):
        _fail("worker core parent authentication differs")
    expected_provenance = _validate_worker_provenance(
        core.get("provenance"),
        synthetic=synthetic,
        prereg_binding=prereg_binding,
        parent_authentication=parent_authentication,
    )
    counters = core.get("access_counters")
    _validate_counter_contract(counters)
    authenticated_paths = {
        str(row.get("path"))
        for row in parent_authentication.get("runtime_import_closure", [])
        if isinstance(row, Mapping) and isinstance(row.get("path"), str)
    }
    _validate_counter_consistency(
        rows,
        counters,
        synthetic=synthetic,
        authenticated_import_path_count=len(authenticated_paths),
    )
    expected = build_core(
        csv_raw,
        counters,
        expected_provenance,
        claim_binding,
        sentinel_binding,
        authority_amendments,
        parent_authentication,
        expected_authority_amendments=authority_amendments,
    )
    if core != expected:
        _fail("worker core contract or authentication differs")
    return rows, core


def _final_manifest(
    core: Mapping[str, Any],
    worker_capability_consumption: Sequence[Mapping[str, Any]],
    rebuild_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _validate_prohibited_output_placement(core)
    consumption = [dict(row) for row in worker_capability_consumption]
    receipts = [dict(row) for row in rebuild_receipts]
    if (
        len(consumption) != 2
        or [row.get("slot") for row in consumption] != [1, 2]
        or any(len(row) != 8 for row in consumption)
        or len(receipts) != 2
        or [row.get("slot") for row in receipts] != [1, 2]
        or any(len(row) != 20 for row in receipts)
    ):
        _fail("final worker evidence lists differ")
    body = {key: value for key, value in core.items() if key != "manifest_hash"}
    body.update(
        {
            "status": "published_manifest_last",
            "one_shot": True,
            "retry_allowed": False,
            "resume_allowed": False,
            "worker_capability_consumption": consumption,
            "rebuild_receipts": receipts,
        }
    )
    manifest = _with_hash(body, "manifest_hash")
    _validate_prohibited_output_placement(manifest)
    return manifest


def _validate_final_manifest_contract(
    manifest: Mapping[str, Any],
    *,
    csv_raw: bytes,
    rows: Sequence[Mapping[str, Any]],
    synthetic: bool,
    prereg_binding: Mapping[str, Any],
    claim_binding: Mapping[str, Any],
    sentinel_binding: Mapping[str, Any],
    authority_amendments: Sequence[Mapping[str, Any]],
    capabilities: Sequence[Mapping[str, Any]],
    expected_consumption: Sequence[Mapping[str, Any]],
    ledger_hashes: Mapping[int, str],
    expected_parent_authentication: Mapping[str, Any],
    sentinel_parent_authentication_sha256: str,
) -> dict[str, Any]:
    _validate_prohibited_output_placement(manifest)
    if (
        manifest.get("identity") != IDENTITY
        or manifest.get("protocol_version") != PROTOCOL_VERSION
        or manifest.get("status") != "published_manifest_last"
        or manifest.get("one_shot") is not True
        or manifest.get("retry_allowed") is not False
        or manifest.get("resume_allowed") is not False
    ):
        _fail("committed final manifest status differs")
    if (
        manifest.get("authority_amendments")
        != [dict(row) for row in authority_amendments]
        or manifest.get("access_claim") != dict(claim_binding)
        or manifest.get("attempt_consumed") != dict(sentinel_binding)
        or manifest.get("worker_capability_consumption")
        != [dict(row) for row in expected_consumption]
    ):
        _fail("committed final manifest authority bindings differ")

    csv_decompressed = gzip.decompress(csv_raw)
    if (
        manifest.get("csv_schema") != list(CSV_COLUMNS)
        or manifest.get("csv_byte_length") != len(csv_decompressed)
        or manifest.get("csv_sha256")
        != _sha256_bytes(csv_decompressed)
        or manifest.get("csv_gzip_byte_length") != len(csv_raw)
        or manifest.get("csv_gzip_sha256") != _sha256_bytes(csv_raw)
    ):
        _fail("committed final manifest CSV binding differs")

    final_only = {
        "manifest_hash",
        "status",
        "one_shot",
        "retry_allowed",
        "resume_allowed",
        "worker_capability_consumption",
        "rebuild_receipts",
    }
    core = _with_hash(
        {
            key: value
            for key, value in manifest.items()
            if key not in final_only
        },
        "manifest_hash",
    )
    parent_authentication = core.get("parent_authentication")
    counters = core.get("access_counters")
    if not isinstance(parent_authentication, Mapping):
        _fail("committed core parent authentication is absent")
    if dict(parent_authentication) != dict(expected_parent_authentication):
        _fail("committed core parent authentication binding differs")
    parent_hash = _sha256_bytes(
        _canonical_json_bytes(
            parent_authentication,
            trailing_lf=False,
        )
    )
    if (
        core.get("parent_authentication_sha256") != parent_hash
        or parent_hash != sentinel_parent_authentication_sha256
    ):
        _fail("committed core parent authentication hash differs")
    _validate_counter_contract(counters)
    runtime_closure = parent_authentication.get(
        "runtime_import_closure"
    )
    if not isinstance(runtime_closure, list):
        _fail("committed core runtime closure is absent")
    _validate_counter_consistency(
        rows,
        counters,
        synthetic=synthetic,
        authenticated_import_path_count=len(runtime_closure),
    )
    provenance = _validate_worker_provenance(
        core.get("provenance"),
        synthetic=synthetic,
        prereg_binding=prereg_binding,
        parent_authentication=parent_authentication,
    )
    expected_core = build_core(
        csv_raw,
        counters,
        provenance,
        claim_binding,
        sentinel_binding,
        authority_amendments,
        parent_authentication,
        expected_authority_amendments=authority_amendments,
    )
    if core != expected_core:
        _fail("committed per-pass core contract differs")
    core_sha256 = _sha256_bytes(_canonical_json_bytes(expected_core))

    receipts = manifest.get("rebuild_receipts")
    expected_receipt_keys = {
        "identity",
        "protocol_version",
        "slot",
        "parent_pid",
        "worker_pid",
        "stage_directory",
        "consumed_ledger_path",
        "consumed_ledger_sha256",
        "rebuild_invocations_started",
        "rebuild_invocations_completed",
        "child_process_creation_events",
        "other_stage_access_events",
        "other_stage_absence_checks",
        "other_slot_ledger_access_events",
        "unauthorized_write_or_ipc_events",
        "csv_gzip_sha256",
        "per_pass_core_sha256",
        "completion_hmac_sha256",
        "receipt_hash",
        "pass_receipt_sha256",
    }
    if (
        not isinstance(receipts, list)
        or len(receipts) != 2
        or [row.get("slot") for row in receipts if isinstance(row, Mapping)]
        != [1, 2]
    ):
        _fail("committed final manifest receipts differ")
    worker_pids: set[int] = set()
    for row, capability in zip(receipts, capabilities, strict=True):
        if not isinstance(row, Mapping) or set(row) != expected_receipt_keys:
            _fail("committed receipt schema differs")
        slot = row.get("slot")
        worker_pid = row.get("worker_pid")
        if (
            type(slot) is not int
            or type(worker_pid) is not int
            or worker_pid <= 0
            or worker_pid == capability.get("parent_pid")
            or worker_pid in worker_pids
        ):
            _fail("committed receipt PID or slot differs")
        worker_pids.add(worker_pid)
        exact_fields = {
            "identity": IDENTITY,
            "protocol_version": PROTOCOL_VERSION,
            "slot": capability["slot"],
            "parent_pid": capability["parent_pid"],
            "stage_directory": capability["stage_directory"],
            "consumed_ledger_path": capability[
                "consumed_ledger_path"
            ],
            "consumed_ledger_sha256": ledger_hashes[slot],
            "rebuild_invocations_started": 1,
            "rebuild_invocations_completed": 1,
            "child_process_creation_events": 0,
            "other_stage_access_events": 0,
            "other_stage_absence_checks": 1,
            "other_slot_ledger_access_events": 0,
            "unauthorized_write_or_ipc_events": 0,
            "csv_gzip_sha256": _sha256_bytes(csv_raw),
            "per_pass_core_sha256": core_sha256,
        }
        if any(row.get(key) != value for key, value in exact_fields.items()):
            _fail("committed receipt binding differs")
        completion_hmac = row.get("completion_hmac_sha256")
        if not isinstance(
            completion_hmac, str
        ) or not _SHA_RE.fullmatch(completion_hmac):
            _fail("committed receipt HMAC shape differs")
        receipt = dict(row)
        pass_receipt_sha256 = receipt.pop(
            "pass_receipt_sha256", None
        )
        if (
            not isinstance(pass_receipt_sha256, str)
            or not _SHA_RE.fullmatch(pass_receipt_sha256)
            or pass_receipt_sha256
            != _sha256_bytes(_canonical_json_bytes(receipt))
        ):
            _fail("committed pass receipt hash differs")
        receipt_hash = receipt.pop("receipt_hash", None)
        if (
            not isinstance(receipt_hash, str)
            or not _SHA_RE.fullmatch(receipt_hash)
            or receipt_hash
            != _sha256_bytes(
                _canonical_json_bytes(receipt, trailing_lf=False)
            )
        ):
            _fail("committed receipt self-hash differs")

    expected_manifest = _final_manifest(
        expected_core,
        expected_consumption,
        receipts,
    )
    if dict(manifest) != expected_manifest:
        _fail("committed final manifest contract differs")
    return expected_core


def validate_committed_publication(
    root: Path = REPOSITORY_ROOT,
    *,
    expected_security_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the pushed Q -> P -> C -> D chain and committed artifacts."""

    root = root.resolve()
    head = _require_clean_pushed_branch(root, prereg.EXPECTED_BRANCH)
    preregistration_head = _decode_canonical_object(
        _git(root, "show", f"HEAD:{PREREGISTRATION_PATH.as_posix()}"),
        PREREGISTRATION_PATH.as_posix(),
        "manifest_hash",
    )
    claim_head = _decode_canonical_object(
        _git(root, "show", f"HEAD:{CLAIM_PATH.as_posix()}"),
        CLAIM_PATH.as_posix(),
        "claim_hash",
    )
    security_profile = (
        _validated_expected_security_profile(
            root, preregistration_head, expected_security_profile
        )
        if expected_security_profile is not None
        else _official_expected_security_profile(root, preregistration_head)
    )
    commits = _validate_committed_publication_topology(
        root,
        preregistration_head,
        head,
    )
    for profile_key, commit_key in (
        ("protocol_implementation_commit", "protocol_implementation_commit"),
        ("preregistration_seal_commit", "preregistration_seal_commit"),
        ("claim_commit", "claim_commit"),
        ("publication_commit", "publication_commit"),
    ):
        if profile_key in security_profile and (
            commits[commit_key] != security_profile[profile_key]
        ):
            _fail(f"committed topology differs from security profile: {profile_key}")
    security_profile["claim_commit"] = commits["claim_commit"]
    security_profile["publication_commit"] = commits[
        "publication_commit"
    ]
    snapshot, pairs = _preauthenticate_parent_snapshot(
        root,
        preregistration_head,
        content_mode=_PREREGISTRATION_PLUS_CLAIM,
        extra_tracked_paths=(
            SENTINEL_PATH,
            *WORKER_LEDGER_PATHS,
            CSV_PATH,
            MANIFEST_PATH,
        ),
        expected_security_profile=security_profile,
    )
    publication_context = _PublicationContext(
        root, expected_security_profile=security_profile
    )
    _validate_closed_entry_phase(
        publication_context, D12_COMMITTED_VERIFICATION
    )
    preregistration, prereg_binding = validate_preregistration(
        root,
        raw_cache=snapshot,
        expected_security_profile=security_profile,
    )
    synthetic = (
        preregistration.get("bindings", {}).get(
            "source_manifest_ordered_inventory"
        )
        == []
    )
    if preregistration != preregistration_head:
        _fail("committed preregistration differs from HEAD bootstrap")
    implementation = commits["protocol_implementation_commit"]
    preregistration_seal = commits["preregistration_seal_commit"]
    claim_commit = commits["claim_commit"]
    publication_commit = commits["publication_commit"]
    if preregistration["protocol_implementation_commit"] != implementation:
        _fail("committed preregistration implementation binding differs")

    active_raw = snapshot[PREREGISTRATION_PATH.as_posix()][0]
    if _sha256_bytes(active_raw) != prereg_binding["sha256"]:
        _fail("committed preregistration file hash differs")
    claim_tracked_raw = snapshot[CLAIM_PATH.as_posix()][0]
    claim, claim_raw, _claim_info = _read_canonical_object(
        _rooted(root, CLAIM_PATH),
        "claim_hash",
        path_text=CLAIM_PATH.as_posix(),
        raw_cache=snapshot,
    )
    if claim_raw != claim_tracked_raw or claim != claim_head:
        _fail("committed claim read differs")
    authority_amendments = _authority_amendment_bindings(
        preregistration,
        expected=security_profile["authority_amendments"],
    )
    protocol_files = claim.get("protocol_files")
    opaque_inputs = claim.get("opaque_inputs_authenticated")
    if not isinstance(protocol_files, list) or not isinstance(
        opaque_inputs, list
    ):
        _fail("committed claim bindings are incomplete")
    if [
        row.get("path") for row in protocol_files if isinstance(row, Mapping)
    ] != _planned_protocol_paths(preregistration):
        _fail("committed claim protocol path order differs")
    for row in protocol_files:
        if not isinstance(row, Mapping):
            _fail("committed claim protocol row is invalid")
        path = row.get("path")
        if not isinstance(path, str) or _head_blob_binding(
            root,
            path,
            raw_cache=snapshot,
            preclassified_pairs=pairs,
        ) != row:
            _fail("committed claim protocol binding differs")
    if claim != _claim_payload(
        preregistration_seal,
        prereg_binding,
        authority_amendments,
        protocol_files,
        opaque_inputs,
    ):
        _fail("committed claim schema differs")
    claim_binding = {
        "path": CLAIM_PATH.as_posix(),
        "sha256": _sha256_bytes(claim_raw),
        "claim_hash": claim["claim_hash"],
        "protocol_parent_commit": preregistration_seal,
        "claim_commit": claim_commit,
    }

    sentinel_tracked_raw = snapshot[SENTINEL_PATH.as_posix()][0]
    sentinel, sentinel_raw, _sentinel_info = _read_canonical_object(
        _rooted(root, SENTINEL_PATH),
        "manifest_hash",
        path_text=SENTINEL_PATH.as_posix(),
        raw_cache=snapshot,
    )
    if sentinel_raw != sentinel_tracked_raw:
        _fail("committed sentinel read differs")
    capabilities = _normalized_worker_capabilities(
        sentinel.get("worker_capabilities")
    )
    parent_authentication_sha256 = sentinel.get(
        "parent_authentication_sha256"
    )
    if not isinstance(
        parent_authentication_sha256, str
    ) or not _SHA_RE.fullmatch(parent_authentication_sha256):
        _fail("committed sentinel parent authentication hash differs")
    if sentinel != _sentinel_payload(
        claim_binding,
        prereg_binding,
        authority_amendments,
        parent_authentication_sha256,
        capabilities,
        expected_authority_amendments=security_profile[
            "authority_amendments"
        ],
    ):
        _fail("committed sentinel schema differs")
    sentinel_binding = {
        "path": SENTINEL_PATH.as_posix(),
        "sha256": _sha256_bytes(sentinel_raw),
        "manifest_hash": sentinel["manifest_hash"],
    }

    expected_consumption: list[dict[str, Any]] = []
    ledger_hashes: dict[int, str] = {}
    for capability, ledger_path in zip(
        capabilities,
        WORKER_LEDGER_PATHS,
        strict=True,
    ):
        ledger_tracked_raw = snapshot[ledger_path.as_posix()][0]
        ledger, ledger_raw, _ledger_info = _read_canonical_object(
            _rooted(root, ledger_path),
            None,
            path_text=ledger_path.as_posix(),
            raw_cache=snapshot,
        )
        if ledger_raw != ledger_tracked_raw:
            _fail("committed worker ledger read differs")
        expected_ledger = _worker_ledger_payload(
            binding=capability,
            claim=claim,
            preregistration=preregistration,
            sentinel=sentinel,
            authority_amendments=authority_amendments,
        )
        if ledger != expected_ledger:
            _fail("committed worker ledger schema differs")
        ledger_sha256 = _sha256_bytes(ledger_raw)
        ledger_hashes[int(capability["slot"])] = ledger_sha256
        expected_consumption.append(
            {
                "slot": capability["slot"],
                "parent_pid": capability["parent_pid"],
                "path": capability["consumed_ledger_path"],
                "sha256": ledger_sha256,
                "carrier_kind": capability["carrier_kind"],
                "carrier_device": capability["carrier_device"],
                "carrier_inode": capability["carrier_inode"],
                "token_sha256": capability["token_sha256"],
            }
        )

    csv_raw = snapshot[CSV_PATH.as_posix()][0]
    rows = validate_csv_gzip(csv_raw, require_all_sleeves=True)

    manifest_tracked_raw = snapshot[MANIFEST_PATH.as_posix()][0]
    manifest, manifest_raw, _manifest_info = _read_canonical_object(
        _rooted(root, MANIFEST_PATH),
        "manifest_hash",
        path_text=MANIFEST_PATH.as_posix(),
        raw_cache=snapshot,
    )
    if manifest_raw != manifest_tracked_raw:
        _fail("committed final manifest read differs")
    preregistration_bindings = preregistration.get("bindings")
    if not isinstance(preregistration_bindings, Mapping):
        _fail("committed preregistration bindings are absent")
    expected_parent_authentication = {
        "environment": preregistration_bindings.get("environment"),
        "hashed_inputs": opaque_inputs,
        "preregistration_authentication": {
            "manifest_hash": prereg_binding["manifest_hash"],
            "path": prereg_binding["path"],
            "protocol_implementation_commit": preregistration[
                "protocol_implementation_commit"
            ],
            "sha256": prereg_binding["sha256"],
        },
        "runtime_import_closure": preregistration_bindings.get(
            "runtime_import_closure"
        ),
    }
    _validate_final_manifest_contract(
        manifest,
        csv_raw=csv_raw,
        rows=rows,
        synthetic=synthetic,
        prereg_binding=prereg_binding,
        claim_binding=claim_binding,
        sentinel_binding=sentinel_binding,
        authority_amendments=authority_amendments,
        capabilities=capabilities,
        expected_consumption=expected_consumption,
        ledger_hashes=ledger_hashes,
        expected_parent_authentication=expected_parent_authentication,
        sentinel_parent_authentication_sha256=(
            parent_authentication_sha256
        ),
    )

    _final_parent_snapshot_recheck(
        root,
        snapshot,
        pairs,
        publication_context,
        D12_COMMITTED_VERIFICATION,
        prereg.EXPECTED_BRANCH,
    )
    result = {
        **commits,
        "identity": IDENTITY,
        "protocol_version": PROTOCOL_VERSION,
        "preregistration_manifest_hash": preregistration["manifest_hash"],
        "claim_hash": claim["claim_hash"],
        "sentinel_manifest_hash": sentinel["manifest_hash"],
        "final_manifest_hash": manifest["manifest_hash"],
        "csv_gzip_sha256": _sha256_bytes(csv_raw),
        "interval_count": len(rows),
        "head": publication_commit,
    }
    snapshot.close()
    publication_context.close()
    return result


def publish_v12_handoff(
    root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Supervise V12 once in memory, then publish the immutable H12 handoff."""

    root = root.resolve()
    _validate_bytecode_preflight(root)
    head = _require_clean_pushed_branch(root, prereg.EXPECTED_BRANCH)
    for relative in (H12_SUPERVISOR_SENTINEL_PATH, H12_HANDOFF_PATH):
        if _path_lexists(_rooted(root, relative)):
            _fail(f"H12 write-once path already exists: {relative}")
        if _git(root, "ls-files", "--", relative.as_posix()).strip():
            _fail(f"H12 fresh path is already tracked: {relative}")

    uv_executable = _authenticated_h12_uv_executable()
    bindings = _expected_h12_predecessor_bindings(root, head)
    stage_worktree = _h12_stage_worktree_snapshot(root, bindings)
    context = _PublicationContext(root)
    try:
        _validate_closed_entry_phase(
            context, D12_COMMITTED_VERIFICATION
        )
        baseline_entries = context.entries
    finally:
        context.close()
    if _h12_v12_artifact_names(baseline_entries):
        _fail("preexisting V12 file artifact exists")
    pre_sentinel_namespace = _repository_namespace_snapshot(root)

    def sentinel_prelink_recheck() -> None:
        _validate_bytecode_preflight(root)
        if _require_clean_pushed_branch(root, prereg.EXPECTED_BRANCH) != head:
            _fail("H12 repository HEAD changed before supervisor link")
        _require_h12_stage_worktree_unchanged(root, stage_worktree)
        if _repository_namespace_snapshot(root) != pre_sentinel_namespace:
            _fail("H12 repository namespace changed before supervisor link")

    parent = _single_parent_commit(root, head)
    capability = bytearray(os.urandom(32))
    capability_sha256 = _sha256_bytes(capability)
    read_fd, write_fd = os.pipe2(getattr(os, "O_CLOEXEC", 0))
    supervisor = _h12_supervisor_payload(
        head,
        parent,
        capability_sha256,
        os.getpid(),
    )
    supervisor_raw = _canonical_h12_json_bytes(supervisor)
    try:
        supervisor_publication = _publish_h12_leaf(
            root,
            H12_SUPERVISOR_SENTINEL_PATH,
            supervisor_raw,
            expected_entries=baseline_entries,
            prelink_recheck=sentinel_prelink_recheck,
        )
        supervisor_sha256 = supervisor_publication["sha256"]
        if supervisor_sha256 != _sha256_bytes(supervisor_raw):
            _fail("H12 supervisor publication hash differs")
        if os.write(write_fd, capability) != len(capability):
            _fail("H12 V12 capability pipe write was incomplete")
        os.close(write_fd)
        write_fd = -1

        environment = {
            key: os.environ[key]
            for key in (
                "HOME",
                "LANG",
                "LC_ALL",
                "SSL_CERT_DIR",
                "SSL_CERT_FILE",
            )
            if key in os.environ
        }
        environment.update(
            {
                "PATH": "/usr/bin:/bin",
                "PYTHONHASHSEED": "0",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": str(root),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUTF8": "1",
                "TZ": "UTC",
                "UV_NO_SYNC": "1",
                H12_CAPABILITY_FD_ENV: str(read_fd),
                H12_CAPABILITY_SHA256_ENV: capability_sha256,
                H12_SUPERVISOR_ENV: supervisor_sha256,
                H12_SUPERVISOR_PID_ENV: str(os.getpid()),
            }
        )
        before_v12_namespace = _repository_namespace_snapshot(root)
        completed = subprocess.run(
            [
                uv_executable,
                "run",
                "python",
                "-B",
                "-m",
                "training.build_gross9_structural_clock_bundle",
                "--verify-publication",
            ],
            cwd=root,
            env=environment,
            pass_fds=(read_fd,),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    finally:
        for descriptor in (write_fd, read_fd):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        capability[:] = b"\0" * len(capability)
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        _fail(f"nested V12 failed after H12 consumption: {detail}")
    if completed.stderr != b"":
        _fail("nested V12 emitted stderr")
    _validated_v12_stdout(completed.stdout, bindings)
    if _repository_namespace_snapshot(root) != before_v12_namespace:
        _fail("nested V12 changed the repository namespace")
    _require_h12_stage_worktree_unchanged(root, stage_worktree)

    if _require_clean_pushed_branch(root, prereg.EXPECTED_BRANCH) != head:
        _fail("nested V12 changed repository HEAD")
    _validate_bytecode_preflight(root)
    after_v12_entries = tuple(
        sorted(os.listdir(_rooted(root, Path("results"))))
    )
    expected_after_v12 = tuple(
        sorted((*baseline_entries, H12_SUPERVISOR_SENTINEL_PATH.name))
    )
    if (
        after_v12_entries != expected_after_v12
        or _h12_v12_artifact_names(after_v12_entries)
    ):
        _fail("nested V12 created a file or changed results inventory")

    handoff = _h12_handoff_payload(bindings, completed.stdout)
    validate_g9cb12_h12_handoff(
        handoff,
        v12_stdout=completed.stdout,
        predecessor_bindings=bindings,
    )
    handoff_raw = _canonical_h12_json_bytes(handoff)
    validate_g9cb12_h12_handoff(
        handoff_raw,
        v12_stdout=completed.stdout,
        predecessor_bindings=bindings,
    )

    def handoff_prelink_recheck() -> None:
        _validate_bytecode_preflight(root)
        if _require_clean_pushed_branch(root, prereg.EXPECTED_BRANCH) != head:
            _fail("H12 repository HEAD changed before handoff link")
        _require_h12_stage_worktree_unchanged(root, stage_worktree)
        if _repository_namespace_snapshot(root) != before_v12_namespace:
            _fail("H12 repository namespace changed before handoff link")

    handoff_publication = _publish_h12_leaf(
        root,
        H12_HANDOFF_PATH,
        handoff_raw,
        expected_entries=after_v12_entries,
        prelink_recheck=handoff_prelink_recheck,
    )
    _require_h12_stage_worktree_unchanged(root, stage_worktree)
    if _require_clean_pushed_branch(root, prereg.EXPECTED_BRANCH) != head:
        _fail("H12 publication changed repository HEAD")
    final_entries = tuple(
        sorted(os.listdir(_rooted(root, Path("results"))))
    )
    if _h12_v12_artifact_names(final_entries) != (
        H12_HANDOFF_PATH.name,
    ):
        _fail("H12 final V12 artifact inventory differs")
    for relative in (H12_SUPERVISOR_SENTINEL_PATH, H12_HANDOFF_PATH):
        if _git(root, "ls-files", "--", relative.as_posix()).strip():
            _fail(f"H12 command unexpectedly tracked a path: {relative}")
    return {
        "handoff": handoff,
        "handoff_publication": handoff_publication,
        "supervisor_publication": supervisor_publication,
    }


def produce_one_shot(
    root: Path = REPOSITORY_ROOT,
    *,
    synthetic_input: Path | None = None,
    expected_security_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticate, consume exactly two carriers, and publish manifest-last."""

    root = root.resolve()
    _validate_bytecode_preflight(root)
    raw_cache: dict[str, tuple[bytes, os.stat_result]] = {}
    preclassified_pairs: dict[str, tuple[str, str] | None] = {}
    snapshot_resources: dict[str, Any] = {}
    (
        claim,
        claim_binding,
        preregistration,
        prereg_binding,
    ) = _validate_claim_commit(
        root,
        raw_cache=raw_cache,
        preclassified_pairs=preclassified_pairs,
        _resource_out=snapshot_resources,
        expected_security_profile=expected_security_profile,
    )
    if set(snapshot_resources) != {"snapshot", "pairs", "security_profile"}:
        _fail("production parent snapshot resources are incomplete")
    snapshot = snapshot_resources["snapshot"]
    snapshot_pairs = snapshot_resources["pairs"]
    security_profile = snapshot_resources["security_profile"]
    authority_amendments = _authority_amendment_bindings(
        preregistration,
        expected=security_profile["authority_amendments"],
    )
    if claim.get("authority_amendments") != authority_amendments:
        _fail("claim authority amendments differ before production")
    environment = _validate_environment(preregistration, root)
    inputs = _validate_regular_hashed_inputs(
        root,
        preregistration,
        raw_cache=raw_cache,
        preclassified_pairs=preclassified_pairs,
    )
    closures = _validate_static_closures(
        root,
        preregistration,
        raw_cache=raw_cache,
    )
    if claim.get("opaque_inputs_authenticated") != inputs:
        _fail("current hashed inputs differ from the immutable claim")
    parent_authentication = {
        "environment": environment,
        "hashed_inputs": inputs,
        "preregistration_authentication": {
            "manifest_hash": prereg_binding["manifest_hash"],
            "path": prereg_binding["path"],
            "protocol_implementation_commit": preregistration[
                "protocol_implementation_commit"
            ],
            "sha256": prereg_binding["sha256"],
        },
        "runtime_import_closure": closures["runtime"],
    }
    parent_authentication_sha256 = _sha256_bytes(
        _canonical_json_bytes(
            parent_authentication, trailing_lf=False
        )
    )
    publication_context = _PublicationContext(
        root, expected_security_profile=security_profile
    )
    production_state = _ProductionStateMachine()
    _validate_closed_entry_phase(
        publication_context, C12_PRODUCTION_PREFLIGHT
    )
    production_state.advance(
        "C12_PRODUCTION_PREFLIGHT",
        lambda: _validate_production_namespace(
            publication_context,
            "C12_PRODUCTION_PREFLIGHT",
            Path("<slot-1-unreserved>"),
            Path("<slot-2-unreserved>"),
        ),
    )
    publication_context.probe()
    snapshot.rebaseline_directory_timestamps(
        matching_identity=publication_context.results_token
    )
    _validate_closed_entry_phase(
        publication_context, C12_PRODUCTION_PREFLIGHT
    )
    production_state.advance(
        "CAPABILITY_PROBE_COMPLETE",
        lambda: _validate_production_namespace(
            publication_context,
            "CAPABILITY_PROBE_COMPLETE",
            Path("<slot-1-unreserved>"),
            Path("<slot-2-unreserved>"),
        ),
    )
    results_directory = _rooted(root, SENTINEL_PATH).parent
    staging_patterns = [
        ".gross9-structural-clock-g9cb12-worker-*",
        f".{SENTINEL_PATH.name}.stage-*",
        f".{CSV_PATH.name}.stage-*",
        f".{MANIFEST_PATH.name}.stage-*",
        *[f".{path.name}.stage-*" for path in WORKER_LEDGER_PATHS],
    ]
    current_names = set(_directory_entries(publication_context.results_fd))
    leftovers = [
        name
        for name in current_names
        if any(fnmatch.fnmatch(name, pattern) for pattern in staging_patterns)
    ]
    if leftovers:
        _fail("leftover pre-access publication staging path exists")
    for path in WORKER_LEDGER_PATHS:
        if path.name in current_names:
            _fail(f"worker consumption ledger already exists: {path}")

    while True:
        suffixes = (os.urandom(12).hex(), os.urandom(12).hex())
        if suffixes[0] == suffixes[1]:
            continue
        stages = tuple(
            results_directory
            / f".gross9-structural-clock-g9cb12-worker-{suffix}"
            for suffix in suffixes
        )
        if not any(path.name in current_names for path in stages):
            break
    stage_one, stage_two = stages
    _create_stage_directory(
        stage_one,
        results_directory,
        publication_context=publication_context,
    )
    snapshot.rebaseline_directory_timestamps(
        matching_identity=publication_context.results_token
    )
    if stage_two.name in set(_directory_entries(publication_context.results_fd)):
        _fail("slot-2 reserved stage exists before slot 1")
    production_state.advance(
        "SLOT1_PREPARED",
        lambda: _validate_production_namespace(
            publication_context, "SLOT1_PREPARED", stage_one, stage_two
        ),
    )

    parent_pid = os.getpid()
    capabilities: list[dict[str, Any]] = []
    sentinel_published = False
    try:
        capabilities.append(
            _prepare_worker_capability(
                root=root,
                output_dir=stage_one,
                slot=1,
                parent_pid=parent_pid,
                results_fd=publication_context.results_fd,
            )
        )
        capabilities.append(
            _prepare_worker_capability(
                root=root,
                output_dir=stage_two,
                slot=2,
                parent_pid=parent_pid,
                results_fd=publication_context.results_fd,
            )
        )
        rows = [capability["row"] for capability in capabilities]
        _normalized_worker_capabilities(rows)
        invocation_one = _prepare_worker(
            root=root,
            capability=capabilities[0],
            other_stage_directory=rows[1]["stage_directory"],
            synthetic_input=synthetic_input,
            parent_authentication=parent_authentication,
            expected_security_profile=security_profile,
        )
        invocation_two = _prepare_worker(
            root=root,
            capability=capabilities[1],
            other_stage_directory=rows[0]["stage_directory"],
            synthetic_input=synthetic_input,
            parent_authentication=parent_authentication,
            expected_security_profile=security_profile,
        )
        sentinel = _sentinel_payload(
            claim_binding,
            prereg_binding,
            authority_amendments,
            parent_authentication_sha256,
            rows,
            expected_authority_amendments=security_profile[
                "authority_amendments"
            ],
        )
        sentinel_raw = _canonical_json_bytes(sentinel)
        _atomic_link_write_once(
            _rooted(root, SENTINEL_PATH),
            sentinel_raw,
            prelink_recheck=lambda: (
                _final_parent_snapshot_recheck(
                    root,
                    snapshot,
                    snapshot_pairs,
                    publication_context,
                    C12_PRODUCTION_PREFLIGHT,
                    prereg.EXPECTED_BRANCH,
                    path_state_check=lambda: (
                        _validate_production_checkpoint_two(
                            publication_context, stage_one, stage_two
                        )
                    ),
                )
            ),
            publication_context=publication_context,
        )
        sentinel_published = True
        snapshot.close()
        production_state.advance(
            "SENTINEL_LINKED",
            lambda: _validate_production_namespace(
                publication_context,
                "SENTINEL_LINKED",
                stage_one,
                stage_two,
            ),
        )

        invocation_one["ledger_observation"] = {
            "context": publication_context,
            "stage": stage_one,
            "ledger": WORKER_LEDGER_PATHS[0],
            "advance": lambda: production_state.advance(
                "PASS1_LEDGER_LINKED",
                lambda: (
                    _validate_production_ledger_checkpoint(
                        publication_context, WORKER_LEDGER_PATHS[0]
                    ),
                    _validate_production_namespace(
                        publication_context,
                        "PASS1_LEDGER_LINKED",
                        stage_one,
                        stage_two,
                    ),
                ),
            ),
        }
        worker_one_pid = _execute_prepared_worker(invocation_one)
        pass_one = _validate_worker_ledger_and_receipt(
            root=root,
            capability=capabilities[0],
            observed_worker_pid=worker_one_pid,
            claim=claim,
            preregistration=preregistration,
            sentinel=sentinel,
            authority_amendments=authority_amendments,
            publication_context=publication_context,
        )
        production_state.advance(
            "PASS1_OUTPUT_READY",
            lambda: _validate_production_namespace(
                publication_context,
                "PASS1_OUTPUT_READY",
                stage_one,
                stage_two,
            ),
        )
        _zero_token(capabilities[0]["token"])
        sentinel_binding = {
            "path": SENTINEL_PATH.as_posix(),
            "sha256": _sha256_bytes(sentinel_raw),
            "manifest_hash": sentinel["manifest_hash"],
        }
        rows_one, core_one = _validate_worker_product(
            pass_one["csv_raw"],
            pass_one["core_raw"],
            synthetic=synthetic_input is not None,
            prereg_binding=prereg_binding,
            parent_authentication=parent_authentication,
            claim_binding=claim_binding,
            sentinel_binding=sentinel_binding,
            authority_amendments=authority_amendments,
        )
        _cleanup_successful_stage(
            stage_one,
            results_directory,
            publication_context=publication_context,
        )
        if stage_two.name in set(
            _directory_entries(publication_context.results_fd)
        ):
            _fail("slot-2 reserved stage changed during slot 1")
        _create_stage_directory(
            stage_two,
            results_directory,
            publication_context=publication_context,
        )
        production_state.advance(
            "SLOT_TRANSITION",
            lambda: _validate_production_namespace(
                publication_context,
                "SLOT_TRANSITION",
                stage_one,
                stage_two,
            ),
        )

        invocation_two["ledger_observation"] = {
            "context": publication_context,
            "stage": stage_two,
            "ledger": WORKER_LEDGER_PATHS[1],
            "advance": lambda: production_state.advance(
                "PASS2_LEDGER_LINKED",
                lambda: (
                    _validate_production_ledger_checkpoint(
                        publication_context, WORKER_LEDGER_PATHS[1]
                    ),
                    _validate_production_namespace(
                        publication_context,
                        "PASS2_LEDGER_LINKED",
                        stage_one,
                        stage_two,
                    ),
                ),
            ),
        }
        worker_two_pid = _execute_prepared_worker(invocation_two)
        if worker_two_pid == worker_one_pid:
            _fail("the two fresh workers reported the same PID")
        pass_two = _validate_worker_ledger_and_receipt(
            root=root,
            capability=capabilities[1],
            observed_worker_pid=worker_two_pid,
            claim=claim,
            preregistration=preregistration,
            sentinel=sentinel,
            authority_amendments=authority_amendments,
            publication_context=publication_context,
        )
        production_state.advance(
            "PASS2_OUTPUT_READY",
            lambda: _validate_production_namespace(
                publication_context,
                "PASS2_OUTPUT_READY",
                stage_one,
                stage_two,
            ),
        )
        _zero_token(capabilities[1]["token"])
        rows_two, core_two = _validate_worker_product(
            pass_two["csv_raw"],
            pass_two["core_raw"],
            synthetic=synthetic_input is not None,
            prereg_binding=prereg_binding,
            parent_authentication=parent_authentication,
            claim_binding=claim_binding,
            sentinel_binding=sentinel_binding,
            authority_amendments=authority_amendments,
        )
        if (
            pass_one["csv_raw"] != pass_two["csv_raw"]
            or pass_one["core_raw"] != pass_two["core_raw"]
            or rows_one != rows_two
            or core_one != core_two
        ):
            _fail("independent rebuild bytes or reparses differ")

        _atomic_link_write_once(
            _rooted(root, CSV_PATH),
            pass_one["csv_raw"],
            publication_context=publication_context,
        )
        production_state.advance(
            "CANONICAL_CSV_LINKED",
            lambda: _validate_production_namespace(
                publication_context,
                "CANONICAL_CSV_LINKED",
                stage_one,
                stage_two,
            ),
        )
        manifest = _final_manifest(
            core_one,
            [pass_one["consumption"], pass_two["consumption"]],
            [
                pass_one["rebuild_receipt"],
                pass_two["rebuild_receipt"],
            ],
        )
        manifest_raw = _canonical_json_bytes(manifest)
        _atomic_link_write_once(
            _rooted(root, MANIFEST_PATH),
            manifest_raw,
            publication_context=publication_context,
        )
        production_state.advance(
            "MANIFEST_LINKED_LAST",
            lambda: _validate_production_namespace(
                publication_context,
                "MANIFEST_LINKED_LAST",
                stage_one,
                stage_two,
            ),
        )
        _cleanup_successful_stage(
            stage_two,
            results_directory,
            publication_context=publication_context,
        )
        production_state.advance(
            "FINAL_CLEANUP",
            lambda: _validate_production_namespace(
                publication_context,
                "FINAL_CLEANUP",
                stage_one,
                stage_two,
            ),
        )
        production_state.require_complete()
        publication_context.close()
        return {
            "identity": IDENTITY,
            "rows": len(rows_one),
            "csv_gzip_sha256": _sha256_bytes(pass_one["csv_raw"]),
            "manifest_hash": manifest["manifest_hash"],
            "terminal_on_any_later_failure": TERMINAL_ACTION,
        }
    except BaseException as exc:
        for capability in capabilities:
            descriptor = int(capability["read_fd"])
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                capability["read_fd"] = -1
            ledger_descriptor = int(capability["ledger_fd"])
            if ledger_descriptor >= 0:
                try:
                    os.close(ledger_descriptor)
                except OSError:
                    pass
                capability["ledger_fd"] = -1
            token = capability.get("token")
            if isinstance(token, bytearray):
                _zero_token(token)
        if (
            not sentinel_published
            and stage_one.is_dir()
            and not stage_one.is_symlink()
            and not any(stage_one.iterdir())
        ):
            stage_one.rmdir()
        snapshot.close()
        publication_context.close()
        if isinstance(exc, TerminalG9CB12Failure):
            raise
        raise TerminalG9CB12Failure(f"{TERMINAL_ACTION}: {exc}") from exc

def _raw_worker_option(arguments: Sequence[str], name: str) -> str:
    positions = [
        index for index, value in enumerate(arguments) if value == name
    ]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        _fail(f"internal worker raw option is absent or repeated: {name}")
    return arguments[positions[0] + 1]


def _raw_worker_descriptor(arguments: Sequence[str], name: str) -> int:
    text = _raw_worker_option(arguments, name)
    if not re.fullmatch(r"(?:0|[1-9][0-9]*)", text):
        _fail(f"internal worker raw descriptor is invalid: {name}")
    return int(text)


def _validate_inherited_worker_descriptors(
    capability_fd: int, ledger_fd: int
) -> None:
    if capability_fd == ledger_fd or capability_fd < 0 or ledger_fd < 0:
        _fail("worker inherited descriptors are not distinct nonnegative values")
    capability_info = os.fstat(capability_fd)
    ledger_info = os.fstat(ledger_fd)
    if not stat.S_ISFIFO(capability_info.st_mode):
        _fail("worker inherited capability descriptor is not a FIFO")
    if (
        not stat.S_ISREG(ledger_info.st_mode)
        or stat.S_IMODE(ledger_info.st_mode) != 0o600
        or ledger_info.st_size != 0
    ):
        _fail("worker inherited ledger descriptor differs")
    expected = {0, 1, 2, capability_fd, ledger_fd}
    observed: set[int] = set()
    upper = int(os.sysconf("SC_OPEN_MAX"))
    for descriptor in range(upper):
        try:
            os.fstat(descriptor)
        except OSError as exc:
            if exc.errno != errno.EBADF:
                raise
        else:
            observed.add(descriptor)
    if observed != expected:
        _fail("worker inherited descriptor table differs")


def _early_worker_bootstrap(
    arguments: Sequence[str],
) -> _WorkerIsolationGuard | None:
    internal_count = sum(value == "--internal-worker" for value in arguments)
    if internal_count == 0:
        return None
    if internal_count != 1:
        _fail("internal worker marker is repeated")
    expected_parent_text = _raw_worker_option(
        arguments, "--expected-parent-pid"
    )
    if not re.fullmatch(r"[1-9][0-9]*", expected_parent_text):
        _fail("expected parent PID raw argument is invalid")
    expected_parent_pid = int(expected_parent_text)
    _establish_parent_death_contract(expected_parent_pid)
    capability_fd = _raw_worker_descriptor(
        arguments, "--worker-capability-fd"
    )
    ledger_fd = _raw_worker_descriptor(arguments, "--worker-ledger-fd")
    _validate_inherited_worker_descriptors(capability_fd, ledger_fd)
    root_text = _raw_worker_option(arguments, "--repository-root")
    own_stage_text = _raw_worker_option(arguments, "--output-dir")
    other_stage_text = _raw_worker_option(
        arguments, "--other-stage-directory"
    )
    root = Path(root_text)
    if Path.cwd() != root:
        _fail("worker current directory is not the exact repository root")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    repository_fd = results_fd = filesystem_root_fd = -1
    try:
        repository_fd = os.open(".", flags)
        results_fd = os.open("results", flags, dir_fd=repository_fd)
        filesystem_root_fd = os.open("/", flags)
        guard = _WorkerIsolationGuard(
            root=root,
            own_stage=own_stage_text,
            other_stage=other_stage_text,
            ledger_paths=WORKER_LEDGER_PATHS,
            repository_fd=repository_fd,
            results_fd=results_fd,
            filesystem_root_fd=filesystem_root_fd,
            ledger_fd=ledger_fd,
        )
    except BaseException:
        for descriptor in (
            filesystem_root_fd,
            results_fd,
            repository_fd,
        ):
            if descriptor >= 0:
                os.close(descriptor)
        raise
    guard.install()
    return guard


G9CB13_IDENTITY = prereg.G9CB13_IDENTITY
G9CB13_PROTOCOL_VERSION = prereg.G9CB13_PROTOCOL_VERSION
G9CB13_PREREGISTRATION_PATH = prereg.G9CB13_PREREGISTRATION_PATH
G9CB13_CLAIM_PATH = prereg.G9CB13_ACCESS_CLAIM_PATH
G9CB13_SENTINEL_PATH = prereg.G9CB13_ATTEMPT_SENTINEL_PATH
G9CB13_WORKER_LEDGER_PATHS = (
    prereg.G9CB13_WORKER_CAPABILITY_CONSUMPTION_LEDGER_PATHS
)
G9CB13_CSV_PATH = prereg.G9CB13_BUNDLE_PATH
G9CB13_MANIFEST_PATH = prereg.G9CB13_FINAL_MANIFEST_PATH
H13_HANDOFF_PATH = Path(
    "results/gross9_structural_clock_bundle_g9cb13_v13_handoff_"
    "2026-08-01.json"
)
H13_SUPERVISOR_SENTINEL_PATH = Path(
    "results/gross9_structural_clock_bundle_g9cb13_h13_supervisor_"
    "attempt_consumed_2026-08-01.json"
)
V13_COMMAND = (
    "PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m "
    "training.build_gross9_structural_clock_bundle --verify-publication"
)
H13_COMMAND = (
    "PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m "
    "training.build_gross9_structural_clock_bundle --publish-v13-handoff"
)
H13_SUPERVISOR_ENV = "G9CB13_H13_SUPERVISOR_SENTINEL_SHA256"
H13_CAPABILITY_FD_ENV = "G9CB13_H13_V13_CAPABILITY_FD"
H13_CAPABILITY_SHA256_ENV = "G9CB13_H13_V13_CAPABILITY_SHA256"
H13_SUPERVISOR_PID_ENV = "G9CB13_H13_SUPERVISOR_PID"
H13_STATE_TRACE = (
    "PRE_SUPERVISOR",
    "SUPERVISOR_LINKED",
    "V13_VERIFIED",
    "HANDOFF_LINKED",
)
H13_TOP_LEVEL_KEYS = (
    "active_alpha_goal",
    "adopted_source_bindings",
    "adopted_source_generation",
    "identity",
    "ledger_kind",
    "next_workflow",
    "no_economics",
    "no_future_commit_prediction",
    "schema_version",
    "source_adoption_mode",
    "successor_bindings",
    "successor_generation",
    "t12_persisted_sha256",
    "t12_terminal_failure_hash",
    "v13_stdout_hash",
)
V13_STDOUT_KEYS = V12_STDOUT_KEYS
H13_SUPERVISOR_KEYS = (
    "attempt_hash",
    "capability_sha256",
    "expected_handoff_path",
    "h13_command",
    "identity",
    "one_shot",
    "repository_head",
    "repository_parent",
    "resume_allowed",
    "retry_allowed",
    "supervisor_pid",
    "uv_executable",
    "uv_executable_sha256",
    "v13_command",
    "zero_economics",
)
H13_SUCCESSOR_STAGE_PATHS = {
    "A13": (
        "docs/gross9-structural-clock-bundle-g9cb13-successor-authority-"
        "decision-2026-08-01.md",
    ),
    "T12": (prereg.G9CB12_T12_TERMINAL_FAILURE_PATH.as_posix(),),
    "Q13": prereg.G9CB13_Q13_PATHS,
    "P13": (G9CB13_PREREGISTRATION_PATH.as_posix(),),
    "C13": (G9CB13_CLAIM_PATH.as_posix(),),
    "D13": (
        G9CB13_SENTINEL_PATH.as_posix(),
        G9CB13_WORKER_LEDGER_PATHS[0].as_posix(),
        G9CB13_WORKER_LEDGER_PATHS[1].as_posix(),
        G9CB13_CSV_PATH.as_posix(),
        G9CB13_MANIFEST_PATH.as_posix(),
    ),
}


class _H13StateMachine:
    def __init__(self) -> None:
        self.state = "PRE_SUPERVISOR"
        self.events = [self.state]

    def advance(self, expected: str, target: str) -> None:
        if self.state != expected:
            _fail(f"H13 illegal state transition: {self.state} -> {target}")
        expected_target = H13_STATE_TRACE[H13_STATE_TRACE.index(expected) + 1]
        if target != expected_target:
            _fail(f"H13 illegal state transition: {expected} -> {target}")
        self.state = target
        self.events.append(target)


def _contains_forbidden_economics(value: Any) -> bool:
    forbidden = {"candidate", "comparator", "economic_result", "economic-result"}
    if isinstance(value, Mapping):
        return any(
            key in forbidden
            or _contains_forbidden_economics(member)
            for key, member in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_economics(member) for member in value)
    return False


def _g9cb13_topology_active(root: Path) -> bool:
    return bool(
        _git(
            root,
            "ls-files",
            "--",
            prereg.G9CB12_T12_TERMINAL_FAILURE_PATH.as_posix(),
        ).strip()
    )


def _activate_g9cb13_topology() -> tuple[dict[str, Any], dict[str, Any]]:
    builder_updates = {
        "IDENTITY": G9CB13_IDENTITY,
        "PROTOCOL_VERSION": G9CB13_PROTOCOL_VERSION,
        "PREREGISTRATION_PATH": G9CB13_PREREGISTRATION_PATH,
        "CLAIM_PATH": G9CB13_CLAIM_PATH,
        "SENTINEL_PATH": G9CB13_SENTINEL_PATH,
        "WORKER_LEDGER_PATHS": G9CB13_WORKER_LEDGER_PATHS,
        "CSV_PATH": G9CB13_CSV_PATH,
        "MANIFEST_PATH": G9CB13_MANIFEST_PATH,
        "TERMINAL_ACTION": "TERMINAL_G9CB13_ATTEMPT_CONSUMED_NO_RETRY",
        "_PYCACHE_PREFIX_RELATIVE": Path("results/.g9cb13-bytecode-cache-disabled"),
    }
    prereg_updates = {
        "IDENTITY": prereg.G9CB13_IDENTITY,
        "PROTOCOL_VERSION": prereg.G9CB13_PREREGISTRATION_PROTOCOL_VERSION,
        "PREREGISTRATION_PATH": prereg.G9CB13_PREREGISTRATION_PATH,
        "ACCESS_CLAIM_PATH": prereg.G9CB13_ACCESS_CLAIM_PATH,
        "ATTEMPT_SENTINEL_PATH": prereg.G9CB13_ATTEMPT_SENTINEL_PATH,
        "WORKER_CAPABILITY_CONSUMPTION_LEDGER_PATHS": (
            prereg.G9CB13_WORKER_CAPABILITY_CONSUMPTION_LEDGER_PATHS
        ),
        "BUNDLE_PATH": prereg.G9CB13_BUNDLE_PATH,
        "FINAL_MANIFEST_PATH": prereg.G9CB13_FINAL_MANIFEST_PATH,
    }
    saved_builder = {key: globals()[key] for key in builder_updates}
    saved_prereg = {key: getattr(prereg, key) for key in prereg_updates}
    globals().update(builder_updates)
    for key, value in prereg_updates.items():
        setattr(prereg, key, value)
    return saved_builder, saved_prereg


def _restore_g9cb12_topology(
    saved: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    saved_builder, saved_prereg = saved
    globals().update(saved_builder)
    for key, value in saved_prereg.items():
        setattr(prereg, key, value)


def _validated_v13_stdout(
    raw: bytes, successor_bindings: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    payload = _decode_h12_canonical_object(raw, "V13 stdout")
    if tuple(payload) != V13_STDOUT_KEYS:
        _fail("V13 stdout keys/order differ")
    for key in (
        "claim_commit",
        "head",
        "preregistration_seal_commit",
        "protocol_implementation_commit",
        "publication_commit",
    ):
        if not isinstance(payload.get(key), str) or not _COMMIT_RE.fullmatch(
            payload[key]
        ):
            _fail(f"V13 stdout commit differs: {key}")
    for key in (
        "claim_hash",
        "csv_gzip_sha256",
        "final_manifest_hash",
        "preregistration_manifest_hash",
        "sentinel_manifest_hash",
    ):
        if not isinstance(payload.get(key), str) or not _SHA_RE.fullmatch(
            payload[key]
        ):
            _fail(f"V13 stdout hash differs: {key}")
    by_stage = {row["stage"]: row for row in successor_bindings}
    if (
        payload.get("identity") != G9CB13_IDENTITY
        or payload.get("protocol_version") != G9CB13_PROTOCOL_VERSION
        or type(payload.get("interval_count")) is not int
        or payload["interval_count"] < 0
        or payload["head"] != payload["publication_commit"]
        or payload["protocol_implementation_commit"] != by_stage["Q13"]["commit"]
        or payload["preregistration_seal_commit"] != by_stage["P13"]["commit"]
        or payload["claim_commit"] != by_stage["C13"]["commit"]
        or payload["publication_commit"] != by_stage["D13"]["commit"]
    ):
        _fail("V13 stdout topology differs")
    return payload


def _h13_supervisor_payload(
    head: str, parent: str, capability_sha256: str, supervisor_pid: int
) -> dict[str, Any]:
    core = {
        "capability_sha256": capability_sha256,
        "expected_handoff_path": H13_HANDOFF_PATH.as_posix(),
        "h13_command": H13_COMMAND,
        "identity": "G9CB-13-H13-SUPERVISOR",
        "one_shot": True,
        "repository_head": head,
        "repository_parent": parent,
        "resume_allowed": False,
        "retry_allowed": False,
        "supervisor_pid": supervisor_pid,
        "uv_executable": H12_UV_EXECUTABLE.as_posix(),
        "uv_executable_sha256": H12_UV_EXECUTABLE_SHA256,
        "v13_command": V13_COMMAND,
        "zero_economics": True,
    }
    return {"attempt_hash": _sha256_bytes(_canonical_h12_json_bytes(core, trailing_lf=False)), **core}


def validate_g9cb13_h13_supervisor(payload: Mapping[str, Any] | bytes) -> dict[str, Any]:
    observed = (
        _decode_h12_canonical_object(payload, "H13 supervisor")
        if isinstance(payload, bytes)
        else dict(payload)
    )
    if tuple(observed) != H13_SUPERVISOR_KEYS:
        _fail("H13 supervisor keys/order differ")
    core = dict(observed)
    attempt_hash = core.pop("attempt_hash", None)
    if (
        attempt_hash != _sha256_bytes(_canonical_h12_json_bytes(core, trailing_lf=False))
        or observed.get("identity") != "G9CB-13-H13-SUPERVISOR"
        or observed.get("one_shot") is not True
        or observed.get("retry_allowed") is not False
        or observed.get("resume_allowed") is not False
        or observed.get("zero_economics") is not True
        or _contains_forbidden_economics(observed)
    ):
        _fail("H13 supervisor contract differs")
    return observed


def _h13_handoff_payload(
    successor_bindings: Sequence[Mapping[str, Any]], v13_stdout: bytes
) -> dict[str, Any]:
    _validated_v13_stdout(v13_stdout, successor_bindings)
    return {
        "active_alpha_goal": "incomplete",
        "adopted_source_bindings": prereg.g9cb13_adopted_source_bindings(),
        "adopted_source_generation": "G9CB12",
        "identity": G9CB13_IDENTITY,
        "ledger_kind": "gross9_structural_clock_bundle_g9cb13_v13_handoff_v1",
        "next_workflow": "ralplan",
        "no_economics": True,
        "no_future_commit_prediction": True,
        "schema_version": 1,
        "source_adoption_mode": (
            "authenticated_g9cb12_source_bytes_consumable_by_g9cb13_"
            "no_republication_v1"
        ),
        "successor_bindings": [dict(row) for row in successor_bindings],
        "successor_generation": "G9CB13",
        "t12_persisted_sha256": prereg.G9CB13_T12_PERSISTED_SHA256,
        "t12_terminal_failure_hash": prereg.G9CB13_T12_TERMINAL_FAILURE_HASH,
        "v13_stdout_hash": _sha256_bytes(v13_stdout),
    }


def validate_g9cb13_h13_handoff(
    payload: Mapping[str, Any] | bytes,
    *,
    v13_stdout: bytes,
    successor_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    observed = (
        _decode_h12_canonical_object(payload, "H13 handoff")
        if isinstance(payload, bytes)
        else dict(payload)
    )
    if (
        tuple(observed) != H13_TOP_LEVEL_KEYS
        or observed != _h13_handoff_payload(successor_bindings, v13_stdout)
        or observed.get("no_economics") is not True
        or observed.get("no_future_commit_prediction") is not True
        or _contains_forbidden_economics(observed)
    ):
        _fail("H13 verified handoff contract differs")
    for row in observed["successor_bindings"]:
        if row.get("stage") not in H13_SUCCESSOR_STAGE_PATHS:
            _fail("H13 successor stage differs")
        if [entry["path"] for entry in row.get("tracked_files", [])] != list(
            H13_SUCCESSOR_STAGE_PATHS[row["stage"]]
        ):
            _fail("H13 successor tracked-file inventory differs")
        if any(
            entry["path"] == H13_SUPERVISOR_SENTINEL_PATH.as_posix()
            for entry in row["tracked_files"]
        ):
            _fail("H13 supervisor leaked into successor bindings")
    return observed


def _g9cb13_tracked_file_row(
    root: Path, commit: str, path_text: str, worktree_mode: str
) -> dict[str, Any]:
    raw_entry = _git(root, "ls-tree", commit, "--", path_text).decode().strip()
    metadata, observed_path = raw_entry.split("\t", 1)
    git_mode, kind, blob = metadata.split()
    raw = _git(root, "show", f"{commit}:{path_text}")
    if observed_path != path_text or git_mode != "100644" or kind != "blob":
        _fail(f"G13 stage tree entry differs: {path_text}")
    worktree = _rooted(root, Path(path_text))
    if (
        not worktree.is_file()
        or worktree.is_symlink()
        or _sha256_file(worktree) != _sha256_bytes(raw)
        or f"{stat.S_IMODE(worktree.stat().st_mode):04o}" != worktree_mode
    ):
        _fail(f"G13 stage worktree binding differs: {path_text}")
    return {
        "git_blob": blob,
        "git_mode": git_mode,
        "path": path_text,
        "sha256": _sha256_bytes(raw),
        "size_bytes": len(raw),
        "worktree_mode": worktree_mode,
    }


def _expected_h13_successor_bindings(
    root: Path, d13_head: str
) -> list[dict[str, Any]]:
    stages = tuple(H13_SUCCESSOR_STAGE_PATHS)
    commits: dict[str, str] = {"D13": d13_head}
    for current, prior in zip(reversed(stages[1:]), reversed(stages[:-1]), strict=True):
        commits[prior] = _single_parent_commit(root, commits[current])
    q12 = _single_parent_commit(root, commits["A13"])
    if q12 != "e64f8de05e18b1d0fdfc9f3582d5f32041d0fa54":
        _fail("A13 direct parent differs from Q12 authority")
    parents = {"A13": q12}
    for prior, current in zip(stages, stages[1:], strict=False):
        parents[current] = commits[prior]
    rows: list[dict[str, Any]] = []
    for stage in stages:
        paths = H13_SUCCESSOR_STAGE_PATHS[stage]
        status = "M" if stage == "Q13" else "A"
        expected_diff = tuple(
            f"{status}\t{path_text}" for path_text in sorted(paths)
        )
        if _commit_name_status(root, parents[stage], commits[stage]) != expected_diff:
            _fail(f"H13 {stage} parent diff differs")
        worktree_mode = "0644" if stage in {"A13", "Q13"} else "0444"
        rows.append(
            {
                "commit": commits[stage],
                "parent_commit": parents[stage],
                "stage": stage,
                "tracked_files": [
                    _g9cb13_tracked_file_row(
                        root, commits[stage], path_text, worktree_mode
                    )
                    for path_text in paths
                ],
            }
        )
    return rows


def _require_h13_v13_supervision(root: Path) -> dict[str, Any]:
    sentinel_sha = os.environ.get(H13_SUPERVISOR_ENV, "")
    capability_sha = os.environ.get(H13_CAPABILITY_SHA256_ENV, "")
    descriptor_text = os.environ.get(H13_CAPABILITY_FD_ENV, "")
    pid_text = os.environ.get(H13_SUPERVISOR_PID_ENV, "")
    if (
        not _SHA_RE.fullmatch(sentinel_sha)
        or not _SHA_RE.fullmatch(capability_sha)
        or not re.fullmatch(r"[0-9]+", descriptor_text)
        or not re.fullmatch(r"[1-9][0-9]*", pid_text)
    ):
        _fail("V13 direct invocation is forbidden")
    descriptor = int(descriptor_text)
    try:
        capability = os.read(descriptor, 33)
    finally:
        os.close(descriptor)
    if len(capability) != 32 or _sha256_bytes(capability) != capability_sha:
        _fail("V13 capability differs")
    raw = _rooted(root, H13_SUPERVISOR_SENTINEL_PATH).read_bytes()
    if _sha256_bytes(raw) != sentinel_sha:
        _fail("V13 supervisor persisted hash differs")
    sentinel = validate_g9cb13_h13_supervisor(raw)
    if sentinel["capability_sha256"] != capability_sha:
        _fail("V13 supervisor capability binding differs")
    return sentinel


def publish_v13_handoff(root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    """Publish supervisor, run V13 once, then publish the verified H13 handoff."""

    root = root.resolve()
    _validate_bytecode_preflight(root)
    head = _require_clean_pushed_branch(root, prereg.EXPECTED_BRANCH)
    bindings = _expected_h13_successor_bindings(root, head)
    for relative in (H13_SUPERVISOR_SENTINEL_PATH, H13_HANDOFF_PATH):
        if _path_lexists(_rooted(root, relative)):
            _fail(f"H13 write-once path already exists: {relative}")
    state = _H13StateMachine()
    baseline = prereg.g9cb13_results_inventory_preflight(root)["names"]
    capability = bytearray(os.urandom(32))
    capability_sha = _sha256_bytes(capability)
    read_fd, write_fd = os.pipe2(getattr(os, "O_CLOEXEC", 0))
    supervisor = _h13_supervisor_payload(
        head, _single_parent_commit(root, head), capability_sha, os.getpid()
    )
    validate_g9cb13_h13_supervisor(supervisor)
    supervisor_raw = _canonical_h12_json_bytes(supervisor)
    try:
        supervisor_publication = _publish_h12_leaf(
            root,
            H13_SUPERVISOR_SENTINEL_PATH,
            supervisor_raw,
            expected_entries=tuple(baseline),
            prelink_recheck=lambda: prereg.g9cb13_results_inventory_preflight(root),
        )
        state.advance("PRE_SUPERVISOR", "SUPERVISOR_LINKED")
        if os.write(write_fd, capability) != 32:
            _fail("H13 capability write was incomplete")
        os.close(write_fd)
        write_fd = -1
        environment = dict(os.environ)
        environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(root),
                H13_CAPABILITY_FD_ENV: str(read_fd),
                H13_CAPABILITY_SHA256_ENV: capability_sha,
                H13_SUPERVISOR_ENV: _sha256_bytes(supervisor_raw),
                H13_SUPERVISOR_PID_ENV: str(os.getpid()),
            }
        )
        before_v13 = _repository_namespace_snapshot(root)
        completed = subprocess.run(
            [
                H12_UV_EXECUTABLE,
                "run",
                "python",
                "-B",
                "-m",
                "training.build_gross9_structural_clock_bundle",
                "--verify-publication",
            ],
            cwd=root,
            env=environment,
            pass_fds=(read_fd,),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    finally:
        for descriptor in (write_fd, read_fd):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        capability[:] = b"\0" * len(capability)
    if completed.returncode != 0 or completed.stderr:
        _fail("nested V13 failed or emitted stderr")
    _validated_v13_stdout(completed.stdout, bindings)
    if _repository_namespace_snapshot(root) != before_v13:
        _fail("nested V13 changed repository namespace")
    state.advance("SUPERVISOR_LINKED", "V13_VERIFIED")
    handoff = _h13_handoff_payload(bindings, completed.stdout)
    validate_g9cb13_h13_handoff(
        handoff, v13_stdout=completed.stdout, successor_bindings=bindings
    )
    after_supervisor = prereg.g9cb13_results_inventory_preflight(
        root, active_untracked_prefix=(H13_SUPERVISOR_SENTINEL_PATH.name,)
    )["names"]
    handoff_raw = _canonical_h12_json_bytes(handoff)
    handoff_publication = _publish_h12_leaf(
        root,
        H13_HANDOFF_PATH,
        handoff_raw,
        expected_entries=tuple(after_supervisor),
        prelink_recheck=lambda: prereg.g9cb13_results_inventory_preflight(
            root, active_untracked_prefix=(H13_SUPERVISOR_SENTINEL_PATH.name,)
        ),
    )
    state.advance("V13_VERIFIED", "HANDOFF_LINKED")
    return {
        "events": tuple(state.events),
        "handoff": handoff,
        "handoff_publication": handoff_publication,
        "supervisor_publication": supervisor_publication,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--create-claim", action="store_true")
    actions.add_argument("--produce", action="store_true")
    actions.add_argument("--verify-publication", action="store_true")
    actions.add_argument("--publish-v12-handoff", action="store_true")
    actions.add_argument("--publish-v13-handoff", action="store_true")
    actions.add_argument("--internal-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repository-root", default=str(REPOSITORY_ROOT), help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", help=argparse.SUPPRESS)
    parser.add_argument("--other-stage-directory", help=argparse.SUPPRESS)
    parser.add_argument("--worker-capability-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-ledger-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--expected-parent-pid", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--parent-auth-json", default="{}", help=argparse.SUPPRESS)
    parser.add_argument(
        "--expected-security-profile-json",
        default="{}",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--synthetic-input", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    root_text = str(REPOSITORY_ROOT)
    if "--repository-root" in raw_arguments:
        index = raw_arguments.index("--repository-root")
        if index + 1 < len(raw_arguments):
            root_text = raw_arguments[index + 1]
    root = Path(root_text)
    active_g13 = root.exists() and _g9cb13_topology_active(root)
    saved = _activate_g9cb13_topology() if active_g13 else None
    try:
        guard = _early_worker_bootstrap(raw_arguments)
        arguments = _parser().parse_args(raw_arguments)
        if arguments.internal_worker:
            if (
                guard is None
                or not isinstance(arguments.output_dir, str)
                or not arguments.output_dir
                or not isinstance(arguments.other_stage_directory, str)
                or not arguments.other_stage_directory
                or type(arguments.worker_capability_fd) is not int
                or arguments.worker_capability_fd < 0
                or type(arguments.worker_ledger_fd) is not int
                or arguments.worker_ledger_fd < 0
                or arguments.worker_capability_fd == arguments.worker_ledger_fd
                or type(arguments.expected_parent_pid) is not int
                or arguments.expected_parent_pid <= 0
                or not isinstance(arguments.expected_security_profile_json, str)
                or arguments.expected_security_profile_json == "{}"
            ):
                _fail("internal worker arguments are incomplete")
            return _worker_main(arguments, guard)
        if guard is not None:
            _fail("worker bootstrap was installed for a non-worker action")
        if arguments.synthetic_input:
            _fail("synthetic input is accepted only by an authenticated internal worker")
        if active_g13 and getattr(arguments, "publish_v12_handoff", False):
            _fail("legacy G12 execution is prohibited under active G13 topology")
        if arguments.create_claim:
            create_claim_only(Path(arguments.repository_root))
        elif arguments.verify_publication:
            if active_g13:
                _require_h13_v13_supervision(Path(arguments.repository_root))
            else:
                _require_h12_v12_supervision(Path(arguments.repository_root))
            result = validate_committed_publication(
                Path(arguments.repository_root)
            )
            print(
                _canonical_json_bytes(
                    result,
                    trailing_lf=False,
                ).decode("ascii")
            )
        elif getattr(arguments, "publish_v13_handoff", False):
            if not active_g13:
                _fail("H13 requires active G13 topology")
            publish_v13_handoff(Path(arguments.repository_root))
        elif getattr(arguments, "publish_v12_handoff", False):
            publish_v12_handoff(Path(arguments.repository_root))
        else:
            produce_one_shot(Path(arguments.repository_root))
        return 0
    finally:
        if saved is not None:
            _restore_g9cb12_topology(saved)


if __name__ == "__main__":
    raise SystemExit(main())
