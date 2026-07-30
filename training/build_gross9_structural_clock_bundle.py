"""Claim and one-shot builder for the G9CB-1 structural clock bundle.

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
import csv
import ctypes
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import errno
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
from typing import Any, Iterable, Mapping, Sequence
import zlib

from training import preregister_gross9_structural_clock_bundle as prereg


IDENTITY = "G9CB-1"
PROTOCOL_VERSION = "gross9_structural_clock_bundle_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = Path("training/build_gross9_structural_clock_bundle.py")
BUILDER_TEST_PATH = Path("tests/test_build_gross9_structural_clock_bundle.py")
PREREGISTER_PATH = Path("training/preregister_gross9_structural_clock_bundle.py")
PREREGISTRATION_PATH = Path(
    "results/gross9_structural_clock_bundle_preregistration_2026-07-31.json"
)
CLAIM_PATH = Path(
    "results/gross9_structural_clock_bundle_access_claim_2026-07-31.json"
)
SENTINEL_PATH = Path(
    "results/gross9_structural_clock_bundle_attempt_consumed_2026-07-31.json"
)
CSV_PATH = Path("results/gross9_structural_clock_bundle_2026-07-31.csv.gz")
MANIFEST_PATH = Path(
    "results/gross9_structural_clock_bundle_manifest_2026-07-31.json"
)
WORKER_LEDGER_PATHS = (
    Path(
        "results/"
        "gross9_structural_clock_bundle_worker_capability_consumed_pass1_"
        "2026-07-31.json"
    ),
    Path(
        "results/"
        "gross9_structural_clock_bundle_worker_capability_consumed_pass2_"
        "2026-07-31.json"
    ),
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
TERMINAL_ACTION = "TERMINAL_G9CB1_ATTEMPT_CONSUMED_NO_RETRY"
_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
_DIST_NORMALIZE_RE = re.compile(r"[-_.]+")
_STAGED_CSV_NAME = "gross9_structural_clock_bundle.csv.gz"
_STAGED_CORE_NAME = "gross9_structural_clock_bundle_core.json"
_STAGED_RECEIPT_NAME = "gross9_structural_clock_bundle_pass_receipt.json"
_PYCACHE_PREFIX_RELATIVE = Path("results/.g9cb-bytecode-cache-disabled")
_PR_SET_PDEATHSIG = 1
_LIBC = ctypes.CDLL(None, use_errno=True)


class TerminalG9CB1Failure(RuntimeError):
    """A terminal protocol or post-sentinel failure."""


def _fail(message: str) -> None:
    raise TerminalG9CB1Failure(message)


def _canonical_json_bytes(payload: Any, *, trailing_lf: bool = True) -> bytes:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
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


def _read_canonical_object(path: Path, hash_field: str | None = None) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        _fail(f"canonical object is absent or unsafe: {path}")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TerminalG9CB1Failure(f"invalid canonical JSON: {path}") from exc
    if not isinstance(payload, dict) or raw != _canonical_json_bytes(payload):
        _fail(f"noncanonical JSON bytes: {path}")
    if hash_field is not None:
        value = payload.get(hash_field)
        if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
            _fail(f"missing {hash_field}: {path}")
        if value != _object_hash(payload, hash_field):
            _fail(f"{hash_field} mismatch: {path}")
    return payload, raw


def _rooted(root: Path, relative: Path) -> Path:
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


def _bound_regular_path(root: Path, path_text: str) -> tuple[Path, bool]:
    if not path_text or "\x00" in path_text:
        _fail("bound input path is empty or contains NUL")
    candidate_text = Path(path_text)
    if candidate_text.is_absolute():
        candidate = Path(path_text)
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise TerminalG9CB1Failure(
                f"bound absolute input cannot be resolved: {path_text}"
            ) from exc
        if path_text != resolved.as_posix():
            _fail(f"bound absolute input is not canonical: {path_text}")
        current = Path(candidate.anchor)
        for part in candidate.parts[1:]:
            current /= part
            try:
                if stat.S_ISLNK(os.lstat(current).st_mode):
                    _fail(f"bound absolute input traverses a symlink: {path_text}")
            except OSError as exc:
                raise TerminalG9CB1Failure(
                    f"bound absolute input cannot be inspected: {path_text}"
                ) from exc
        return resolved, False

    if "\\" in path_text:
        _fail(f"bound repository path is not POSIX text: {path_text}")
    components = path_text.split("/")
    if any(component in ("", ".", "..") for component in components):
        _fail(f"bound repository path is not normalized: {path_text}")
    pure = PurePosixPath(path_text)
    if pure.is_absolute() or pure.as_posix() != path_text:
        _fail(f"bound repository path is not normalized: {path_text}")
    repository_root = root.resolve(strict=True)
    candidate = repository_root.joinpath(*components)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(repository_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise TerminalG9CB1Failure(
            f"bound repository input escapes or is absent: {path_text}"
        ) from exc
    if resolved != candidate:
        _fail(f"bound repository input traverses a symlink: {path_text}")
    current = repository_root
    for component in components:
        current /= component
        try:
            if stat.S_ISLNK(os.lstat(current).st_mode):
                _fail(f"bound repository input traverses a symlink: {path_text}")
        except OSError as exc:
            raise TerminalG9CB1Failure(
                f"bound repository input cannot be inspected: {path_text}"
            ) from exc
    return candidate, True


def _validate_zero_access(payload: Mapping[str, Any]) -> None:
    forbidden_nonzero = {
        "source_value_rows_opened",
        "pre2025_anchor_value_rows_opened",
        "runtime_modules_imported",
        "esdi_runtime_or_private_invocations",
        "model_files_loaded",
        "model_or_history_rows_opened",
        "market_rows_opened",
        "open_interest_rows_opened",
        "funding_rows_opened",
        "premium_rows_opened",
        "outcome_dependent_ohlc_rows_opened",
        "gross9_clock_rows_opened",
        "candidate_rows_opened",
        "comparator_clock_rows_opened",
    }

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in forbidden_nonzero and item != 0:
                    _fail(f"preregistration records preclaim access: {key}={item!r}")
                if key.endswith("_computed") and item is not False:
                    _fail(f"preregistration records prohibited computation: {key}")
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)


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
    ]


def _authority_amendment_bindings(
    preregistration: Mapping[str, Any],
) -> list[dict[str, Any]]:
    bindings = preregistration.get("bindings")
    observed = (
        bindings.get("authority_amendments")
        if isinstance(bindings, Mapping)
        else None
    )
    expected = _expected_authority_amendment_bindings()
    if observed != expected:
        _fail("authority amendment bindings mismatch")
    return expected


def validate_preregistration(
    root: Path = REPOSITORY_ROOT,
    *,
    invoke_prereg_validator: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate the canonical preregistration without opening source values."""

    path = _rooted(root, PREREGISTRATION_PATH)
    payload, raw = _read_canonical_object(path, "manifest_hash")
    if payload.get("identity") != IDENTITY:
        _fail("preregistration identity mismatch")
    _validate_zero_access(payload)
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
    required_binding_keys = {
        "protocol",
        "authority_amendments",
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
    for prohibited in (
        "authority_amendment",
        "adapter_import_roots",
        "adapter_import_closure",
    ):
        if prohibited in bindings:
            _fail(f"superseded preregistration binding is present: {prohibited}")
    _authority_amendment_bindings(payload)
    git_seal = payload.get("git_seal")
    if not isinstance(git_seal, Mapping) or not isinstance(
        git_seal.get("expected_branch"), str
    ):
        _fail("preregistration git_seal is incomplete")

    helper = getattr(prereg, "validate_manifest", None)
    if invoke_prereg_validator and callable(helper):
        try:
            helper(
                payload,
                repository_root=root,
                verify_files=False,
                verify_environment=False,
                verify_git_seal=False,
            )
        except (TypeError, ValueError) as exc:
            raise TerminalG9CB1Failure(
                "preregistration producer validation failed"
            ) from exc

    return payload, {
        "path": PREREGISTRATION_PATH.as_posix(),
        "sha256": _sha256_bytes(raw),
        "manifest_hash": payload["manifest_hash"],
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


def _head_blob_binding(root: Path, path: str) -> dict[str, str]:
    if not _git(root, "ls-files", "--error-unmatch", "--", path, allow_failure=True).strip():
        _fail(f"required protocol path is not tracked: {path}")
    candidate = root / path
    if candidate.is_symlink() or not candidate.is_file():
        _fail(f"required protocol path is not a regular file: {path}")
    blob = _git_text(root, "rev-parse", f"HEAD:{path}")
    mode_line = _git_text(root, "ls-tree", "HEAD", "--", path).split()
    if not mode_line or mode_line[0] != "100644":
        _fail(f"required protocol mode differs: {path}")
    if _git_text(root, "hash-object", "--", path) != blob:
        _fail(f"required protocol path differs from HEAD: {path}")
    return {"path": path, "sha256": _sha256_file(candidate), "git_blob": blob, "mode": "100644"}


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


def _atomic_link_write_once(path: Path, raw: bytes, *, mode: int = 0o444) -> None:
    """Publish complete bytes using an exclusive staging inode and hard link."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"write-once path exists: {path}")
    prefix = f".{path.name}.stage-"
    fd, stage_name = tempfile.mkstemp(prefix=prefix, dir=path.parent)
    stage = Path(stage_name)
    linked = False
    try:
        with os.fdopen(fd, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fchmod(handle.fileno(), mode)
            os.fsync(handle.fileno())
        stage_info = os.stat(stage, follow_symlinks=False)
        os.link(stage, path, follow_symlinks=False)
        path_info = os.stat(path, follow_symlinks=False)
        if (
            stage_info.st_dev != path_info.st_dev
            or stage_info.st_ino != path_info.st_ino
        ):
            _fail(f"published path is not the staged inode: {path}")
        linked = True
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if stage.exists():
            stage.unlink()
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    if not linked:
        _fail(f"publication failed: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        observed = bytearray()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            observed.extend(chunk)
        info = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if bytes(observed) != raw or stat.S_IMODE(info.st_mode) != mode:
        _fail(f"published inode verification failed: {path}")


def validate_claim_preflight(
    root: Path = REPOSITORY_ROOT,
    *,
    invoke_prereg_validator: bool = True,
) -> dict[str, Any]:
    """Run the complete metadata-only claim preflight without writing a claim."""

    preregistration, prereg_binding = validate_preregistration(
        root, invoke_prereg_validator=invoke_prereg_validator
    )
    opaque_inputs = _validate_regular_hashed_inputs(root, preregistration)
    environment = _validate_environment(preregistration, root)
    closures = _validate_static_closures(root, preregistration)
    parent = _require_clean_pushed_branch(root, _expected_branch(preregistration))
    for relative in (
        CLAIM_PATH,
        SENTINEL_PATH,
        *WORKER_LEDGER_PATHS,
        CSV_PATH,
        MANIFEST_PATH,
    ):
        candidate = _rooted(root, relative)
        if candidate.exists() or candidate.is_symlink():
            raise FileExistsError(f"planned artifact already exists: {relative}")
    bindings = [
        _head_blob_binding(root, path)
        for path in _planned_protocol_paths(preregistration)
    ]
    return {
        "preregistration": preregistration,
        "preregistration_binding": prereg_binding,
        "authority_amendments": _authority_amendment_bindings(
            preregistration
        ),
        "opaque_inputs": opaque_inputs,
        "environment": environment,
        "closures": closures,
        "protocol_parent_commit": parent,
        "protocol_bindings": bindings,
    }


def create_claim_only(root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    preflight = validate_claim_preflight(root)
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
    _atomic_link_write_once(_rooted(root, CLAIM_PATH), raw)
    return {
        "path": CLAIM_PATH.as_posix(),
        "sha256": _sha256_bytes(raw),
        "claim_hash": payload["claim_hash"],
        "protocol_parent_commit": parent,
    }


def _validate_claim_commit(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    claim_path = _rooted(root, CLAIM_PATH)
    claim, raw = _read_canonical_object(claim_path, "claim_hash")
    if claim.get("identity") != IDENTITY or claim.get("one_shot") is not True:
        _fail("claim identity or one-shot status mismatch")
    if claim.get("retry_allowed") is not False or claim.get("resume_allowed") is not False:
        _fail("claim permits retry or resume")
    preregistration, prereg_binding = validate_preregistration(root)
    if claim.get("preregistration") != prereg_binding:
        _fail("claim preregistration binding mismatch")
    authority_amendments = _authority_amendment_bindings(preregistration)
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
    head = _require_clean_pushed_branch(root, _expected_branch(preregistration))
    parents = _git_text(root, "rev-list", "--parents", "-n", "1", "HEAD").split()
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
            parent,
            head,
        ).splitlines()
        if line
    ]
    if changed != [f"A\t{CLAIM_PATH.as_posix()}"]:
        _fail("HEAD is not a claim-only direct-child commit")
    if not _git(root, "ls-files", "--error-unmatch", "--", CLAIM_PATH.as_posix(), allow_failure=True).strip():
        _fail("claim is not tracked")
    if _git_text(root, "hash-object", "--", CLAIM_PATH.as_posix()) != _git_text(
        root, "rev-parse", f"HEAD:{CLAIM_PATH.as_posix()}"
    ):
        _fail("claim differs from HEAD")
    for row in protocol_files:
        if not isinstance(row, Mapping) or _head_blob_binding(root, str(row.get("path"))) != row:
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
    return claim, {
        "path": CLAIM_PATH.as_posix(),
        "sha256": _sha256_bytes(raw),
        "claim_hash": claim["claim_hash"],
        "protocol_parent_commit": parent,
        "claim_commit": head,
    }


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


def _module_file_candidates(module: str, root: Path) -> list[Path]:
    if not module:
        return []
    parts = module.split(".")
    found: list[Path] = []
    for index in range(1, len(parts)):
        initializer = Path(*parts[:index]) / "__init__.py"
        if (root / initializer).is_file():
            found.append(initializer)
    module_file = Path(*parts).with_suffix(".py")
    package_file = Path(*parts) / "__init__.py"
    if (root / module_file).is_file():
        found.append(module_file)
    elif (root / package_file).is_file():
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
        paths.update(_module_file_candidates(module, root))
    return paths


def _discover_import_closure(root: Path, entry_paths: Sequence[str]) -> list[Path]:
    pending = {Path(path) for path in entry_paths}
    discovered: set[Path] = set()
    while pending:
        current = min(pending, key=lambda path: path.as_posix())
        pending.remove(current)
        if current in discovered:
            continue
        source_path = root / current
        if source_path.is_symlink() or not source_path.is_file():
            _fail(f"import root or closure member is absent: {current}")
        try:
            tree = ast.parse(
                source_path.read_text(encoding="utf-8"),
                filename=current.as_posix(),
            )
        except (UnicodeDecodeError, SyntaxError) as exc:
            raise TerminalG9CB1Failure(
                f"import closure source cannot be parsed: {current}"
            ) from exc
        discovered.add(current)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                pending.update(_local_import_paths(node, current, root) - discovered)
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


def _read_bound_regular_bytes(path: Path, path_text: str) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TerminalG9CB1Failure(
            f"bound input cannot be opened without following symlinks: {path_text}"
        ) from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            _fail(f"bound input is not a regular file: {path_text}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if len(raw) != info.st_size:
        _fail(f"bound input changed while authenticating: {path_text}")
    return raw, info


def _validate_regular_hashed_inputs(
    root: Path,
    preregistration: Mapping[str, Any],
    *,
    verify_git: bool = True,
) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    declarations: dict[str, dict[str, Any]] = {}
    for binding in _iter_bindings(preregistration):
        path_text = _binding_path(binding)
        digest = binding.get("sha256")
        if not isinstance(digest, str) or not _SHA_RE.fullmatch(digest):
            _fail(f"bound input SHA-256 is not lowercase hexadecimal: {path_text}")
        candidate, repository_relative = _bound_regular_path(root, path_text)
        raw, info = _read_bound_regular_bytes(candidate, path_text)
        actual = _sha256_bytes(raw)
        if actual != digest:
            _fail(f"bound input hash mismatch: {path_text}")
        size = info.st_size
        if "size_bytes" in binding:
            declared_size = binding["size_bytes"]
            if (
                type(declared_size) is not int
                or declared_size < 0
                or declared_size != size
            ):
                _fail(f"bound input size mismatch: {path_text}")
        if "path_type" in binding and binding["path_type"] != "regular_file":
            _fail(f"bound input path_type mismatch: {path_text}")

        declared = declarations.setdefault(path_text, {})
        for key in ("path_type", "git_blob", "git_mode"):
            if key in binding:
                value = binding[key]
                if key in declared and declared[key] != value:
                    _fail(
                        f"conflicting duplicate input metadata: {path_text}:{key}"
                    )
                declared[key] = value

        if verify_git and repository_relative and (
            "git_blob" in binding or "git_mode" in binding
        ):
            if not isinstance(binding.get("git_blob"), str) or not isinstance(
                binding.get("git_mode"), str
            ):
                _fail(f"incomplete bound input Git metadata: {path_text}")
            tree = _git_text(root, "ls-tree", "HEAD", "--", path_text).split()
            if (
                len(tree) < 3
                or tree[0] != binding["git_mode"]
                or tree[1] != "blob"
                or tree[2] != binding["git_blob"]
            ):
                _fail(f"bound input Git metadata mismatch: {path_text}")

        record = {"path": path_text, "sha256": actual, "size_bytes": size}
        prior = seen.get(path_text)
        if prior is not None:
            if prior != record:
                _fail(f"conflicting duplicate input binding: {path_text}")
            continue
        seen[path_text] = record
    if not seen:
        _fail("preregistration exposed no path/hash bindings")
    return [seen[path] for path in sorted(seen)]


def _validate_one_static_closure(
    root: Path,
    preregistration: Mapping[str, Any],
    *,
    roots_key: str,
    closure_key: str,
    verify_git: bool,
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
    discovered = _discover_import_closure(root, roots)
    if [path.as_posix() for path in discovered] != [
        str(row.get("path")) for row in members if isinstance(row, Mapping)
    ]:
        _fail(f"{closure_key} independently discovered path set differs")
    result: list[dict[str, Any]] = []
    for row in members:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            _fail("invalid import closure member")
        path = root / row["path"]
        if path.is_symlink() or not path.is_file():
            _fail(f"closure member is absent: {row['path']}")
        raw = path.read_bytes()
        try:
            ast.parse(raw, filename=row["path"])
        except SyntaxError as exc:
            raise TerminalG9CB1Failure(f"closure source cannot be parsed: {row['path']}") from exc
        observed = {
            "path": row["path"],
            "path_type": "regular_file",
            "sha256": _sha256_bytes(raw),
            "git_blob": (
                _git_text(root, "rev-parse", f"HEAD:{row['path']}")
                if verify_git
                else row.get("git_blob")
            ),
            "git_mode": "100644",
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
        not candidate.name.startswith(".gross9-structural-clock-worker-")
        or candidate.name == ".gross9-structural-clock-worker-"
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
) -> dict[str, Any]:
    if type(slot) is not int or slot not in (1, 2):
        _fail("worker capability slot is invalid")
    if type(parent_pid) is not int or parent_pid <= 0:
        _fail("worker capability parent PID is invalid")
    stage_directory = _worker_stage_path(root, output_dir)
    read_fd = write_fd = -1
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
        }
        return {
            "row": row,
            "read_fd": read_fd,
            "token": token,
            "stage_path": Path(root).resolve() / stage_directory,
        }
    except BaseException:
        if write_fd >= 0:
            os.close(write_fd)
        if read_fd >= 0:
            os.close(read_fd)
        _zero_token(token)
        raise


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
            or not stage.startswith("results/.gross9-structural-clock-worker-")
            or row["carrier_kind"] != "anonymous_pipe_v1"
            or type(row["carrier_device"]) is not int
            or type(row["carrier_inode"]) is not int
            or row["carrier_device"] < 0
            or row["carrier_inode"] <= 0
            or not isinstance(row["token_sha256"], str)
            or not _SHA_RE.fullmatch(row["token_sha256"])
            or row["consumed_ledger_path"]
            != WORKER_LEDGER_PATHS[row["slot"] - 1].as_posix()
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
    return rows


def _sentinel_payload(
    claim_binding: Mapping[str, Any],
    prereg_binding: Mapping[str, Any],
    authority_amendments: Sequence[Mapping[str, Any]],
    parent_authentication_sha256: str,
    worker_capabilities: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    capabilities = _normalized_worker_capabilities(worker_capabilities)
    amendments = [dict(row) for row in authority_amendments]
    if amendments != _expected_authority_amendment_bindings():
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
    repository_root = root.resolve(strict=True)
    prefix = repository_root / _PYCACHE_PREFIX_RELATIVE
    if prefix.exists() or prefix.is_symlink():
        _fail("fixed worker bytecode-cache prefix already exists")
    for current, directories, files in os.walk(
        repository_root,
        topdown=True,
        followlinks=False,
    ):
        current_path = Path(current)
        if current_path == repository_root / ".git":
            directories[:] = []
            continue
        if "__pycache__" in directories:
            _fail(f"repository bytecode-cache directory exists: {current_path}")
        pyc = next((name for name in files if name.endswith(".pyc")), None)
        if pyc is not None:
            _fail(f"repository bytecode file exists: {current_path / pyc}")


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
        self._original_readlink = os.readlink
        self._original_realpath = os.path.realpath

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
            raise TerminalG9CB1Failure("guarded path is not path-like") from exc
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
                    raise TerminalG9CB1Failure(
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
            self._reject_dir_fds(kwargs)
            tmpfile = getattr(os, "O_TMPFILE", 0)
            mutation = bool(flags & mutation_mask) or bool(
                tmpfile and (flags & tmpfile) == tmpfile
            )
            canonical = self._checked_path(
                path,
                mutation=mutation,
                fifo_open=True,
            )
            descriptor = original(path, flags, *args, **kwargs)
            info = os.fstat(descriptor)
            is_directory = stat.S_ISDIR(info.st_mode)
            if stat.S_ISFIFO(info.st_mode):
                os.close(descriptor)
                self.unauthorized_write_or_ipc_events += 1
                _fail(f"path-resolved FIFO access is forbidden: {canonical}")
            if is_directory and (
                info.st_dev,
                info.st_ino,
            ) not in self.allowed_directory_identities:
                os.close(descriptor)
                self.unauthorized_write_or_ipc_events += 1
                _fail(f"directory descriptor is not authorized: {canonical}")
            self.descriptors[descriptor] = (
                canonical,
                int(info.st_dev),
                int(info.st_ino),
                is_directory,
            )
            return descriptor

        return guarded

    def _wrap_path_observation(self, original: Any, *, path_index: int = 0):
        def guarded(*args: Any, **kwargs: Any) -> Any:
            self._reject_dir_fds(kwargs)
            if len(args) <= path_index:
                path = "."
            else:
                path = args[path_index]
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
                    self._checked_path(arguments[index])
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

    def authorize_ledger(self, staging: Path, canonical: Path) -> None:
        if self.own_ledger is None or canonical != self.own_ledger:
            _fail("worker attempted to authorize a non-owned ledger")
        self.allowed_mutations.update(
            {
                self._checked_path(staging),
                self._checked_path(canonical),
            }
        )

    def bind_ledger_slot(self, slot: int) -> None:
        if type(slot) is not int or slot not in (1, 2):
            _fail("worker guard slot binding is invalid")
        if self.own_ledger is not None:
            _fail("worker guard ledger slot was already bound")
        self.own_ledger = self.ledger_paths[slot - 1]
        self.other_ledger = self.ledger_paths[1 - (slot - 1)]
        self._forbidden_ledgers = {self.other_ledger}

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
        info = self._original_lstat(path)
        if not stat.S_ISDIR(info.st_mode):
            _fail("worker durability target is not a directory")
        self.allowed_directory_identities.add(
            (int(info.st_dev), int(info.st_ino))
        )

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
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        _write_all(descriptor, raw)
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        info = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    _sync_guarded_directory(sync_directory)
    observed, reopened = _read_guarded_file(path, expected_mode=mode)
    if observed != raw or (
        reopened.st_dev,
        reopened.st_ino,
    ) != (
        info.st_dev,
        info.st_ino,
    ):
        _fail(f"exclusive worker output verification failed: {path}")
    return reopened


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


def _publish_worker_ledger(
    *,
    guard: _WorkerIsolationGuard,
    binding: Mapping[str, Any],
    claim: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    sentinel: Mapping[str, Any],
    authority_amendments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    canonical = _rooted(
        guard.root,
        Path(str(binding["consumed_ledger_path"])),
    )
    staging = canonical.with_name(
        f".{canonical.name}.stage-{os.getpid()}-{binding['slot']}"
    )
    guard.authorize_directory_sync(canonical.parent)
    guard.authorize_ledger(staging, canonical)
    for path in (staging, canonical):
        try:
            os.lstat(path)
        except FileNotFoundError as exc:
            if exc.errno != errno.ENOENT:
                _fail(f"ledger absence check returned wrong errno: {path}")
        else:
            _fail(f"worker ledger path already exists: {path}")
    payload = _worker_ledger_payload(
        binding=binding,
        claim=claim,
        preregistration=preregistration,
        sentinel=sentinel,
        authority_amendments=authority_amendments,
    )
    raw = _canonical_json_bytes(payload)
    descriptor = os.open(
        staging,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        _write_all(descriptor, raw)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        staged_info = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    os.link(staging, canonical, follow_symlinks=False)
    _sync_guarded_directory(canonical.parent)
    os.unlink(staging)
    _sync_guarded_directory(canonical.parent)
    observed, canonical_info = _read_guarded_file(
        canonical,
        expected_mode=0o444,
    )
    if (
        observed != raw
        or not stat.S_ISREG(canonical_info.st_mode)
        or (canonical_info.st_dev, canonical_info.st_ino)
        != (staged_info.st_dev, staged_info.st_ino)
    ):
        _fail("worker consumption ledger publication differs")
    return {
        "payload": payload,
        "raw": raw,
        "sha256": _sha256_bytes(raw),
        "device": int(canonical_info.st_dev),
        "inode": int(canonical_info.st_ino),
    }


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


def _parse_timestamp(value: str) -> int:
    if not isinstance(value, str) or not _TIMESTAMP_RE.fullmatch(value):
        _fail(f"invalid UTC-second timestamp: {value!r}")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise TerminalG9CB1Failure(f"invalid timestamp: {value}") from exc
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
        raise TerminalG9CB1Failure(f"invalid decimal {field}") from exc
    if not result.is_finite():
        _fail(f"nonfinite decimal {field}")
    return result


def _empty_counters() -> dict[str, Any]:
    logical_sources = (
        "market_5m",
        "funding",
        "premium",
        "open_interest",
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
            return index + 1, "stop"
        if take_hit:
            return index + 1, "take"
    return horizon, "fixed"


def reconstruct_intervals(
    bars: Sequence[Mapping[str, Any]],
    *,
    domain_start: str = DOMAIN_START,
    domain_end: str = DOMAIN_END,
    counters: dict[str, Any] | None = None,
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
    time_to_index = {second: index for index, (second, _) in enumerate(normalized)}
    if len(time_to_index) != len(normalized):
        _fail("duplicate market timestamp")

    output: list[dict[str, Any]] = []
    for sleeve in SLEEVES:
        name = sleeve["name"]
        last_exit = start
        index_in_sleeve = 0
        for signal_index, (signal_time, row) in enumerate(normalized):
            if signal_time < start - 300 or signal_time >= end:
                continue
            decisions = row.get("decisions", {})
            if not isinstance(decisions, Mapping):
                _fail("decisions must be an object")
            decision = decisions.get(name)
            audit["per_sleeve"][name]["signal_rows_evaluated"] += 1
            if decision is None:
                continue
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
                if long_gate not in (True, False) or short_gate not in (True, False):
                    _fail("fresh decision lacks exact gate booleans")
                if long_gate == short_gate:
                    _fail("fresh active decision violates exclusive gates")
                expected_side = 1 if long_gate else -1
                if side != expected_side:
                    _fail("fresh side differs from exclusive gate")
            entry_index = signal_index + 1
            if entry_index >= len(normalized):
                continue
            entry_time = normalized[entry_index][0]
            if entry_time < start or entry_time >= end or entry_time < last_exit:
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
            exit_index = entry_index + hold
            if exit_index > len(normalized):
                continue
            if (
                exit_index < len(normalized)
                and normalized[exit_index][0] > end
            ):
                continue
            if kind in ("rank7", "barrier"):
                exit_index, exit_kind = _barrier_exit(
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
            exit_time = (
                normalized[exit_index][0]
                if exit_index < len(normalized)
                else normalized[-1][0] + 300
            )
            if not (start <= entry_time < exit_time <= end):
                _fail(f"interval outside domain for {name}")
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
        raise TerminalG9CB1Failure("invalid gzip stream") from exc
    if compress_csv(decompressed) != raw:
        _fail("gzip bytes are not canonical")
    try:
        text = decompressed.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TerminalG9CB1Failure("CSV is not UTF-8") from exc
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
) -> dict[str, Any]:
    csv_bytes = gzip.decompress(csv_gzip)
    expected_amendments = _expected_authority_amendment_bindings()
    if [dict(row) for row in authority_amendments] != expected_amendments:
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
            "authority_amendments": expected_amendments,
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


def _read_synthetic_worker_input(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload, _ = _read_canonical_object(path)
    bars = payload.get("bars")
    if not isinstance(bars, list):
        _fail("synthetic worker input has no bars")
    counters = _empty_counters()
    supplied = payload.get("physical_counters")
    if isinstance(supplied, Mapping):
        for section in ("file_access", "rows_decoded", "rows_used"):
            if isinstance(supplied.get(section), Mapping):
                counters[section].update(supplied[section])
    return [dict(row) for row in bars], counters


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
            raise TerminalG9CB1Failure(
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
        raise TerminalG9CB1Failure(f"worker JSON input is invalid: {path}") from exc
    if not isinstance(value, dict):
        _fail(f"worker JSON input is not an object: {path}")
    return value


def _market_seconds(market: Any) -> tuple[list[int], list[Any]]:
    try:
        values = list(market["date"])
    except (KeyError, TypeError) as exc:
        raise TerminalG9CB1Failure("generic market lacks date rows") from exc
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


def _generic_time_second(value: Any, pandas_module: Any) -> int:
    try:
        parsed = pandas_module.Timestamp(
            pandas_module.to_datetime(value, utc=True, errors="raise")
        )
    except (TypeError, ValueError) as exc:
        raise TerminalG9CB1Failure("generic source timestamp is invalid") from exc
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
            raise TerminalG9CB1Failure(f"invalid JSONL row: {path}") from exc
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
            raise TerminalG9CB1Failure(
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
            raise TerminalG9CB1Failure(
                "chunked or streaming CSV decode is forbidden"
            ) from exc
        first_ordinal = counters["rows_decoded"][logical_name]
        counters["rows_decoded"][logical_name] += int(decoded_rows)
        try:
            frame.attrs["_g9cb_parser_ordinals"] = tuple(
                range(first_ordinal, first_ordinal + int(decoded_rows))
            )
            frame.attrs["_g9cb_logical_source"] = logical_name
        except (AttributeError, TypeError) as exc:
            raise TerminalG9CB1Failure(
                "decoded CSV frame cannot carry parser ordinals"
            ) from exc
        return frame

    pandas_module.read_csv = counted_read_csv
    return original


def _load_rank7_funding(
    path: str,
    cutoff: str,
    pandas_module: Any,
) -> Any:
    frame = pandas_module.read_csv(path, compression="infer")
    required = {"date", "funding_rate"}
    if not required.issubset(frame.columns):
        _fail("Rank7 funding source lacks required columns")
    frame = frame[["date", "funding_rate"]].copy()
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
    return frame


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
        if signal < 0 or entry < 0 or horizon >= len(self._open):
            self._cache[key] = None
            return None
        entry_price = float(self._open[entry])
        if not entry_price > 0.0:
            _fail(f"{self._sleeve} structural entry price is invalid")
        take_enabled = take_bps < 1_000_000
        stop_enabled = stop_bps < 1_000_000
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
            exclude_from="2026-06-02",
        )
    finally:
        primitives.normalise_market = original_normalise
        primitives.attach_binance_um_aux_frames = original_attach_aux

    open_interest = pd.read_csv(sources["open_interest"], compression="infer")
    causal_rows.handoff_frame("open_interest", open_interest)
    market = primitives.attach_open_interest(market, open_interest)
    seconds, _ = _market_seconds(market)
    if _parse_timestamp(DOMAIN_END) not in seconds:
        _fail("bound market lacks the canonical domain-end boundary row")
    dates = pd.to_datetime(market["date"])
    masks = {
        name: np.asarray(
            (dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end)),
            dtype=bool,
        )
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
        max(0, len(market) - markov_hold - 2),
        markov_stride,
        dtype=np.int64,
    )
    for mask in masks.values():
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
            if exit_position >= len(mask) or not bool(mask[exit_position]):
                continue
            _append_direct_interval(
                intervals,
                counters,
                "markov_transition_long",
                seconds[entry],
                seconds[exit_position],
                1,
                "fixed",
            )
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
        for mask in masks.values():
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
                if source_second % 300 or source_second != seconds[position]:
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
                if exit_position >= len(mask) or not bool(mask[exit_position]):
                    continue
                _append_direct_interval(
                    intervals,
                    counters,
                    sleeve_name,
                    seconds[entry],
                    seconds[exit_position],
                    side,
                    "fixed",
                )
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
        len(market) - int(fresh_cfg["hold_bars"]) - 2,
        int(fresh_cfg["stride_bars"]),
        dtype=np.int64,
    )
    fresh_active = np.logical_xor(fresh_long_active, fresh_short_active)
    for mask in masks.values():
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
            if trade is None or not bool(mask[int(trade.exit_position)]):
                continue
            exit_boundary = int(trade.exit_position) + (
                0 if trade.exit_kind == "fixed" else 1
            )
            _append_direct_interval(
                intervals,
                counters,
                "fresh_kimchi_fx",
                seconds[int(trade.entry_position)],
                seconds[exit_boundary],
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
    if rank7_seconds != seconds:
        _fail("Rank7 generic context market grid differs")
    structural_funding = _load_rank7_funding(
        sources["funding"],
        "2026-06-02",
        pd,
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
            exits.append(len(seconds))
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
        .iloc[np.minimum(exits, len(seconds) - 1)]
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
    for mask in masks.values():
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
            if trade is None or not bool(mask[int(trade.exit_position)]):
                continue
            exit_boundary = int(trade.exit_position) + (
                0 if trade.exit_kind == "fixed" else 1
            )
            _append_direct_interval(
                intervals,
                counters,
                "frozen_annual_rank7",
                seconds[int(trade.entry_position)],
                seconds[exit_boundary],
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
            raise TerminalG9CB1Failure(
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

        sentinel, sentinel_raw = _read_canonical_object(
            _rooted(root, SENTINEL_PATH), "manifest_hash"
        )
        claim, claim_raw = _read_canonical_object(
            _rooted(root, CLAIM_PATH), "claim_hash"
        )
        preregistration, prereg_binding = validate_preregistration(
            root,
            invoke_prereg_validator=not synthetic,
        )
        authority_amendments = _authority_amendment_bindings(
            preregistration
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

        if (
            output_dir.is_symlink()
            or not output_dir.is_dir()
            or list(output_dir.iterdir())
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
        own_ledger = _rooted(
            root, Path(str(binding["consumed_ledger_path"]))
        )
        try:
            os.lstat(own_ledger)
        except FileNotFoundError as exc:
            if exc.errno != errno.ENOENT:
                _fail("worker own-ledger absence check returned wrong errno")
        else:
            _fail("worker own consumption ledger already exists")

        if synthetic:
            if set(parent_authentication) != {
                "environment",
                "hashed_inputs",
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
            worker_closures = _validate_static_closures(
                root,
                preregistration,
                verify_git=False,
            )
            worker_authentication = {
                "environment": _validate_environment(preregistration, root),
                "hashed_inputs": _validate_regular_hashed_inputs(
                    root,
                    preregistration,
                    verify_git=False,
                ),
                "runtime_import_closure": worker_closures["runtime"],
            }
        if worker_authentication != parent_authentication:
            _fail("worker authentication differs from the parent seal")

        token = _consume_worker_capability(capability_fd, binding)
        capability_closed = True
        ledger = _publish_worker_ledger(
            guard=guard,
            binding=binding,
            claim=claim,
            preregistration=preregistration,
            sentinel=sentinel,
            authority_amendments=authority_amendments,
        )

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
        elif (root / ".git").exists():
            _fail("synthetic worker hooks are forbidden in a canonical repository")

        rebuild_invocations_started += 1
        if synthetic:
            synthetic_path = Path(str(arguments.synthetic_input))
            bars, counters = _read_synthetic_worker_input(synthetic_path)
            rows, counters = reconstruct_intervals(bars, counters=counters)
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
        )
        core_raw = _canonical_json_bytes(core)
        rebuild_invocations_completed += 1

        if (
            not output_dir.is_dir()
            or output_dir.is_symlink()
            or list(output_dir.iterdir())
        ):
            _fail("worker staging directory changed before output writing")
        ledger_raw, ledger_info = _read_guarded_file(
            own_ledger,
            expected_mode=0o444,
        )
        if (
            ledger_raw != ledger["raw"]
            or (ledger_info.st_dev, ledger_info.st_ino)
            != (ledger["device"], ledger["inode"])
        ):
            _fail("worker consumption ledger changed before output writing")

        guard.authorize_directory_sync(output_dir)
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
        observed_csv, _ = _read_guarded_file(csv_path, expected_mode=0o400)
        observed_core, _ = _read_guarded_file(core_path, expected_mode=0o400)
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


def _prepare_worker(
    *,
    root: Path,
    capability: dict[str, Any],
    other_stage_directory: str,
    synthetic_input: Path | None,
    parent_authentication: Mapping[str, Any],
) -> dict[str, Any]:
    row = capability["row"]
    parent_auth_json = _canonical_json_bytes(
        dict(parent_authentication), trailing_lf=False
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
        str(capability["read_fd"]),
        "--expected-parent-pid",
        str(row["parent_pid"]),
        "--parent-auth-json",
        parent_auth_json,
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
    descriptor = int(capability["read_fd"])
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            invocation["command"],
            cwd=invocation["cwd"],
            env=invocation["environment"],
            close_fds=True,
            pass_fds=(descriptor,),
        )
        try:
            os.close(descriptor)
        except BaseException:
            process.terminate()
            process.wait()
            raise
        capability["read_fd"] = -1
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


def _run_worker(
    *,
    root: Path,
    capability: dict[str, Any],
    other_stage_directory: str,
    synthetic_input: Path | None,
    parent_authentication: Mapping[str, Any],
) -> int:
    return _execute_prepared_worker(
        _prepare_worker(
            root=root,
            capability=capability,
            other_stage_directory=other_stage_directory,
            synthetic_input=synthetic_input,
            parent_authentication=parent_authentication,
        )
    )

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
) -> dict[str, Any]:
    binding = capability["row"]
    stage = capability["stage_path"]
    ledger_path = _rooted(
        root, Path(str(binding["consumed_ledger_path"]))
    )
    ledger_raw, ledger_info = _read_bound_regular_bytes(
        ledger_path, str(binding["consumed_ledger_path"])
    )
    if stat.S_IMODE(ledger_info.st_mode) != 0o444:
        _fail("worker consumption ledger mode differs")
    try:
        ledger_payload = json.loads(
            ledger_raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TerminalG9CB1Failure(
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

    if stage.is_symlink() or not stage.is_dir():
        _fail("worker stage is absent after successful exit")
    expected_names = {
        _STAGED_CSV_NAME,
        _STAGED_CORE_NAME,
        _STAGED_RECEIPT_NAME,
    }
    if {path.name for path in stage.iterdir()} != expected_names:
        _fail("worker stage output inventory differs")
    csv_raw, _ = _read_bound_regular_bytes(
        stage / _STAGED_CSV_NAME, _STAGED_CSV_NAME
    )
    core_raw, _ = _read_bound_regular_bytes(
        stage / _STAGED_CORE_NAME, _STAGED_CORE_NAME
    )
    receipt_raw, receipt_info = _read_bound_regular_bytes(
        stage / _STAGED_RECEIPT_NAME, _STAGED_RECEIPT_NAME
    )
    for name in expected_names:
        info = os.stat(stage / name, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o400:
            _fail(f"worker staged file mode differs: {name}")
    try:
        receipt = json.loads(
            receipt_raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TerminalG9CB1Failure("worker receipt is invalid") from exc
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


def _cleanup_successful_stage(stage: Path, results_directory: Path) -> None:
    for name in (
        _STAGED_CSV_NAME,
        _STAGED_CORE_NAME,
        _STAGED_RECEIPT_NAME,
    ):
        path = stage / name
        if path.is_symlink() or not path.is_file():
            _fail(f"worker stage cleanup target differs: {path}")
        path.unlink()
    directory_fd = os.open(
        stage, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    stage.rmdir()
    directory_fd = os.open(
        results_directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _create_stage_directory(stage: Path, results_directory: Path) -> None:
    if stage.exists() or stage.is_symlink():
        _fail(f"reserved worker stage already exists: {stage}")
    os.mkdir(stage, 0o700)
    os.chmod(stage, 0o700)
    info = os.stat(stage, follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700:
        _fail("worker stage directory mode differs")
    directory_fd = os.open(
        results_directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


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
        raise TerminalG9CB1Failure("worker core is invalid JSON") from exc
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


def produce_one_shot(
    root: Path = REPOSITORY_ROOT,
    *,
    synthetic_input: Path | None = None,
) -> dict[str, Any]:
    """Authenticate, consume exactly two carriers, and publish manifest-last."""

    root = root.resolve()
    if synthetic_input is not None and (root / ".git").exists():
        _fail("synthetic production hook requires a noncanonical repository")
    claim, claim_binding = _validate_claim_commit(root)
    preregistration, prereg_binding = validate_preregistration(root)
    authority_amendments = _authority_amendment_bindings(preregistration)
    if claim.get("authority_amendments") != authority_amendments:
        _fail("claim authority amendments differ before production")
    environment = _validate_environment(preregistration, root)
    inputs = _validate_regular_hashed_inputs(root, preregistration)
    if claim.get("opaque_inputs_authenticated") != inputs:
        _fail("current hashed inputs differ from the immutable claim")
    closures = _validate_static_closures(root, preregistration)
    parent_authentication = {
        "environment": environment,
        "hashed_inputs": inputs,
        "runtime_import_closure": closures["runtime"],
    }
    parent_authentication_sha256 = _sha256_bytes(
        _canonical_json_bytes(
            parent_authentication, trailing_lf=False
        )
    )
    _validate_bytecode_preflight(root)

    results_directory = _rooted(root, SENTINEL_PATH).parent
    staging_patterns = [
        ".gross9-structural-clock-worker-*",
        f".{SENTINEL_PATH.name}.stage-*",
        f".{CSV_PATH.name}.stage-*",
        f".{MANIFEST_PATH.name}.stage-*",
        *[f".{path.name}.stage-*" for path in WORKER_LEDGER_PATHS],
    ]
    leftovers = [
        path
        for pattern in staging_patterns
        for path in results_directory.glob(pattern)
    ]
    if leftovers:
        _fail("leftover pre-access publication staging path exists")
    for path in WORKER_LEDGER_PATHS:
        candidate = _rooted(root, path)
        if candidate.exists() or candidate.is_symlink():
            _fail(f"worker consumption ledger already exists: {path}")

    while True:
        suffixes = (os.urandom(12).hex(), os.urandom(12).hex())
        if suffixes[0] == suffixes[1]:
            continue
        stages = tuple(
            results_directory
            / f".gross9-structural-clock-worker-{suffix}"
            for suffix in suffixes
        )
        if not any(path.exists() or path.is_symlink() for path in stages):
            break
    stage_one, stage_two = stages
    _create_stage_directory(stage_one, results_directory)
    if stage_two.exists() or stage_two.is_symlink():
        _fail("slot-2 reserved stage exists before slot 1")

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
            )
        )
        capabilities.append(
            _prepare_worker_capability(
                root=root,
                output_dir=stage_two,
                slot=2,
                parent_pid=parent_pid,
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
        )
        invocation_two = _prepare_worker(
            root=root,
            capability=capabilities[1],
            other_stage_directory=rows[0]["stage_directory"],
            synthetic_input=synthetic_input,
            parent_authentication=parent_authentication,
        )
        sentinel = _sentinel_payload(
            claim_binding,
            prereg_binding,
            authority_amendments,
            parent_authentication_sha256,
            rows,
        )
        sentinel_raw = _canonical_json_bytes(sentinel)
        _atomic_link_write_once(
            _rooted(root, SENTINEL_PATH), sentinel_raw
        )
        sentinel_published = True

        worker_one_pid = _execute_prepared_worker(invocation_one)
        pass_one = _validate_worker_ledger_and_receipt(
            root=root,
            capability=capabilities[0],
            observed_worker_pid=worker_one_pid,
            claim=claim,
            preregistration=preregistration,
            sentinel=sentinel,
            authority_amendments=authority_amendments,
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
        _cleanup_successful_stage(stage_one, results_directory)
        if stage_two.exists() or stage_two.is_symlink():
            _fail("slot-2 reserved stage changed during slot 1")
        _create_stage_directory(stage_two, results_directory)

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
            _rooted(root, CSV_PATH), pass_one["csv_raw"]
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
            _rooted(root, MANIFEST_PATH), manifest_raw
        )
        _cleanup_successful_stage(stage_two, results_directory)
        return {
            "identity": IDENTITY,
            "rows": len(rows_one),
            "csv_gzip_sha256": _sha256_bytes(pass_one["csv_raw"]),
            "manifest_hash": manifest["manifest_hash"],
            "terminal_on_any_later_failure": TERMINAL_ACTION,
        }
    except BaseException as exc:
        for capability in capabilities:
            descriptor = int(capability.get("read_fd", -1))
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                capability["read_fd"] = -1
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
        if isinstance(exc, TerminalG9CB1Failure):
            raise
        raise TerminalG9CB1Failure(f"{TERMINAL_ACTION}: {exc}") from exc

def _raw_worker_option(arguments: Sequence[str], name: str) -> str:
    positions = [
        index for index, value in enumerate(arguments) if value == name
    ]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        _fail(f"internal worker raw option is absent or repeated: {name}")
    return arguments[positions[0] + 1]


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
    root_text = _raw_worker_option(arguments, "--repository-root")
    own_stage_text = _raw_worker_option(arguments, "--output-dir")
    other_stage_text = _raw_worker_option(
        arguments, "--other-stage-directory"
    )
    guard = _WorkerIsolationGuard(
        root=Path(root_text),
        own_stage=own_stage_text,
        other_stage=other_stage_text,
        ledger_paths=WORKER_LEDGER_PATHS,
    )
    guard.install()
    return guard


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--create-claim", action="store_true")
    actions.add_argument("--produce", action="store_true")
    actions.add_argument("--internal-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repository-root", default=str(REPOSITORY_ROOT), help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", help=argparse.SUPPRESS)
    parser.add_argument("--other-stage-directory", help=argparse.SUPPRESS)
    parser.add_argument("--worker-capability-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--expected-parent-pid", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--parent-auth-json", default="{}", help=argparse.SUPPRESS)
    parser.add_argument("--synthetic-input", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
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
            or type(arguments.expected_parent_pid) is not int
            or arguments.expected_parent_pid <= 0
        ):
            _fail("internal worker arguments are incomplete")
        return _worker_main(arguments, guard)
    if guard is not None:
        _fail("worker bootstrap was installed for a non-worker action")
    if arguments.synthetic_input:
        _fail("synthetic input is accepted only by an authenticated internal worker")
    if arguments.create_claim:
        create_claim_only(Path(arguments.repository_root))
    else:
        produce_one_shot(Path(arguments.repository_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
