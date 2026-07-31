from __future__ import annotations

import ast
import copy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import gzip
import hashlib
import hmac
import inspect
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import stat
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from training import build_gross9_structural_clock_bundle as builder


SOURCE_KEYS = (
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
SLEEVE_COUNTER_KEYS = (
    "signal_rows_evaluated",
    "intervals_emitted",
    "long_intervals",
    "short_intervals",
    "fixed_horizon_exits",
    "take_exits",
    "stop_exits",
    "outcome_dependent_ohlc_rows_examined",
)
CAPABILITY_KEYS = (
    "slot",
    "parent_pid",
    "stage_directory",
    "carrier_kind",
    "carrier_device",
    "carrier_inode",
    "token_sha256",
    "consumed_ledger_path",
)
LEDGER_KEYS = (
    "identity",
    "protocol_version",
    "slot",
    "parent_pid",
    "stage_directory",
    "carrier_kind",
    "carrier_device",
    "carrier_inode",
    "token_sha256",
    "claim_hash",
    "preregistration_manifest_hash",
    "sentinel_manifest_hash",
    "authority_amendments",
    "status",
)
RECEIPT_KEYS = (
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
)
WORKER_ENVIRONMENT_KEYS = (
    "BLIS_NUM_THREADS",
    "CUDA_VISIBLE_DEVICES",
    "LANG",
    "LC_ALL",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PYTHONHASHSEED",
    "PYTHONIOENCODING",
    "PYTHONNOUSERSITE",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONPATH",
    "PYTHONPYCACHEPREFIX",
    "PYTHONUNBUFFERED",
    "PYTHONUTF8",
    "TZ",
    "VECLIB_MAXIMUM_THREADS",
)
PROCESS_DESCRIPTOR_CALLABLES = (
    "os.dup",
    "os.dup2",
    "fcntl.fcntl:F_DUPFD",
    "fcntl.fcntl:F_DUPFD_CLOEXEC",
    "os.chdir",
    "os.fchdir",
    "os.chroot",
    "os.fork",
    "os.forkpty",
    "os.posix_spawn",
    "os.posix_spawnp",
    "os.execl",
    "os.execle",
    "os.execlp",
    "os.execlpe",
    "os.execv",
    "os.execve",
    "os.execvp",
    "os.execvpe",
    "os.spawnl",
    "os.spawnle",
    "os.spawnlp",
    "os.spawnlpe",
    "os.spawnv",
    "os.spawnve",
    "os.spawnvp",
    "os.spawnvpe",
    "os.system",
    "os.popen",
    "subprocess.Popen",
    "subprocess.run",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "multiprocessing.process.BaseProcess.start",
    "concurrent.futures.ProcessPoolExecutor",
)
PATH_OBSERVATION_CALLABLES = (
    "builtins.open",
    "io.open",
    "os.open",
    "os.stat",
    "os.lstat",
    "os.access",
    "os.readlink",
    "os.listdir",
    "os.scandir",
    "os.walk",
    "os.path.exists",
    "os.path.lexists",
    "os.path.isfile",
    "os.path.isdir",
    "os.path.islink",
    "os.path.getsize",
    "os.path.realpath",
    "pathlib.Path.open",
    "pathlib.Path.read_text",
    "pathlib.Path.read_bytes",
    "pathlib.Path.write_text",
    "pathlib.Path.write_bytes",
    "pathlib.Path.stat",
    "pathlib.Path.lstat",
    "pathlib.Path.exists",
    "pathlib.Path.is_file",
    "pathlib.Path.is_dir",
    "pathlib.Path.is_symlink",
    "pathlib.Path.iterdir",
    "pathlib.Path.glob",
    "pathlib.Path.rglob",
    "pathlib.Path.mkdir",
    "pathlib.Path.touch",
    "pathlib.Path.chmod",
    "pathlib.Path.unlink",
    "pathlib.Path.rename",
    "pathlib.Path.replace",
    "pathlib.Path.symlink_to",
    "pathlib.Path.hardlink_to",
)
FILESYSTEM_MUTATION_CALLABLES = (
    "builtins.open:w",
    "builtins.open:a",
    "builtins.open:x",
    "builtins.open:+",
    "io.open:w",
    "io.open:a",
    "io.open:x",
    "io.open:+",
    "os.open:O_WRONLY",
    "os.open:O_RDWR",
    "os.open:O_CREAT",
    "os.open:O_EXCL",
    "os.open:O_TRUNC",
    "os.open:O_APPEND",
    "os.open:O_TMPFILE",
    "os.write",
    "os.pwrite",
    "os.writev",
    "os.pwritev",
    "os.copy_file_range",
    "os.sendfile",
    "os.splice",
    "os.ftruncate",
    "os.posix_fallocate",
    "os.fchmod",
    "os.fchown",
    "os.fsync",
    "os.fdatasync",
    "os.setxattr",
    "os.removexattr",
    "os.mkdir",
    "os.makedirs",
    "os.mkfifo",
    "os.mknod",
    "os.link",
    "os.symlink",
    "os.rename",
    "os.replace",
    "os.remove",
    "os.unlink",
    "os.rmdir",
    "os.removedirs",
    "os.truncate",
    "os.chmod",
    "os.chown",
    "os.utime",
    "tempfile.TemporaryFile",
    "tempfile.NamedTemporaryFile",
    "tempfile.SpooledTemporaryFile",
    "tempfile.mkstemp",
    "tempfile.mkdtemp",
)
IPC_CALLABLES = (
    "os.pipe",
    "os.pipe2",
    "os.openpty",
    "os.mkfifo",
    "os.mknod",
    "os.memfd_create",
    "os.eventfd",
    "os.pidfd_open",
    "pty.openpty",
    "socket.socket",
    "socket.socketpair",
    "socket.fromfd",
    "multiprocessing.Pipe",
    "multiprocessing.Queue",
    "multiprocessing.SimpleQueue",
    "multiprocessing.JoinableQueue",
    "multiprocessing.Manager",
    "multiprocessing.connection.Listener",
    "multiprocessing.shared_memory.SharedMemory",
    "mmap.mmap",
)
AUDIT_EVENTS = (
    "os.fork",
    "os.forkpty",
    "os.posix_spawn",
    "os.exec",
    "subprocess.Popen",
    "socket.__new__",
    "socket.bind",
    "socket.connect",
    "socket.listen",
    "open:read",
    "open:write",
)
RETAINED_REFERENCE_CASES = (
    "builtins.open",
    "io.open",
    "os.open",
    "os.stat",
    "os.lstat",
    "os.access",
    "os.readlink",
    "pathlib.Path.read_bytes",
    "os.write",
    "os.fsync",
    "os.dup",
    "os.pipe",
    "os.remove",
    "os.rename",
    "os.link",
    "os.mkdir",
    "subprocess.Popen",
    "socket.socket",
    "tempfile.mkstemp",
)
NO_AUDIT_PREBIND_CALLABLES = (
    "stat",
    "lstat",
    "access",
    "readlink",
    "write",
    "fsync",
    "dup",
    "pipe",
)
DIR_FD_CASES = (
    ("os.open", "dir_fd"),
    ("os.open", "dir_fd@AT_FDCWD"),
    ("os.stat", "dir_fd"),
    ("os.lstat", "dir_fd"),
    ("os.access", "dir_fd"),
    ("os.readlink", "dir_fd"),
    ("os.mkdir", "dir_fd"),
    ("os.mkfifo", "dir_fd"),
    ("os.mknod", "dir_fd"),
    ("os.link", "src_dir_fd"),
    ("os.link", "dst_dir_fd"),
    ("os.symlink", "dir_fd"),
    ("os.rename", "src_dir_fd"),
    ("os.rename", "dst_dir_fd"),
    ("os.replace", "src_dir_fd"),
    ("os.replace", "dst_dir_fd"),
    ("os.remove", "dir_fd"),
    ("os.unlink", "dir_fd"),
    ("os.rmdir", "dir_fd"),
    ("os.chmod", "dir_fd"),
    ("os.chown", "dir_fd"),
    ("os.utime", "dir_fd"),
)
FORBIDDEN_PATH_CASES = (
    "proc-self-fd",
    "proc-self-fdinfo",
    "proc-self-task-fd",
    "proc-self-task-fdinfo",
    "proc-thread-self-fd",
    "proc-parent-fd",
    "proc-parent-task-fdinfo",
    "dev-fd",
    "lexical-proc-alias",
    "lexical-dev-alias",
    "named-fifo",
)
IMPORT_RECORDER_TERMINAL_SCENARIOS = (
    "preload-root",
    "preload-unapproved",
    "preload-unrelated-package-init",
    "unrecorded-new",
    "duplicate",
    "removed",
    "late",
    "sourceless",
    "retained-source-loader",
    "retained-sourceless-loader",
    "hash-mismatch",
)
IMPORT_RECORDER_VALID_PRELOADS = (
    ("preload-protocol", "protocol.py"),
    ("preload-protocol-package", "pkg/__init__.py"),
)


def _time(index: int) -> str:
    base = datetime(2023, 6, 1, tzinfo=timezone.utc)
    return (base + timedelta(minutes=5 * index)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _bars(count: int = 700) -> list[dict[str, object]]:
    return [
        {
            "time_utc": _time(index),
            "open": "100",
            "high": "100",
            "low": "100",
            "decisions": {},
        }
        for index in range(count)
    ]


def _all_sleeve_bars() -> list[dict[str, object]]:
    bars = _bars()
    bars[0]["decisions"] = {
        "cand_rex_veto_7": {"active": True, "side": -1},
        "fresh_kimchi_fx": {
            "active": True,
            "side": 1,
            "long_gate": True,
            "short_gate": False,
        },
        "frozen_annual_rank7": {
            "active": True,
            "side": 1,
            "source": "funding",
        },
        "markov_transition_long": {"active": True, "side": 1},
        "rex_taker_low_range_position": {"active": True, "side": -1},
    }
    return bars


def _parent_authentication(root: Path) -> dict[str, object]:
    preregistration_path = root / builder.PREREGISTRATION_PATH
    if preregistration_path.exists():
        raw = preregistration_path.read_bytes()
        payload = json.loads(raw)
        preregistration_authentication = {
            "manifest_hash": payload["manifest_hash"],
            "path": builder.PREREGISTRATION_PATH.as_posix(),
            "protocol_implementation_commit": payload[
                "protocol_implementation_commit"
            ],
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    else:
        preregistration_authentication = {
            "manifest_hash": "0" * 64,
            "path": builder.PREREGISTRATION_PATH.as_posix(),
            "protocol_implementation_commit": "0" * 40,
            "sha256": "0" * 64,
        }
    return {
        "environment": {
            "worker_process_environment": (
                builder.prereg.worker_process_environment(root)
            )
        },
        "hashed_inputs": [],
        "preregistration_authentication": preregistration_authentication,
        "runtime_import_closure": [],
    }


def _synthetic_preregistration(root: Path) -> dict[str, object]:
    return builder._with_hash(
        {
            "protocol_version": (
                "gross9_structural_clock_bundle_g9cb3_preregistration_v1"
            ),
            "identity": builder.IDENTITY,
            "protocol_implementation_commit": "1" * 40,
            "git_seal": {"expected_branch": "synthetic/g9cb1b"},
            "candidate_independence": {
                "candidate_identity_present": False,
                "candidate_artifacts_opened": False,
                "comparator_clock_rows_opened": 0,
                "comparator_clocks_preseen_by_research_program": True,
            },
            "bindings": {
                "protocol": [],
                "authority_amendments": (
                    builder._expected_authority_amendment_bindings()
                ),
                "direct_authority": [],
                "config_metadata_evidence": [],
                "runtime_import_roots": [],
                "runtime_import_closure": [],
                "rank7_bundle": {"declared_files": []},
                "source_manifest_ordered_inventory": [],
                "environment": _parent_authentication(root)["environment"],
                "failed_predecessor_preregistrations": (
                    builder.prereg
                    .expected_failed_predecessor_preregistration_bindings()
                ),
                "failed_predecessor_attempts": (
                    builder.prereg.expected_failed_predecessor_attempts()
                ),
            },
            "creation_evidence_boundary": dict(
                builder.prereg.CREATION_EVIDENCE_BOUNDARY
            ),
            "permanent_prohibited_counters": dict(
                builder.prereg.PERMANENT_PROHIBITED_COUNTERS
            ),
            "pre2025_anchor_boundary": {
                "pre2025_anchor_bytes_hashed": True,
                "pre2025_anchor_git_blob_authenticated": True,
                "pre2025_anchor_json_parsed": False,
                "pre2025_anchor_value_rows_opened": 0,
            },
        },
        "manifest_hash",
    )


def _prepare_synthetic_worker_repository(
    root: Path,
) -> tuple[
    dict[str, object],
    dict[str, str],
    dict[str, object],
    dict[str, str],
    Path,
]:
    training = root / "training"
    results = root / "results"
    fixtures = root / "fixtures"
    training.mkdir()
    results.mkdir()
    fixtures.mkdir()
    (training / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy2(builder.__file__, root / builder.BUILDER_PATH)
    shutil.copy2(
        Path(builder.prereg.__file__),
        root / builder.PREREGISTER_PATH,
    )

    preregistration = _synthetic_preregistration(root)
    preregistration_raw = builder._canonical_json_bytes(preregistration)
    (root / builder.PREREGISTRATION_PATH).write_bytes(preregistration_raw)
    os.chmod(root / builder.PREREGISTRATION_PATH, 0o444)
    preregistration_binding = {
        "path": builder.PREREGISTRATION_PATH.as_posix(),
        "sha256": hashlib.sha256(preregistration_raw).hexdigest(),
        "manifest_hash": str(preregistration["manifest_hash"]),
    }
    amendments = builder._expected_authority_amendment_bindings()
    claim = builder._claim_payload(
        "1" * 40,
        preregistration_binding,
        amendments,
        [],
        [],
    )
    claim_raw = builder._canonical_json_bytes(claim)
    (root / builder.CLAIM_PATH).write_bytes(claim_raw)
    claim_binding = {
        "path": builder.CLAIM_PATH.as_posix(),
        "sha256": hashlib.sha256(claim_raw).hexdigest(),
        "claim_hash": str(claim["claim_hash"]),
        "protocol_parent_commit": "1" * 40,
        "claim_commit": "2" * 40,
    }
    synthetic_input = fixtures / "structural-bars.json"
    synthetic_input.write_bytes(
        builder._canonical_json_bytes({"bars": _all_sleeve_bars()})
    )
    return (
        preregistration,
        preregistration_binding,
        claim,
        claim_binding,
        synthetic_input,
    )


def _prepare_guarded_metadata_repository(
    root: Path,
) -> tuple[dict[str, object], dict[str, str], dict[str, object]]:
    results = root / "results"
    results.mkdir()
    preregistration = _synthetic_preregistration(root)
    source_root = builder.prereg.REPOSITORY_ROOT
    attempt = builder.prereg.expected_failed_predecessor_attempts()[0]
    for key in (
        "authority_decision",
        "preregistration",
        "access_claim",
        "attempt_sentinel",
    ):
        relative = Path(attempt[key]["path"])
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, target)
    for binding in (
        builder.prereg.expected_failed_predecessor_preregistration_bindings()
    ):
        relative = Path(binding["path"])
        shutil.copy2(source_root / relative, root / relative)
    closure_paths = (
        Path("execution/gross9_rank7_clock_runtime.py"),
        Path("training/__init__.py"),
        Path("training/gross9_structural_clock_primitives.py"),
    )
    closure = []
    for relative in closure_paths:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, target)
        raw = target.read_bytes()
        closure.append(
            {
                "path": relative.as_posix(),
                "path_type": "regular_file",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "git_blob": builder._git_blob_id(raw),
                "git_mode": "100644",
                "package_initializer": relative.name == "__init__.py",
            }
        )
    shutil.copy2(builder.__file__, root / builder.BUILDER_PATH)
    shutil.copy2(
        builder.prereg.__file__,
        root / builder.PREREGISTER_PATH,
    )
    slot1 = root / attempt["residue"]["slot1_stage"]["path"]
    slot1.mkdir(mode=0o700)
    os.chmod(slot1, 0o700)
    environment = builder._environment_record()
    environment["worker_process_environment"] = (
        builder.prereg.worker_process_environment(root)
    )
    preregistration["bindings"]["environment"] = environment
    preregistration["bindings"]["runtime_import_roots"] = [
        "execution/gross9_rank7_clock_runtime.py",
        "training/gross9_structural_clock_primitives.py",
    ]
    preregistration["bindings"]["runtime_import_closure"] = closure
    for declared in builder._iter_bindings(preregistration):
        path_text = builder._binding_path(declared)
        relative = Path(path_text)
        if relative.is_absolute() or (root / relative).exists():
            continue
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, target)
    preregistration = builder._with_hash(
        preregistration,
        "manifest_hash",
    )
    active = root / builder.PREREGISTRATION_PATH
    active.write_bytes(builder._canonical_json_bytes(preregistration))
    os.chmod(active, 0o444)
    raw = active.read_bytes()
    binding = {
        "path": builder.PREREGISTRATION_PATH.as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_hash": str(preregistration["manifest_hash"]),
    }
    raw_cache: dict[str, tuple[bytes, os.stat_result]] = {}
    hashed_inputs = builder._validate_regular_hashed_inputs(
        root,
        preregistration,
        verify_git=False,
        raw_cache=raw_cache,
    )
    closures = builder._validate_static_closures(
        root,
        preregistration,
        verify_git=False,
        raw_cache=raw_cache,
    )
    parent_authentication = {
        "environment": environment,
        "hashed_inputs": hashed_inputs,
        "preregistration_authentication": {
            "manifest_hash": binding["manifest_hash"],
            "path": binding["path"],
            "protocol_implementation_commit": preregistration[
                "protocol_implementation_commit"
            ],
            "sha256": binding["sha256"],
        },
        "runtime_import_closure": closures["runtime"],
    }
    return preregistration, binding, parent_authentication


def _capability_row(slot: int = 1, *, parent_pid: int = 1234) -> dict[str, Any]:
    return {
        "slot": slot,
        "parent_pid": parent_pid,
        "stage_directory": (
            f"results/.gross9-structural-clock-g9cb3-worker-slot{slot}"
        ),
        "carrier_kind": "anonymous_pipe_v1",
        "carrier_device": slot,
        "carrier_inode": slot + 100,
        "token_sha256": hashlib.sha256(bytes([slot]) * 32).hexdigest(),
        "consumed_ledger_path": (
            builder.WORKER_LEDGER_PATHS[slot - 1].as_posix()
        ),
    }


def _guard(root: Path) -> builder._WorkerIsolationGuard:
    (root / "results").mkdir(exist_ok=True)
    own = "results/.gross9-structural-clock-g9cb3-worker-own"
    other = "results/.gross9-structural-clock-g9cb3-worker-other"
    (root / own).mkdir(exist_ok=True)
    return builder._WorkerIsolationGuard(
        root=root,
        own_stage=own,
        other_stage=other,
        ledger_paths=builder.WORKER_LEDGER_PATHS,
    )


def _literal_string_tuples(function: Any) -> set[tuple[str, ...]]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return {
        tuple(element.value for element in node.elts)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Tuple, ast.List))
        and node.elts
        and all(
            isinstance(element, ast.Constant)
            and isinstance(element.value, str)
            for element in node.elts
        )
    }


def _no_audit_prebind_violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    os_aliases = {"os"}
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "os":
                    os_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "os":
            for alias in node.names:
                if alias.name in NO_AUDIT_PREBIND_CALLABLES:
                    violations.append(
                        f"{path.name}:{node.lineno}:from-os:{alias.name}"
                    )
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in os_aliases
            and node.attr in NO_AUDIT_PREBIND_CALLABLES
        ):
            violations.append(
                f"{path.name}:{node.lineno}:attribute:{node.attr}"
            )
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in os_aliases
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in NO_AUDIT_PREBIND_CALLABLES
        ):
            violations.append(
                f"{path.name}:{node.lineno}:getattr:{node.args[1].value}"
            )
    return violations


ISOLATION_CALL_HARNESS = r"""
from __future__ import annotations

import builtins
import concurrent.futures
import concurrent.futures.process
import fcntl
import io
import json
import mmap
import multiprocessing
import multiprocessing.connection
import multiprocessing.process
import multiprocessing.shared_memory
import os
from pathlib import Path
import pty
import socket
import subprocess
import sys
import tempfile
import threading

from training import build_gross9_structural_clock_bundle as builder


case_root = Path(sys.argv[1]).resolve()
family = sys.argv[2]
case = sys.argv[3]
case_root.mkdir(parents=True)
results = case_root / "results"
results.mkdir()
own_stage = results / ".gross9-structural-clock-g9cb3-worker-own"
other_stage = results / ".gross9-structural-clock-g9cb3-worker-other"
own_stage.mkdir()
probe = other_stage / "probe"
scratch = case_root / "scratch"
scratch.mkdir()
descriptor = os.open(os.devnull, os.O_RDWR)
directory_descriptor = os.open(own_stage, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
mapping_path = scratch / "shared-mapping.bin"
mapping_path.write_bytes(b"\0" * 4096)
mapping_descriptor = os.open(mapping_path, os.O_RDWR)
retained_open = builtins.open
(scratch / "retained-remove").write_bytes(b"remove")
(scratch / "retained-rename-source").write_bytes(b"rename")
(scratch / "retained-link-source").write_bytes(b"link")


def emit(status, **extra):
    print(json.dumps({"status": status, "case": case, **extra}, sort_keys=True))


def resolve(name):
    roots = {
        "builtins": builtins,
        "concurrent": concurrent,
        "fcntl": fcntl,
        "io": io,
        "mmap": mmap,
        "multiprocessing": multiprocessing,
        "os": os,
        "pathlib": sys.modules["pathlib"],
        "pty": pty,
        "socket": socket,
        "subprocess": subprocess,
        "tempfile": tempfile,
    }
    parts = name.split(".")
    current = roots.get(parts[0])
    if current is None:
        return None
    for part in parts[1:]:
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def clean_result(value):
    if isinstance(value, tuple):
        for item in value:
            if isinstance(item, int):
                try:
                    os.close(item)
                except OSError:
                    pass
            elif hasattr(item, "close"):
                item.close()
    elif isinstance(value, int):
        try:
            os.close(value)
        except OSError:
            pass
    elif hasattr(value, "terminate") and hasattr(value, "wait"):
        value.terminate()
        value.wait()
    elif hasattr(value, "shutdown"):
        value.shutdown(wait=True)
    elif hasattr(value, "close"):
        value.close()


def invoke_process_descriptor(name):
    if name.startswith("fcntl.fcntl:"):
        constant_name = name.split(":", 1)[1]
        command = getattr(fcntl, constant_name, None)
        if command is None:
            emit("absent", exact_name=f"fcntl.{constant_name}")
            return
        fcntl.fcntl(descriptor, command, 0)
        return
    target = resolve(name)
    if target is None:
        emit("absent", exact_name=name)
        return
    if name == "multiprocessing.process.BaseProcess.start":
        multiprocessing.Process(target=lambda: None).start()
    else:
        clean_result(target())


def invoke_path(name):
    if name == "builtins.open":
        builtins.open(probe, "rb")
    elif name == "io.open":
        io.open(probe, "rb")
    elif name == "os.open":
        os.open(probe, os.O_RDONLY)
    elif name == "os.access":
        os.access(probe, os.F_OK)
    elif name.startswith("os.path."):
        target = resolve(name)
        if target is None:
            emit("absent", exact_name=name)
            return
        target(probe)
    elif name.startswith("os."):
        target = resolve(name)
        if target is None:
            emit("absent", exact_name=name)
            return
        value = target(probe)
        if name == "os.walk":
            list(value)
        elif hasattr(value, "close"):
            value.close()
    elif name.startswith("pathlib.Path."):
        method = name.rsplit(".", 1)[1]
        target = getattr(probe, method, None)
        if target is None:
            emit("absent", exact_name=name)
            return
        if method == "open":
            target("rb")
        elif method == "write_text":
            target("x", encoding="utf-8")
        elif method == "write_bytes":
            target(b"x")
        elif method in {"glob", "rglob"}:
            list(target("*"))
        elif method == "chmod":
            target(0o600)
        elif method in {"rename", "replace"}:
            target(own_stage / "target")
        elif method in {"symlink_to", "hardlink_to"}:
            target(own_stage / "target")
        else:
            value = target()
            if hasattr(value, "close"):
                value.close()
            elif method == "iterdir":
                list(value)
    else:
        emit("absent", exact_name=name)


def invoke_mutation(name):
    if name.startswith("builtins.open:") or name.startswith("io.open:"):
        owner, mode = name.split(":", 1)
        actual_mode = "r+" if mode == "+" else mode
        (builtins.open if owner == "builtins.open" else io.open)(
            scratch / "open-target",
            actual_mode,
        )
        return
    if name.startswith("os.open:"):
        flag_name = name.split(":", 1)[1]
        flag = getattr(os, flag_name, None)
        if flag is None:
            emit("absent", exact_name=f"os.{flag_name}")
            return
        target = scratch if flag_name == "O_TMPFILE" else scratch / "os-open-target"
        os.open(target, flag, 0o600)
        return
    target = resolve(name)
    if target is None:
        emit("absent", exact_name=name)
        return
    descriptor_arguments = {
        "os.write": (descriptor, b"x"),
        "os.pwrite": (descriptor, b"x", 0),
        "os.writev": (descriptor, [b"x"]),
        "os.pwritev": (descriptor, [b"x"], 0),
        "os.copy_file_range": (descriptor, descriptor, 1),
        "os.sendfile": (descriptor, descriptor, 0, 1),
        "os.splice": (descriptor, descriptor, 1),
        "os.ftruncate": (descriptor, 0),
        "os.posix_fallocate": (descriptor, 0, 1),
        "os.fchmod": (descriptor, 0o600),
        "os.fchown": (descriptor, os.getuid(), os.getgid()),
        "os.fsync": (descriptor,),
        "os.fdatasync": (descriptor,),
        "os.setxattr": (descriptor, b"user.g9cb", b"x"),
        "os.removexattr": (descriptor, b"user.g9cb"),
    }
    if name in descriptor_arguments:
        target(*descriptor_arguments[name])
        return
    if name.startswith("tempfile."):
        clean_result(target())
        return
    one_path_arguments = {
        "os.mkdir": (scratch / "mkdir",),
        "os.makedirs": (scratch / "makedirs",),
        "os.mkfifo": (scratch / "fifo",),
        "os.mknod": (scratch / "node",),
        "os.remove": (scratch / "remove",),
        "os.unlink": (scratch / "unlink",),
        "os.rmdir": (scratch / "rmdir",),
        "os.removedirs": (scratch / "removedirs",),
        "os.truncate": (scratch / "truncate", 0),
        "os.chmod": (scratch / "chmod", 0o600),
        "os.chown": (scratch / "chown", os.getuid(), os.getgid()),
        "os.utime": (scratch / "utime", None),
    }
    if name in one_path_arguments:
        target(*one_path_arguments[name])
        return
    target(scratch / "source", scratch / "destination")


def invoke_ipc(name):
    target = resolve(name)
    if target is None:
        emit("absent", exact_name=name)
        return
    arguments = {
        "os.pipe": (),
        "os.pipe2": (os.O_CLOEXEC,),
        "os.openpty": (),
        "os.mkfifo": (scratch / "ipc-fifo",),
        "os.mknod": (scratch / "ipc-node",),
        "os.memfd_create": ("g9cb", 0),
        "os.eventfd": (0, 0),
        "os.pidfd_open": (os.getpid(), 0),
        "pty.openpty": (),
        "socket.socket": (),
        "socket.socketpair": (),
        "socket.fromfd": (descriptor, socket.AF_INET, socket.SOCK_STREAM),
        "multiprocessing.Pipe": (),
        "multiprocessing.Queue": (),
        "multiprocessing.SimpleQueue": (),
        "multiprocessing.JoinableQueue": (),
        "multiprocessing.Manager": (),
        "multiprocessing.connection.Listener": (),
        "multiprocessing.shared_memory.SharedMemory": (),
        "mmap.mmap": (
            mapping_descriptor,
            4096,
        ),
    }
    if name == "mmap.mmap":
        clean_result(
            target(
                *arguments[name],
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        )
    elif name == "multiprocessing.shared_memory.SharedMemory":
        clean_result(target(create=True, size=1))
    else:
        clean_result(target(*arguments[name]))


def invoke_audit(name):
    if name == "open:read":
        sys.audit("open", os.fspath(probe), "r", 0)
    elif name == "open:write":
        sys.audit("open", os.fspath(scratch / "audit-write"), "w", os.O_WRONLY)
    else:
        sys.audit(name)


def invoke_retained(name, retained):
    if retained is None:
        emit("absent", exact_name=name)
        return
    if name in {"builtins.open", "io.open"}:
        retained(probe, "rb")
    elif name == "os.open":
        retained(probe, os.O_RDONLY)
    elif name in {"os.stat", "os.lstat", "os.readlink"}:
        retained(probe)
    elif name == "os.access":
        retained(probe, os.F_OK)
    elif name == "pathlib.Path.read_bytes":
        retained(probe)
    elif name == "os.remove":
        retained(scratch / "retained-remove")
    elif name == "os.rename":
        retained(
            scratch / "retained-rename-source",
            scratch / "retained-rename-destination",
        )
    elif name == "os.link":
        retained(
            scratch / "retained-link-source",
            scratch / "retained-link-destination",
        )
    elif name == "os.mkdir":
        retained(scratch / "retained-mkdir")
    elif name == "subprocess.Popen":
        clean_result(retained([sys.executable, "-B", "-c", "pass"]))
    elif name == "socket.socket":
        clean_result(retained())
    elif name == "tempfile.mkstemp":
        clean_result(retained())


def invoke_dir_fd(name, keyword):
    target = resolve(name)
    if target is None:
        emit("absent", exact_name=name)
        return
    if keyword.endswith("@AT_FDCWD"):
        keyword = keyword.split("@", 1)[0]
        kwargs = {keyword: getattr(os, "AT_FDCWD", -100)}
    else:
        kwargs = {keyword: directory_descriptor}
    if name == "os.open":
        target("redirected", os.O_RDONLY, **kwargs)
    elif name == "os.access":
        target("redirected", os.F_OK, **kwargs)
    elif name in {"os.mkdir", "os.mkfifo", "os.mknod"}:
        target("redirected", **kwargs)
    elif name in {"os.link", "os.rename", "os.replace"}:
        target("source", "destination", **kwargs)
    elif name == "os.symlink":
        target("source", "destination", **kwargs)
    elif name == "os.chmod":
        target("redirected", 0o600, **kwargs)
    elif name == "os.chown":
        target("redirected", os.getuid(), os.getgid(), **kwargs)
    elif name == "os.utime":
        target("redirected", None, **kwargs)
    else:
        target("redirected", **kwargs)


def invoke_cross_ledger(name):
    slot_text, operation = name.split(":", 1)
    slot = int(slot_text)
    guard.bind_ledger_slot(slot)
    other_ledger = case_root / builder.WORKER_LEDGER_PATHS[1 - (slot - 1)]
    if operation == "read":
        builtins.open(other_ledger, "rb")
    elif operation == "stat":
        os.stat(other_ledger)
    elif operation == "list":
        os.listdir(other_ledger)
    elif operation == "write":
        builtins.open(other_ledger, "wb")


def invoke_pycache(name):
    location, operation = name.split(":", 1)
    if location == "repository":
        target = case_root / "pkg" / "__pycache__" / "evil.pyc"
    else:
        target = case_root / "results" / ".g9cb3-bytecode-cache-disabled" / "evil.pyc"
    guard.allowed_mutations.add(target.as_posix())
    if operation == "read":
        builtins.open(target, "rb")
    elif operation == "write":
        builtins.open(target, "wb")
    elif operation == "retained-read":
        retained_open(target, "rb")
    else:
        retained_open(target, "wb")


def invoke_pycache_during_source_load(name):
    timing, operation = name.split(":", 1)
    source = scratch / "legitimate_runtime.py"
    target = (
        case_root
        / "results"
        / ".g9cb3-bytecode-cache-disabled"
        / "injected.pyc"
    )

    def inject():
        retained_open(target, "rb" if operation == "read" else "wb")

    class InjectingLoader(importlib.machinery.SourceFileLoader):
        def get_data(self, path):
            if Path(path) == source:
                if timing == "reentrant":
                    inject()
                else:
                    errors = []

                    def concurrent_inject():
                        try:
                            inject()
                        except BaseException as exc:
                            errors.append(exc)

                    thread = threading.Thread(target=concurrent_inject)
                    thread.start()
                    thread.join()
                    if errors:
                        raise errors[0]
            return super().get_data(path)

    loader = InjectingLoader("legitimate_runtime", str(source))
    spec = importlib.util.spec_from_loader(
        "legitimate_runtime",
        loader,
    )
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)


def invoke_forbidden_path(name):
    fifo = scratch / "named.fifo"
    paths = {
        "proc-self-fd": f"/proc/self/fd/{descriptor}",
        "proc-self-fdinfo": f"/proc/self/fdinfo/{descriptor}",
        "proc-self-task-fd": (
            f"/proc/self/task/{os.getpid()}/fd/{descriptor}"
        ),
        "proc-self-task-fdinfo": (
            f"/proc/self/task/{os.getpid()}/fdinfo/{descriptor}"
        ),
        "proc-thread-self-fd": f"/proc/thread-self/fd/{descriptor}",
        "proc-parent-fd": f"/proc/{os.getppid()}/fd/{descriptor}",
        "proc-parent-task-fdinfo": (
            f"/proc/{os.getppid()}/task/{os.getppid()}/fdinfo/{descriptor}"
        ),
        "dev-fd": f"/dev/fd/{descriptor}",
        "lexical-proc-alias": (
            f"/proc/self/task/../fd/{descriptor}"
        ),
        "lexical-dev-alias": f"/dev/./fd/{descriptor}",
        "named-fifo": fifo,
    }
    builtins.open(paths[name], "rb")


retained = None
if family == "retained":
    retained = resolve(case)
if family == "cross-ledger" and case.startswith("2:"):
    pass_one = case_root / builder.WORKER_LEDGER_PATHS[0]
    pass_one.write_bytes(b"immutable-ledger\n")
if family == "pycache":
    location = case.split(":", 1)[0]
    pycache = (
        case_root / "pkg" / "__pycache__"
        if location == "repository"
        else case_root / "results" / ".g9cb3-bytecode-cache-disabled"
    )
    pycache.mkdir(parents=True)
    (pycache / "evil.pyc").write_bytes(b"malicious-bytecode")
if family == "pycache-source-load":
    import importlib.machinery
    import importlib.util

    source = scratch / "legitimate_runtime.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    pycache = (
        case_root
        / "results"
        / ".g9cb3-bytecode-cache-disabled"
    )
    pycache.mkdir(parents=True)
    (pycache / "injected.pyc").write_bytes(b"malicious-bytecode")
if family == "forbidden-path" and case.split(":", 1)[1] == "named-fifo":
    os.mkfifo(scratch / "named.fifo")

guard = builder._WorkerIsolationGuard(
    root=case_root,
    own_stage=own_stage.relative_to(case_root).as_posix(),
    other_stage=other_stage.relative_to(case_root).as_posix(),
    ledger_paths=builder.WORKER_LEDGER_PATHS,
)

try:
    guard.install()
    if family == "process":
        invoke_process_descriptor(case)
    elif family == "path":
        invoke_path(case)
    elif family == "mutation":
        invoke_mutation(case)
    elif family == "ipc":
        invoke_ipc(case)
    elif family == "audit":
        invoke_audit(case)
    elif family == "retained":
        invoke_retained(case, retained)
    elif family == "dir-fd":
        callable_name, keyword = case.split(":", 1)
        invoke_dir_fd(callable_name, keyword)
    elif family == "cross-ledger":
        invoke_cross_ledger(case)
    elif family == "pycache":
        invoke_pycache(case)
    elif family == "pycache-source-load":
        invoke_pycache_during_source_load(case)
    elif family == "forbidden-path":
        _phase, forbidden_name = case.split(":", 1)
        invoke_forbidden_path(forbidden_name)
    else:
        raise AssertionError(f"unknown family: {family}")
except builder.TerminalG9CB3Failure as exc:
    emit("terminal", error_type=type(exc).__name__, message=str(exc), counters=guard.counters())
except BaseException as exc:
    emit("unexpected", error_type=type(exc).__name__, message=str(exc), counters=guard.counters())
else:
    emit("survived", counters=guard.counters())
finally:
    try:
        os.close(descriptor)
    except OSError:
        pass
    try:
        os.close(directory_descriptor)
    except OSError:
        pass
    try:
        os.close(mapping_descriptor)
    except OSError:
        pass
"""


def _run_isolation_call(
    tmp_path: Path,
    family: str,
    case: str,
) -> dict[str, Any]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(Path(builder.__file__).resolve().parents[1]),
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            ISOLATION_CALL_HARNESS,
            str(tmp_path / "guard-root"),
            family,
            case,
        ],
        cwd=Path(builder.__file__).resolve().parents[1],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line]
    assert lines, completed.stderr
    return json.loads(lines[-1])


def _assert_terminal_or_exact_absence(
    result: dict[str, Any],
    *,
    absent_names: set[str],
) -> None:
    if result["status"] == "absent":
        assert result["exact_name"] in absent_names
        return
    assert result["status"] == "terminal", result
    assert result["error_type"] == "TerminalG9CB3Failure"


IMPORT_RECORDER_HARNESS = r"""
from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import py_compile
import sys
import types

from training import build_gross9_structural_clock_bundle as builder


root = Path(sys.argv[1]).resolve()
scenario = sys.argv[2]
(root / "execution").mkdir(parents=True)
(root / "training").mkdir()
(root / "pkg").mkdir()
sources = {
    "execution/gross9_rank7_clock_runtime.py": "VALUE = 'rank7'\n",
    "training/gross9_structural_clock_primitives.py": "VALUE = 'primitives'\n",
    "training/late_runtime.py": "VALUE = 'late'\n",
    "training/rogue.py": "VALUE = 'rogue'\n",
    "protocol.py": "VALUE = 'protocol'\n",
    "pkg/protocol.py": "import pkg\nVALUE = 'package-protocol'\n",
    "pkg/__init__.py": "VALUE = 'package'\n",
}
for relative, text in sources.items():
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def digest(relative):
    return hashlib.sha256((root / relative).read_bytes()).hexdigest()


def load_source(name, relative):
    path = root / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"no source loader for {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def emit(status, **extra):
    print(json.dumps({"status": status, "scenario": scenario, **extra}, sort_keys=True))


protocol_rows = []
closure_paths = [
    "execution/gross9_rank7_clock_runtime.py",
    "training/gross9_structural_clock_primitives.py",
]
if scenario == "late":
    closure_paths.append("training/late_runtime.py")
if scenario == "hash-mismatch":
    closure = [
        {
            "path": relative,
            "sha256": ("0" * 64 if index == 0 else digest(relative)),
        }
        for index, relative in enumerate(closure_paths)
    ]
else:
    closure = [
        {"path": relative, "sha256": digest(relative)}
        for relative in closure_paths
    ]

try:
    retained_source_exec = importlib.machinery.SourceFileLoader.exec_module
    retained_sourceless_exec = (
        importlib.machinery.SourcelessFileLoader.exec_module
    )
    if scenario == "preload-root":
        load_source("preloaded_rank7", closure_paths[0])
    elif scenario == "preload-unapproved":
        load_source("preloaded_rogue", "training/rogue.py")
    elif scenario in {"preload-protocol", "preload-protocol-package"}:
        protocol_path = (
            "pkg/protocol.py"
            if scenario == "preload-protocol-package"
            else "protocol.py"
        )
        protocol_rows = [{"path": protocol_path}]
        sys.path.insert(0, str(root))
        load_source("preloaded_protocol", protocol_path)
    elif scenario == "preload-unrelated-package-init":
        load_source("preloaded_package", "pkg/__init__.py")

    preregistration = {"bindings": {"protocol": protocol_rows}}
    recorder = builder._RuntimeImportRecorder(
        root=root,
        preregistration=preregistration,
        runtime_closure=closure,
    )
    recorder.install()

    if scenario in {"preload-protocol", "preload-protocol-package"}:
        emit(
            "passed",
            preloaded=sorted(recorder.preloaded_repository_paths),
        )
    elif scenario == "preload-unrelated-package-init":
        emit(
            "survived",
            preloaded=sorted(recorder.preloaded_repository_paths),
        )
    else:
        if scenario == "sourceless":
            pyc = root / "training" / "rogue.pyc"
            py_compile.compile(
                str(root / "training/rogue.py"),
                cfile=str(pyc),
                doraise=True,
            )
            loader = importlib.machinery.SourcelessFileLoader(
                "sourceless_rogue",
                str(pyc),
            )
            spec = importlib.util.spec_from_loader(
                "sourceless_rogue",
                loader,
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules["sourceless_rogue"] = module
            loader.exec_module(module)
        elif scenario == "retained-source-loader":
            path = root / closure_paths[0]
            loader = importlib.machinery.SourceFileLoader(
                "retained_source",
                str(path),
            )
            spec = importlib.util.spec_from_loader(
                "retained_source",
                loader,
            )
            module = importlib.util.module_from_spec(spec)
            retained_source_exec(loader, module)
            recorder.reconcile()
            emit("survived")
        elif scenario == "retained-sourceless-loader":
            pyc = root / "training" / "retained-rogue.pyc"
            py_compile.compile(
                str(root / "training/rogue.py"),
                cfile=str(pyc),
                doraise=True,
            )
            loader = importlib.machinery.SourcelessFileLoader(
                "retained_sourceless",
                str(pyc),
            )
            spec = importlib.util.spec_from_loader(
                "retained_sourceless",
                loader,
            )
            module = importlib.util.module_from_spec(spec)
            retained_sourceless_exec(loader, module)
            recorder.reconcile()
            emit("survived")
        elif scenario == "unrecorded-new":
            module = types.ModuleType("unrecorded_rogue")
            module.__file__ = str(root / "training/rogue.py")
            sys.modules[module.__name__] = module
            recorder.reconcile()
        else:
            first = load_source("recorded_rank7", closure_paths[0])
            if scenario == "hash-mismatch":
                raise AssertionError("hash mismatch execution survived")
            if scenario == "duplicate":
                load_source("recorded_rank7_duplicate", closure_paths[0])
            elif scenario == "removed":
                del sys.modules[first.__name__]
                recorder.reconcile()
            else:
                load_source("recorded_primitives", closure_paths[1])
                if scenario == "new-and-freeze":
                    reconciled = recorder.reconcile()
                    frozen = recorder.freeze()
                    emit("passed", reconciled=reconciled, frozen=frozen)
                elif scenario == "late":
                    recorder.freeze()
                    load_source("recorded_late", "training/late_runtime.py")
                else:
                    raise AssertionError(f"unknown scenario: {scenario}")
except builder.TerminalG9CB3Failure as exc:
    emit("terminal", error_type=type(exc).__name__, message=str(exc))
except BaseException as exc:
    emit("unexpected", error_type=type(exc).__name__, message=str(exc))
"""


def _run_import_recorder_case(
    tmp_path: Path,
    scenario: str,
) -> dict[str, Any]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(Path(builder.__file__).resolve().parents[1]),
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            IMPORT_RECORDER_HARNESS,
            str(tmp_path / "import-root"),
            scenario,
        ],
        cwd=Path(builder.__file__).resolve().parents[1],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line]
    assert lines, completed.stderr
    return json.loads(lines[-1])


PARENT_DEATH_WORKER_HARNESS = r"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

from training import build_gross9_structural_clock_bundle as builder


phase = sys.argv[1]
marker_directory = Path(sys.argv[2])
expected_parent_pid = int(sys.argv[3])
descriptor = int(sys.argv[4])
builder._establish_parent_death_contract(expected_parent_pid)
observed = {"phase": phase, "worker_pid": os.getpid(), "bytes_read": 0}
if phase == "after-handoff-before-read":
    pass
elif phase == "partial-read":
    observed["bytes_read"] = len(os.read(descriptor, 7))
elif phase == "complete-consumption":
    token = bytearray()
    while len(token) < 32:
        chunk = os.read(descriptor, 32 - len(token))
        if not chunk:
            raise AssertionError("carrier reached EOF before complete consumption")
        token.extend(chunk)
    if os.read(descriptor, 1) != b"":
        raise AssertionError("carrier contains extra bytes")
    observed["bytes_read"] = len(token)
    os.close(descriptor)
(marker_directory / "phase.json").write_text(
    json.dumps(observed, sort_keys=True),
    encoding="utf-8",
)
time.sleep(30)
"""


PARENT_DEATH_PARENT_HARNESS = r"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


phase = sys.argv[1]
marker_directory = Path(sys.argv[2])
repository_root = sys.argv[3]
worker_harness = sys.argv[4]
read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
os.write(write_fd, b"x" * 32)
os.close(write_fd)
environment = dict(os.environ)
environment.update(
    {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": repository_root,
    }
)
worker = subprocess.Popen(
    [
        sys.executable,
        "-B",
        "-c",
        worker_harness,
        phase,
        str(marker_directory),
        str(os.getpid()),
        str(read_fd),
    ],
    cwd=repository_root,
    env=environment,
    close_fds=True,
    pass_fds=(read_fd,),
)
(marker_directory / "worker-pid").write_text(
    str(worker.pid),
    encoding="ascii",
)
os.close(read_fd)
deadline = time.monotonic() + 10
while not (marker_directory / "phase.json").exists():
    if worker.poll() is not None:
        raise SystemExit(f"worker exited before phase marker: {worker.returncode}")
    if time.monotonic() >= deadline:
        worker.terminate()
        worker.wait()
        raise SystemExit("worker phase marker timed out")
    time.sleep(0.01)
os._exit(0)
"""


def _process_is_dead_or_zombie(pid: int) -> bool:
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        fields = stat_path.read_text(encoding="ascii").split()
    except FileNotFoundError:
        return True
    return len(fields) > 2 and fields[2] == "Z"


def _run_parent_death_phase(
    tmp_path: Path,
    phase: str,
) -> tuple[dict[str, Any], int]:
    marker_directory = tmp_path / phase
    marker_directory.mkdir()
    repository_root = Path(builder.__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            PARENT_DEATH_PARENT_HARNESS,
            phase,
            str(marker_directory),
            str(repository_root),
            PARENT_DEATH_WORKER_HARNESS,
        ],
        cwd=repository_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    marker = json.loads(
        (marker_directory / "phase.json").read_text(encoding="utf-8")
    )
    worker_pid = int(
        (marker_directory / "worker-pid").read_text(encoding="ascii")
    )
    deadline = time.monotonic() + 5
    while not _process_is_dead_or_zombie(worker_pid):
        if time.monotonic() >= deadline:
            os.kill(worker_pid, signal.SIGKILL)
            raise AssertionError(
                f"worker {worker_pid} survived parent death in {phase}"
            )
        time.sleep(0.01)
    return marker, worker_pid


def test_top_level_is_stdlib_only_and_uses_only_isolated_runtime_roots() -> None:
    source = Path(builder.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not {
        name
        for name in imported
        if name.split(".")[0]
        not in getattr(sys, "stdlib_module_names")
        and name.split(".")[0] != "training"
        and name != "__future__"
    }
    assert builder.RUNTIME_IMPORT_MODULES == (
        "execution.gross9_rank7_clock_runtime",
        "training.gross9_structural_clock_primitives",
    )
    assert "ethereum_settlement_demand_impulse" not in source.lower()
    assert "_reconstruct_gross9_runtime_clocks" not in source
    assert "--worker-token" not in source
    assert "G9CB_WORKER_TOKEN" not in source


def test_three_authority_amendments_have_exact_order_and_schema() -> None:
    amendments = builder._expected_authority_amendment_bindings()
    assert [row["identity"] for row in amendments] == [
        "G9CB-1A",
        "G9CB-1B",
        "G9CB-1C",
    ]
    assert all(
        tuple(row)
        == (
            "identity",
            "path",
            "path_type",
            "sha256",
            "git_blob",
            "git_mode",
            "authority_commit",
        )
        for row in amendments
    )
    assert all(row["path_type"] == "regular_file" for row in amendments)
    assert all(row["git_mode"] == "100644" for row in amendments)
    assert not hasattr(builder, "_expected_authority_amendment_binding")


def test_builder_zero_access_accepts_exact_integer_prohibited_counters(
    tmp_path: Path,
) -> None:
    preregistration = _synthetic_preregistration(tmp_path)
    builder._validate_zero_access(preregistration)
    assert preregistration["permanent_prohibited_counters"][
        "cagr_values_computed"
    ] == 0


@pytest.mark.parametrize("invalid", [False, 1, -1, 0.0, "0", None, []])
def test_builder_zero_access_rejects_noninteger_zero_counter(
    tmp_path: Path,
    invalid: object,
) -> None:
    preregistration = _synthetic_preregistration(tmp_path)
    preregistration["permanent_prohibited_counters"][
        "cagr_values_computed"
    ] = invalid
    with pytest.raises(
        builder.TerminalG9CB3Failure,
        match="zero-access schema differs",
    ):
        builder._validate_zero_access(preregistration)


def test_malformed_preregistration_stops_before_all_downstream_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "results").mkdir()
    preregistration = _synthetic_preregistration(tmp_path)
    preregistration["permanent_prohibited_counters"][
        "cagr_values_computed"
    ] = False
    preregistration = builder._with_hash(
        preregistration,
        "manifest_hash",
    )
    (tmp_path / builder.PREREGISTRATION_PATH).write_bytes(
        builder._canonical_json_bytes(preregistration)
    )
    called: list[str] = []

    def forbidden(name: str) -> object:
        def invoke(*_args: object, **_kwargs: object) -> object:
            called.append(name)
            raise AssertionError(f"{name} must not run")

        return invoke

    monkeypatch.setattr(
        builder,
        "_validate_regular_hashed_inputs",
        forbidden("hashed-input"),
    )
    monkeypatch.setattr(
        builder,
        "_validate_environment",
        forbidden("environment"),
    )
    monkeypatch.setattr(
        builder,
        "_validate_static_closures",
        forbidden("closure"),
    )
    monkeypatch.setattr(
        builder,
        "_require_clean_pushed_branch",
        forbidden("git-seal"),
    )
    monkeypatch.setattr(
        builder.prereg,
        "validate_failed_predecessor_preregistrations",
        forbidden("historical"),
    )
    monkeypatch.setattr(
        builder.prereg,
        "validate_protocol_commit_topology",
        forbidden("topology"),
    )
    with pytest.raises(
        builder.TerminalG9CB3Failure,
        match="zero-access schema differs",
    ):
        builder.validate_claim_preflight(tmp_path)
    assert called == []
    for path in (
        builder.CLAIM_PATH,
        builder.SENTINEL_PATH,
        *builder.WORKER_LEDGER_PATHS,
        builder.CSV_PATH,
        builder.MANIFEST_PATH,
    ):
        assert not (tmp_path / path).exists()


def test_historical_v1_bytes_are_rejected_by_operative_version_before_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "results").mkdir()
    historical = (
        builder.prereg.REPOSITORY_ROOT
        / builder.prereg.HISTORICAL_PREREGISTRATION_PATH
    ).read_bytes()
    (tmp_path / builder.PREREGISTRATION_PATH).write_bytes(historical)
    monkeypatch.setattr(
        builder,
        "_validate_zero_access",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("zero schema must not classify historical v1")
        ),
    )
    with pytest.raises(
        builder.TerminalG9CB3Failure,
        match="operative preregistration protocol version",
    ):
        builder.validate_preregistration(
            tmp_path,
            validation_mode="synthetic",
        )


def test_preregistration_seal_head_rejects_intervening_commit(
    tmp_path: Path,
) -> None:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=tmp_path,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()

    git("init")
    git("config", "user.email", "g9cb-test@example.invalid")
    git("config", "user.name", "G9CB Test")
    (tmp_path / "protocol.txt").write_text("Q\n", encoding="utf-8")
    git("add", "protocol.txt")
    git("commit", "-m", "Q")
    implementation = git("rev-parse", "HEAD")

    artifact = tmp_path / builder.PREREGISTRATION_PATH
    artifact.parent.mkdir()
    artifact.write_text("{}\n", encoding="utf-8")
    git("add", "-f", builder.PREREGISTRATION_PATH.as_posix())
    git("commit", "-m", "P")
    seal = git("rev-parse", "HEAD")
    preregistration = {
        "protocol_implementation_commit": implementation,
    }
    builder._validate_preregistration_seal_head(
        tmp_path,
        preregistration,
        seal,
    )

    (tmp_path / "intervening.txt").write_text("X\n", encoding="utf-8")
    git("add", "intervening.txt")
    git("commit", "-m", "X")
    intervening = git("rev-parse", "HEAD")
    with pytest.raises(
        builder.TerminalG9CB3Failure,
        match="direct preregistration-seal child",
    ):
        builder._validate_preregistration_seal_head(
            tmp_path,
            preregistration,
            intervening,
        )


def test_committed_publication_topology_requires_exact_q_p_c_d(
    tmp_path: Path,
) -> None:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=tmp_path,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()

    git("init")
    git("config", "user.email", "g9cb-test@example.invalid")
    git("config", "user.name", "G9CB Test")
    (tmp_path / "protocol.txt").write_text("Q\n", encoding="utf-8")
    git("add", "protocol.txt")
    git("commit", "-m", "Q")
    implementation = git("rev-parse", "HEAD")

    preregistration_path = tmp_path / builder.PREREGISTRATION_PATH
    preregistration_path.parent.mkdir()
    preregistration_path.write_text("{}\n", encoding="utf-8")
    git("add", "-f", builder.PREREGISTRATION_PATH.as_posix())
    git("commit", "-m", "P")
    preregistration_seal = git("rev-parse", "HEAD")

    claim_path = tmp_path / builder.CLAIM_PATH
    claim_path.write_text("{}\n", encoding="utf-8")
    git("add", "-f", builder.CLAIM_PATH.as_posix())
    git("commit", "-m", "C")
    claim = git("rev-parse", "HEAD")

    for path in (
        builder.SENTINEL_PATH,
        *builder.WORKER_LEDGER_PATHS,
        builder.CSV_PATH,
        builder.MANIFEST_PATH,
    ):
        candidate = tmp_path / path
        candidate.write_text("{}\n", encoding="utf-8")
        git("add", "-f", path.as_posix())
    git("commit", "-m", "D")
    publication = git("rev-parse", "HEAD")

    assert builder._validate_committed_publication_topology(
        tmp_path,
        {"protocol_implementation_commit": implementation},
        publication,
    ) == {
        "protocol_implementation_commit": implementation,
        "preregistration_seal_commit": preregistration_seal,
        "claim_commit": claim,
        "publication_commit": publication,
    }

    (tmp_path / "intervening.txt").write_text("X\n", encoding="utf-8")
    git("add", "intervening.txt")
    git("commit", "-m", "X")
    with pytest.raises(
        builder.TerminalG9CB3Failure,
        match="publication chain",
    ):
        builder._validate_committed_publication_topology(
            tmp_path,
            {"protocol_implementation_commit": implementation},
            git("rev-parse", "HEAD"),
        )


def test_parser_exposes_read_only_committed_publication_verifier() -> None:
    arguments = builder._parser().parse_args(["--verify-publication"])
    assert arguments.verify_publication is True
    assert arguments.create_claim is False
    assert arguments.produce is False
    assert arguments.internal_worker is False


def test_frozen_contract_and_deterministic_csv_gzip() -> None:
    assert builder.IDENTITY == "G9CB-3"
    assert builder.PROTOCOL_VERSION == (
        "gross9_structural_clock_bundle_g9cb3_v1"
    )
    assert builder.TERMINAL_ACTION == (
        "TERMINAL_G9CB3_ATTEMPT_CONSUMED_NO_RETRY"
    )
    assert builder._PYCACHE_PREFIX_RELATIVE == Path(
        "results/.g9cb3-bytecode-cache-disabled"
    )
    assert builder.PREREGISTRATION_PATH == (
        Path(
            "results/"
            "gross9_structural_clock_bundle_g9cb3_preregistration_2026-07-31.json"
        )
    )
    assert builder.DOMAIN_START == "2023-06-01T00:00:00Z"
    assert builder.DOMAIN_END == "2026-06-01T00:00:00Z"
    assert [row["name"] for row in builder.SLEEVES] == [
        "cand_rex_veto_7",
        "fresh_kimchi_fx",
        "frozen_annual_rank7",
        "markov_transition_long",
        "rex_taker_low_range_position",
    ]
    assert [row["weight"] for row in builder.SLEEVES] == [
        "1.6",
        "2.0",
        "3.0",
        "2.0",
        "0.4",
    ]
    assert sum(
        Decimal(row["weight"]) for row in builder.SLEEVES
    ) == Decimal("9.0")
    rows = [
        {
            "identity": "G9CB-3",
            "sleeve": "cand_rex_veto_7",
            "sleeve_order": 0,
            "configured_weight": "1.6",
            "interval_index": 0,
            "entry_time_utc": "2023-06-01T00:05:00Z",
            "exit_time_utc": "2023-06-01T12:05:00Z",
            "side": 1,
        }
    ]
    plain = builder.serialize_csv(rows)
    first = builder.compress_csv(plain)
    second = builder.compress_csv(plain)
    assert first == second
    assert first[:10] == bytes.fromhex("1f8b08000000000002ff")
    assert gzip.decompress(first) == plain
    assert plain.endswith(b"\n") and b"\r" not in plain


def test_g9cb3_stage_prefix_is_distinct_from_preserved_g9cb2_residue(
    tmp_path: Path,
) -> None:
    (tmp_path / "results").mkdir()
    active = Path(
        "results/.gross9-structural-clock-g9cb3-worker-active"
    )
    historical = Path(
        "results/.gross9-structural-clock-worker-ca9ca670ffb0d1b377ed6aef"
    )
    assert builder._worker_stage_path(tmp_path, active) == active.as_posix()
    with pytest.raises(builder.TerminalG9CB3Failure, match="name differs"):
        builder._worker_stage_path(tmp_path, historical)
    (tmp_path / historical).mkdir(mode=0o700)
    assert list(
        (tmp_path / "results").glob(
            ".gross9-structural-clock-g9cb3-worker-*"
        )
    ) == []


def test_five_sleeves_preserve_schedule_side_and_barrier_geometry() -> None:
    bars = _all_sleeve_bars()
    bars[1]["high"] = "105"
    bars[1]["low"] = "94"
    rows, counters = builder.reconstruct_intervals(bars)
    assert [row["sleeve"] for row in rows] == [
        row["name"] for row in builder.SLEEVES
    ]
    by_name = {row["sleeve"]: row for row in rows}
    assert by_name["fresh_kimchi_fx"]["exit_time_utc"] == _time(2)
    assert counters["per_sleeve"]["fresh_kimchi_fx"]["stop_exits"] == 1
    assert by_name["frozen_annual_rank7"]["exit_time_utc"] == _time(2)
    assert counters["per_sleeve"]["frozen_annual_rank7"]["take_exits"] == 1
    assert by_name["cand_rex_veto_7"]["exit_time_utc"] == _time(145)
    assert by_name["markov_transition_long"]["exit_time_utc"] == _time(577)
    assert by_name["rex_taker_low_range_position"]["side"] == -1
    assert counters["rows_used"]["outcome_dependent_ohlc_rows_examined"] == 2


def test_fresh_requires_exactly_one_side_gate() -> None:
    bars = _bars()
    bars[0]["decisions"] = {
        "fresh_kimchi_fx": {
            "active": True,
            "side": 1,
            "long_gate": True,
            "short_gate": True,
        }
    }
    with pytest.raises(builder.TerminalG9CB3Failure, match="exclusive"):
        builder.reconstruct_intervals(bars)


def test_reconstruction_rejects_zero_side() -> None:
    bars = _bars()
    bars[0]["decisions"] = {
        "cand_rex_veto_7": {"active": True, "side": 0}
    }
    with pytest.raises(builder.TerminalG9CB3Failure, match="forbidden side"):
        builder.reconstruct_intervals(bars)


def test_per_sleeve_nonoverlap_preserves_cross_sleeve_overlap() -> None:
    bars = _bars()
    for index in (0, 1, 145):
        bars[index]["decisions"] = {
            "cand_rex_veto_7": {"active": True, "side": 1},
            "rex_taker_low_range_position": {"active": True, "side": -1},
        }
    rows, _ = builder.reconstruct_intervals(bars)
    veto = [row for row in rows if row["sleeve"] == "cand_rex_veto_7"]
    taker = [
        row
        for row in rows
        if row["sleeve"] == "rex_taker_low_range_position"
    ]
    assert [row["entry_time_utc"] for row in veto] == [
        _time(1),
        _time(146),
    ]
    assert [row["entry_time_utc"] for row in taker] == [
        _time(1),
        _time(146),
    ]
    assert veto[0]["entry_time_utc"] == taker[0]["entry_time_utc"]


def test_rank7_premium_source_uses_exact_source_routed_exit() -> None:
    bars = _bars(600)
    bars[0]["decisions"] = {
        "frozen_annual_rank7": {
            "active": True,
            "side": 1,
            "source": "premium",
        }
    }
    bars[3]["low"] = "96"
    rows, counters = builder.reconstruct_intervals(bars)
    row = next(
        row for row in rows if row["sleeve"] == "frozen_annual_rank7"
    )
    assert row["exit_time_utc"] == _time(4)
    assert counters["per_sleeve"]["frozen_annual_rank7"]["stop_exits"] == 1
    assert counters["per_sleeve"]["frozen_annual_rank7"]["take_exits"] == 0


def test_structural_engine_is_ohlc_only_and_stop_precedes_take() -> None:
    market = pd.DataFrame(
        {
            "open": [100.0] * 8,
            "high": [100.0, 105.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "low": [100.0, 94.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        }
    )
    counters = builder._empty_counters()
    engine = builder._StructuralTradeEngine(
        market, counters, "fresh_kimchi_fx"
    )
    trade = engine.trade_at(0, 1, 4, 400, 250)
    assert trade is not None
    assert trade.exit_position == 1
    assert trade.exit_kind == "stop"
    for forbidden in (
        "price_factor",
        "funding_factor",
        "funding_debit_factor",
        "gross_return",
        "adverse_price_factor",
    ):
        assert not hasattr(trade, forbidden)
    assert counters["rows_used"]["outcome_dependent_ohlc_rows_examined"] == 1
    assert engine.trade_at(0, 1, 4, 400, 250) is trade
    assert counters["rows_used"]["outcome_dependent_ohlc_rows_examined"] == 1


def test_rank7_label_engine_exposes_only_authorized_factors() -> None:
    dates = pd.date_range("2023-06-01", periods=8, freq="5min")
    market = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0] * 8,
            "high": [100.0, 105.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "low": [100.0, 94.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        }
    )
    funding = pd.DataFrame(
        {"date": [dates[1]], "funding_rate": [0.001]}
    )
    counters = builder._empty_counters()
    engine = builder._Rank7LabelEngine(market, funding, counters, np)
    trade = engine.trade_at(0, 1, 4, 400, 250)
    assert trade is not None
    assert trade.exit_position == 1
    assert trade.price_factor == pytest.approx(0.9875)
    assert trade.adverse_price_factor == pytest.approx(0.9875)
    assert trade.funding_factor == pytest.approx(0.9995)
    assert trade.funding_debit_factor == pytest.approx(0.9995)
    assert not hasattr(trade, "__dict__")
    for forbidden in (
        "gross_return",
        "favorable_price_factor",
        "net",
        "pnl",
        "entry_date",
    ):
        assert not hasattr(trade, forbidden)


def test_terminal_domain_boundary_does_not_require_a_boundary_row() -> None:
    fixed = _bars(145)
    fixed[0]["decisions"] = {
        "cand_rex_veto_7": {"active": True, "side": 1}
    }
    rows, _ = builder.reconstruct_intervals(
        fixed,
        domain_end=_time(145),
    )
    assert rows[0]["exit_time_utc"] == _time(145)


def test_csv_reparse_rejects_noncanonical_gzip_header() -> None:
    rows, _ = builder.reconstruct_intervals(_all_sleeve_bars())
    raw = builder.compress_csv(builder.serialize_csv(rows))
    assert len(builder.validate_csv_gzip(raw)) == 5
    changed = bytearray(raw)
    changed[9] = 3
    with pytest.raises(builder.TerminalG9CB3Failure, match="prefix"):
        builder.validate_csv_gzip(bytes(changed))


def test_counter_schema_has_exact_source_and_sleeve_fields() -> None:
    counters = builder._empty_counters()
    assert tuple(counters) == (
        "file_access",
        "rows_decoded",
        "rows_used",
        "per_sleeve",
    )
    assert tuple(
        counters["file_access"]["bytes_read_by_logical_source"]
    ) == SOURCE_KEYS
    assert tuple(counters["rows_decoded"]) == SOURCE_KEYS
    assert tuple(
        counters["rows_used"]["causal_feature_rows_by_source"]
    ) == SOURCE_KEYS
    assert tuple(counters["per_sleeve"]) == tuple(
        row["name"] for row in builder.SLEEVES
    )
    assert all(
        tuple(fields) == SLEEVE_COUNTER_KEYS
        for fields in counters["per_sleeve"].values()
    )
    builder._validate_counter_contract(counters)


def test_counter_schema_rejects_an_extra_source_key() -> None:
    counters = builder._empty_counters()
    counters["rows_decoded"]["legacy_adapter"] = 0
    with pytest.raises(builder.TerminalG9CB3Failure, match="counter names"):
        builder._validate_counter_contract(counters)


def test_counted_csv_reader_counts_physical_decoder_reads(tmp_path: Path) -> None:
    source = tmp_path / "market.csv"
    source.write_bytes(b"a,b\n1,2\n3,4\n")
    counters = builder._empty_counters()
    original = builder._install_counted_csv_reader(
        pd,
        tmp_path,
        {"market_5m": str(source)},
        counters,
    )
    try:
        frame = pd.read_csv(source)
    finally:
        pd.read_csv = original
    assert len(frame) == 2
    assert counters["file_access"]["source_files_opened"] == 1
    assert (
        counters["file_access"]["bytes_read_by_logical_source"]["market_5m"]
        > 0
    )
    assert counters["rows_decoded"]["market_5m"] == 2


def test_jsonl_counter_increments_at_each_successful_physical_row(
    tmp_path: Path,
) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_bytes(b'{"value":1}\n\n{"value":2}\n')
    counters = builder._empty_counters()
    rows = builder._read_jsonl_rows(
        tmp_path,
        str(source),
        counters,
        "rex_taker_train",
    )
    assert [
        row["_g9cb_parser_ordinal"] for row in rows
    ] == [0, 1]
    assert counters["rows_decoded"]["rex_taker_train"] == 2
    assert (
        counters["file_access"]["bytes_read_by_logical_source"][
            "rex_taker_train"
        ]
        == source.stat().st_size
    )


def test_jsonl_counter_preserves_prior_success_before_terminal_bad_row(
    tmp_path: Path,
) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_bytes(b'{"value":1}\nnot-json\n')
    counters = builder._empty_counters()
    with pytest.raises(
        builder.TerminalG9CB3Failure,
        match="invalid JSONL row",
    ):
        builder._read_jsonl_rows(
            tmp_path,
            str(source),
            counters,
            "rex_taker_train",
        )
    assert counters["rows_decoded"]["rex_taker_train"] == 1


def test_rank7_runtime_rejects_duplicate_model_open(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    models = bundle / "models"
    state_dir = bundle / "state"
    models.mkdir(parents=True)
    state_dir.mkdir()
    history = state_dir / "completed_hourly_history.csv.gz"
    history.write_bytes(b"history")
    model_paths = []
    for seed in (7, 71, 715, 2026, 71515):
        path = models / f"seed_{seed}.npz"
        np.savez(path, seed=np.asarray([seed], dtype=np.int64))
        model_paths.append(path)
    preregistration = _synthetic_preregistration(tmp_path)
    preregistration["bindings"]["rank7_bundle"] = {
        "declared_files": [
            {"path": str(history)},
            *[{"path": str(path)} for path in model_paths],
        ]
    }

    class FakeModel:
        def predict(self, matrix: np.ndarray) -> np.ndarray:
            values = np.asarray(matrix)
            if values.ndim == 1:
                values = values.reshape(1, -1)
            return np.zeros((len(values), 2))

    runtime = SimpleNamespace(np=np, FrozenExtraTreesModel=FakeModel)
    counters = builder._empty_counters()
    original_load, original_predict, _ = (
        builder._install_counted_rank7_runtime(
            runtime,
            tmp_path,
            preregistration,
            counters,
        )
    )
    try:
        with runtime.np.load(model_paths[0], allow_pickle=False):
            pass
        with pytest.raises(
            builder.TerminalG9CB3Failure,
            match="more than once",
        ):
            runtime.np.load(model_paths[0], allow_pickle=False)
    finally:
        builder._restore_counted_rank7_runtime(
            runtime,
            original_load,
            original_predict,
        )


def test_core_is_deterministic_and_binds_parent_authentication() -> None:
    rows, counters = builder.reconstruct_intervals(_all_sleeve_bars())
    compressed = builder.compress_csv(builder.serialize_csv(rows))
    parent_authentication = {
        "environment": {"worker_process_environment": {}},
        "hashed_inputs": [],
        "runtime_import_closure": [],
    }
    arguments = (
        compressed,
        counters,
        {"synthetic_test_only": True},
        {"path": "claim", "sha256": "a" * 64, "claim_hash": "b" * 64},
        {
            "path": "sentinel",
            "sha256": "c" * 64,
            "manifest_hash": "d" * 64,
        },
        builder._expected_authority_amendment_bindings(),
        parent_authentication,
    )
    first = builder.build_core(*arguments)
    second = builder.build_core(*arguments)
    assert builder._canonical_json_bytes(first) == (
        builder._canonical_json_bytes(second)
    )
    assert first["manifest_hash"] == builder._object_hash(
        first, "manifest_hash"
    )
    assert first["parent_authentication"] == parent_authentication
    assert first["parent_authentication_sha256"] == hashlib.sha256(
        builder._canonical_json_bytes(
            parent_authentication,
            trailing_lf=False,
        )
    ).hexdigest()


def test_core_rejects_three_amendment_drift() -> None:
    rows, counters = builder.reconstruct_intervals(_all_sleeve_bars())
    compressed = builder.compress_csv(builder.serialize_csv(rows))
    amendments = builder._expected_authority_amendment_bindings()
    amendments[1]["authority_commit"] = "0" * 40
    with pytest.raises(
        builder.TerminalG9CB3Failure,
        match="authority amendment",
    ):
        builder.build_core(
            compressed,
            counters,
            {"synthetic_test_only": True},
            {
                "path": "claim",
                "sha256": "a" * 64,
                "claim_hash": "b" * 64,
            },
            {
                "path": "sentinel",
                "sha256": "c" * 64,
                "manifest_hash": "d" * 64,
            },
            amendments,
            {},
        )


def test_prohibited_output_keys_are_valid_only_at_the_exact_zero_schema() -> None:
    malformed = {
        "evidence_boundary": {
            "prohibited_output_counters": builder._prohibited_assertions()
        },
        "nested": {"portfolio_return_values_computed": 0},
    }
    with pytest.raises(
        builder.TerminalG9CB3Failure,
        match="outside the canonical zero assertion",
    ):
        builder._validate_prohibited_output_placement(malformed)


def test_recursive_hashed_inputs_are_deduplicated_and_path_sorted(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.bin").write_bytes(b"a")
    (tmp_path / "z.bin").write_bytes(b"zzz")
    a_digest = hashlib.sha256(b"a").hexdigest()
    z_digest = hashlib.sha256(b"zzz").hexdigest()
    payload = {
        "z": [
            {
                "repository_path": "z.bin",
                "logical_path": "z.bin",
                "sha256": z_digest,
                "size_bytes": 3,
            },
            {"path": "a.bin", "sha256": a_digest},
        ],
        "a": {
            "duplicate": {
                "path": "z.bin",
                "sha256": z_digest,
                "path_type": "regular_file",
            }
        },
    }
    assert builder._validate_regular_hashed_inputs(
        tmp_path,
        payload,
        verify_git=False,
    ) == [
        {"path": "a.bin", "sha256": a_digest, "size_bytes": 1},
        {"path": "z.bin", "sha256": z_digest, "size_bytes": 3},
    ]


def test_hashed_inputs_accept_exact_tracked_untracked_and_external_pairs(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()

    git("init")
    git("config", "user.email", "g9cb3-test@example.invalid")
    git("config", "user.name", "G9CB-3 Test")
    tracked = repository / "tracked.bin"
    tracked.write_bytes(b"tracked")
    git("add", "tracked.bin")
    git("commit", "-m", "tracked")
    tracked_blob = git("rev-parse", "HEAD:tracked.bin")

    untracked = repository / "untracked.bin"
    untracked.write_bytes(b"untracked")
    external = tmp_path / "external.bin"
    external.write_bytes(b"external")
    assert builder.prereg._optional_git_metadata(
        "tracked.bin", repository
    ) == {"git_blob": tracked_blob, "git_mode": "100644"}
    assert builder.prereg._optional_git_metadata(
        "untracked.bin", repository
    ) == {"git_blob": None, "git_mode": None}
    assert builder.prereg._optional_git_metadata(
        external.as_posix(), repository
    ) == {"git_blob": None, "git_mode": None}
    payload = {
        "tracked": {
            "path": "tracked.bin",
            "sha256": hashlib.sha256(b"tracked").hexdigest(),
            "git_blob": tracked_blob,
            "git_mode": "100644",
        },
        "untracked": {
            "path": "untracked.bin",
            "sha256": hashlib.sha256(b"untracked").hexdigest(),
            "git_blob": None,
            "git_mode": None,
        },
        "external": {
            "path": external.as_posix(),
            "sha256": hashlib.sha256(b"external").hexdigest(),
            "git_blob": None,
            "git_mode": None,
        },
    }
    assert builder._validate_regular_hashed_inputs(
        repository, payload
    ) == [
        {
            "path": external.as_posix(),
            "sha256": hashlib.sha256(b"external").hexdigest(),
            "size_bytes": len(b"external"),
        },
        {
            "path": "tracked.bin",
            "sha256": hashlib.sha256(b"tracked").hexdigest(),
            "size_bytes": len(b"tracked"),
        },
        {
            "path": "untracked.bin",
            "sha256": hashlib.sha256(b"untracked").hexdigest(),
            "size_bytes": len(b"untracked"),
        },
    ]


@pytest.mark.parametrize(
    "pair",
    [
        {"git_blob": None},
        {"git_mode": None},
        {"git_blob": None, "git_mode": "100644"},
        {"git_blob": "a" * 40, "git_mode": None},
        {"git_blob": False, "git_mode": False},
        {"git_blob": 0, "git_mode": 0},
        {"git_blob": 0.0, "git_mode": 0.0},
        {"git_blob": [], "git_mode": []},
        {"git_blob": {}, "git_mode": {}},
        {"git_blob": "a" * 39, "git_mode": "100644"},
        {"git_blob": "a" * 40, "git_mode": "100755"},
    ],
)
def test_hashed_inputs_reject_malformed_git_pairs_before_byte_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pair: dict[str, object],
) -> None:
    (tmp_path / "input.bin").write_bytes(b"input")
    called = False

    def forbidden_read(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("bound bytes must not be read")

    monkeypatch.setattr(builder, "_read_bound_regular_bytes", forbidden_read)
    binding = {
        "path": "input.bin",
        "sha256": hashlib.sha256(b"input").hexdigest(),
        **pair,
    }
    with pytest.raises(builder.TerminalG9CB3Failure, match="Git metadata"):
        builder._validate_regular_hashed_inputs(
            tmp_path,
            {"binding": binding},
            verify_git=False,
        )
    assert called is False


def test_hashed_inputs_reject_absolute_repository_spelling_before_byte_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.bin"
    source.write_bytes(b"input")
    monkeypatch.setattr(
        builder,
        "_read_bound_regular_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("bound bytes must not be read")
        ),
    )
    with pytest.raises(
        builder.TerminalG9CB3Failure,
        match="must be repository-relative",
    ):
        builder._validate_regular_hashed_inputs(
            tmp_path,
            {
                "binding": {
                    "path": source.resolve().as_posix(),
                    "sha256": hashlib.sha256(b"input").hexdigest(),
                    "git_blob": None,
                    "git_mode": None,
                }
            },
        )


@pytest.mark.parametrize("kind", ["directory", "fifo", "socket", "device"])
def test_hashed_inputs_reject_nonregular_paths_before_byte_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    opened_socket: socket.socket | None = None
    if kind == "directory":
        candidate = tmp_path / "candidate"
        candidate.mkdir()
        path_text = "candidate"
    elif kind == "fifo":
        candidate = tmp_path / "candidate"
        os.mkfifo(candidate)
        path_text = "candidate"
    elif kind == "socket":
        candidate = tmp_path / "candidate"
        opened_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        opened_socket.bind(candidate.as_posix())
        path_text = "candidate"
    else:
        candidate = Path("/dev/null")
        path_text = candidate.resolve().as_posix()

    monkeypatch.setattr(
        builder,
        "_read_bound_regular_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("bound bytes must not be read")
        ),
    )
    try:
        with pytest.raises(
            builder.TerminalG9CB3Failure,
            match="not a regular file",
        ):
            builder._validate_regular_hashed_inputs(
                tmp_path,
                {
                    "binding": {
                        "path": path_text,
                        "sha256": "0" * 64,
                        "git_blob": None,
                        "git_mode": None,
                    }
                },
                verify_git=False,
            )
    finally:
        if opened_socket is not None:
            opened_socket.close()


def test_nonblocking_reader_rejects_fifo_without_waiting_for_writer(
    tmp_path: Path,
) -> None:
    fifo = tmp_path / "candidate"
    os.mkfifo(fifo)
    started = time.monotonic()
    with pytest.raises(
        builder.TerminalG9CB3Failure,
        match="not a regular file",
    ):
        builder._read_bound_regular_bytes(fifo, "candidate")
    assert time.monotonic() - started < 1.0


def test_hashed_inputs_reject_tracked_null_and_untracked_string_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=tmp_path,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()

    git("init")
    git("config", "user.email", "g9cb3-test@example.invalid")
    git("config", "user.name", "G9CB-3 Test")
    tracked = tmp_path / "tracked.bin"
    tracked.write_bytes(b"tracked")
    git("add", "tracked.bin")
    git("commit", "-m", "tracked")
    untracked = tmp_path / "untracked.bin"
    untracked.write_bytes(b"untracked")
    monkeypatch.setattr(
        builder,
        "_read_bound_regular_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("bound bytes must not be read")
        ),
    )
    cases = [
        (
            {
                "path": "tracked.bin",
                "sha256": hashlib.sha256(b"tracked").hexdigest(),
                "git_blob": None,
                "git_mode": None,
            },
            "absence proof",
        ),
        (
            {
                "path": "untracked.bin",
                "sha256": hashlib.sha256(b"untracked").hexdigest(),
                "git_blob": "a" * 40,
                "git_mode": "100644",
            },
            "tracked bound input Git classification",
        ),
    ]
    for binding, message in cases:
        with pytest.raises(builder.TerminalG9CB3Failure, match=message):
            builder._validate_regular_hashed_inputs(
                tmp_path, {"binding": binding}
            )
    for path in (
        builder.CLAIM_PATH,
        builder.SENTINEL_PATH,
        *builder.WORKER_LEDGER_PATHS,
        builder.CSV_PATH,
        builder.MANIFEST_PATH,
    ):
        assert not (tmp_path / path).exists()


def test_hashed_inputs_reject_paired_null_with_unborn_head_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subprocess.run(
        ["git", "init"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    (tmp_path / "untracked.bin").write_bytes(b"untracked")
    monkeypatch.setattr(
        builder,
        "_read_bound_regular_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("bound bytes must not be read")
        ),
    )
    with pytest.raises(
        builder.TerminalG9CB3Failure,
        match="Git absence proof differs",
    ):
        builder._validate_regular_hashed_inputs(
            tmp_path,
            {
                "binding": {
                    "path": "untracked.bin",
                    "sha256": hashlib.sha256(b"untracked").hexdigest(),
                    "git_blob": None,
                    "git_mode": None,
                }
            },
        )


def test_hashed_inputs_reject_index_head_drift_before_byte_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=tmp_path,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()

    git("init")
    git("config", "user.email", "g9cb3-test@example.invalid")
    git("config", "user.name", "G9CB-3 Test")
    source = tmp_path / "tracked.bin"
    source.write_bytes(b"head")
    git("add", "tracked.bin")
    git("commit", "-m", "head")
    head_blob = git("rev-parse", "HEAD:tracked.bin")
    source.write_bytes(b"index")
    git("add", "tracked.bin")
    monkeypatch.setattr(
        builder,
        "_read_bound_regular_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("bound bytes must not be read")
        ),
    )
    with pytest.raises(
        builder.TerminalG9CB3Failure,
        match="index/HEAD metadata mismatch",
    ):
        builder._validate_regular_hashed_inputs(
            tmp_path,
            {
                "binding": {
                    "path": "tracked.bin",
                    "sha256": hashlib.sha256(b"index").hexdigest(),
                    "git_blob": head_blob,
                    "git_mode": "100644",
                }
            },
        )


def test_hashed_inputs_reject_staged_mode_drift_before_byte_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=tmp_path,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()

    git("init")
    git("config", "user.email", "g9cb3-test@example.invalid")
    git("config", "user.name", "G9CB-3 Test")
    git("config", "core.filemode", "true")
    source = tmp_path / "tracked.bin"
    source.write_bytes(b"head")
    git("add", "tracked.bin")
    git("commit", "-m", "head")
    head_blob = git("rev-parse", "HEAD:tracked.bin")
    source.chmod(0o755)
    git("add", "tracked.bin")
    monkeypatch.setattr(
        builder,
        "_read_bound_regular_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("bound bytes must not be read")
        ),
    )
    with pytest.raises(
        builder.TerminalG9CB3Failure,
        match="index/HEAD metadata mismatch",
    ):
        builder._validate_regular_hashed_inputs(
            tmp_path,
            {
                "binding": {
                    "path": "tracked.bin",
                    "sha256": hashlib.sha256(b"head").hexdigest(),
                    "git_blob": head_blob,
                    "git_mode": "100644",
                }
            },
        )


def test_hashed_inputs_reject_worktree_blob_drift_after_one_opaque_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=tmp_path,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()

    git("init")
    git("config", "user.email", "g9cb3-test@example.invalid")
    git("config", "user.name", "G9CB-3 Test")
    source = tmp_path / "tracked.bin"
    source.write_bytes(b"head")
    git("add", "tracked.bin")
    git("commit", "-m", "head")
    head_blob = git("rev-parse", "HEAD:tracked.bin")
    source.write_bytes(b"worktree")
    original_read = builder._read_bound_regular_bytes
    reads: list[str] = []

    def counted_read(path: Path, path_text: str) -> object:
        reads.append(path_text)
        return original_read(path, path_text)

    monkeypatch.setattr(builder, "_read_bound_regular_bytes", counted_read)
    with pytest.raises(
        builder.TerminalG9CB3Failure,
        match="worktree Git blob mismatch",
    ):
        builder._validate_regular_hashed_inputs(
            tmp_path,
            {
                "binding": {
                    "path": "tracked.bin",
                    "sha256": hashlib.sha256(b"worktree").hexdigest(),
                    "git_blob": head_blob,
                    "git_mode": "100644",
                }
            },
        )
    assert reads == ["tracked.bin"]


def test_hashed_inputs_reject_duplicate_null_string_git_declarations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "input.bin").write_bytes(b"input")
    monkeypatch.setattr(
        builder,
        "_read_bound_regular_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("bound bytes must not be read")
        ),
    )
    digest = hashlib.sha256(b"input").hexdigest()
    with pytest.raises(
        builder.TerminalG9CB3Failure,
        match="conflicting duplicate input metadata",
    ):
        builder._validate_regular_hashed_inputs(
            tmp_path,
            {
                "bindings": [
                    {
                        "path": "input.bin",
                        "sha256": digest,
                        "git_blob": None,
                        "git_mode": None,
                    },
                    {
                        "path": "input.bin",
                        "sha256": digest,
                        "git_blob": "a" * 40,
                        "git_mode": "100644",
                    },
                ]
            },
            verify_git=False,
        )


def test_complete_git_inventory_fails_before_reading_earlier_valid_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "a.bin").write_bytes(b"a")
    (tmp_path / "z.bin").write_bytes(b"z")
    reads: list[str] = []
    original_read = builder._read_bound_regular_bytes

    def counted_read(path: Path, path_text: str) -> object:
        reads.append(path_text)
        return original_read(path, path_text)

    monkeypatch.setattr(builder, "_read_bound_regular_bytes", counted_read)
    with pytest.raises(
        builder.TerminalG9CB3Failure,
        match="partial bound input Git metadata pair",
    ):
        builder._validate_regular_hashed_inputs(
            tmp_path,
            {
                "a": {
                    "path": "a.bin",
                    "sha256": hashlib.sha256(b"a").hexdigest(),
                },
                "z": {
                    "path": "z.bin",
                    "sha256": hashlib.sha256(b"z").hexdigest(),
                    "git_blob": None,
                },
            },
            verify_git=False,
        )
    assert reads == []


def test_parent_preclassifies_failed_attempt_git_pairs_before_single_reads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    preregistration, _binding, _parent = (
        _prepare_guarded_metadata_repository(tmp_path)
    )

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=tmp_path,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()

    git("init")
    git("config", "user.email", "g9cb3-test@example.invalid")
    git("config", "user.name", "G9CB-3 Test")
    git("add", ".")
    git("commit", "-m", "fixture")
    events: list[tuple[str, str]] = []
    original_git = builder._git_process
    original_read = builder._read_bound_regular_bytes

    def recorded_git(
        root: Path,
        *arguments: str,
    ) -> subprocess.CompletedProcess[bytes]:
        events.append(("git", arguments[-1]))
        return original_git(root, *arguments)

    def recorded_read(
        path: Path,
        path_text: str,
    ) -> tuple[bytes, os.stat_result]:
        events.append(("read", path_text))
        return original_read(path, path_text)

    monkeypatch.setattr(builder, "_git_process", recorded_git)
    monkeypatch.setattr(
        builder,
        "_read_bound_regular_bytes",
        recorded_read,
    )
    builder._validate_regular_hashed_inputs(
        tmp_path,
        preregistration,
        verify_git=True,
    )
    attempt = builder.prereg.expected_failed_predecessor_attempts()[0]
    current_paths = [
        attempt[key]["path"]
        for key in (
            "authority_decision",
            "preregistration",
            "access_claim",
            "attempt_sentinel",
        )
    ]
    first_read = next(
        index for index, event in enumerate(events) if event[0] == "read"
    )
    for path_text in current_paths:
        git_events = [
            index
            for index, event in enumerate(events)
            if event == ("git", path_text)
        ]
        assert len(git_events) == 3
        assert max(git_events) < first_read
        assert events.count(("read", path_text)) == 1
    for binding in (
        builder.prereg.expected_failed_predecessor_preregistration_bindings()
    ):
        assert events.count(("read", binding["path"])) == 1


@pytest.mark.parametrize(
    "mutation",
    (
        "reserved-output-0",
        "reserved-output-1",
        "reserved-output-2",
        "reserved-output-3",
        "slot1-content",
        "slot1-mode",
        "slot2-created",
    ),
)
def test_parent_rejects_failed_predecessor_permanent_state_before_sentinel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
) -> None:
    preregistration, _binding, _parent = (
        _prepare_guarded_metadata_repository(tmp_path)
    )
    attempt = builder.prereg.expected_failed_predecessor_attempts()[0]
    if mutation.startswith("reserved-output-"):
        index = int(mutation.rsplit("-", 1)[1])
        target = tmp_path / attempt["permanently_absent_outputs"][index]
        target.write_bytes(b"forbidden")
        match = "permanently absent G9CB-2 output exists"
    elif mutation == "slot1-content":
        slot1 = tmp_path / attempt["residue"]["slot1_stage"]["path"]
        (slot1 / "forbidden").write_bytes(b"forbidden")
        match = "G9CB-2 slot-1 residue state differs"
    elif mutation == "slot1-mode":
        slot1 = tmp_path / attempt["residue"]["slot1_stage"]["path"]
        os.chmod(slot1, 0o755)
        match = "G9CB-2 slot-1 residue state differs"
    else:
        slot2 = tmp_path / attempt["residue"]["slot2_stage"]["path"]
        slot2.mkdir()
        match = "G9CB-2 slot-2 residue state differs"

    monkeypatch.setattr(
        builder.prereg,
        "validate_historical_preregistration_topology",
        lambda _root: None,
    )
    monkeypatch.setattr(
        builder.prereg,
        "validate_failed_v2_preregistration_topology",
        lambda _root: None,
    )
    monkeypatch.setattr(
        builder.prereg,
        "validate_failed_predecessor_attempt_history",
        lambda _root: None,
    )
    monkeypatch.setattr(
        builder.prereg,
        "validate_protocol_commit_topology",
        lambda _root: preregistration["protocol_implementation_commit"],
    )
    monkeypatch.setattr(
        builder.prereg,
        "validate_manifest",
        lambda *_args, **_kwargs: None,
    )

    def parent_claim_validation(
        root: Path,
    ) -> tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, object],
    ]:
        payload, binding = builder.validate_preregistration(
            root,
            validation_mode="parent",
        )
        return {}, {}, payload, binding

    sentinel_writes: list[Path] = []

    def record_sentinel_write(path: Path, _raw: bytes) -> None:
        sentinel_writes.append(path)
        raise AssertionError("sentinel writer must not be reached")

    monkeypatch.setattr(
        builder,
        "_validate_claim_commit",
        parent_claim_validation,
    )
    monkeypatch.setattr(
        builder,
        "_atomic_link_write_once",
        record_sentinel_write,
    )
    with pytest.raises(builder.TerminalG9CB3Failure, match=match):
        builder.produce_one_shot(tmp_path)
    assert sentinel_writes == []


def test_duplicate_accepted_bindings_use_one_opaque_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "input.bin").write_bytes(b"input")
    reads: list[str] = []
    original_read = builder._read_bound_regular_bytes

    def counted_read(path: Path, path_text: str) -> object:
        reads.append(path_text)
        return original_read(path, path_text)

    monkeypatch.setattr(builder, "_read_bound_regular_bytes", counted_read)
    digest = hashlib.sha256(b"input").hexdigest()
    assert builder._validate_regular_hashed_inputs(
        tmp_path,
        {
            "bindings": [
                {"path": "input.bin", "sha256": digest},
                {
                    "path": "input.bin",
                    "sha256": digest,
                    "size_bytes": len(b"input"),
                },
            ]
        },
        verify_git=False,
    ) == [
        {
            "path": "input.bin",
            "sha256": digest,
            "size_bytes": len(b"input"),
        }
    ]
    assert reads == ["input.bin"]


def test_stage_zero_parser_rejects_nonzero_and_conflicted_entries() -> None:
    blob = "a" * 40
    for output in (
        f"100644 {blob} 1\tinput.bin\n",
        (
            f"100644 {blob} 1\tinput.bin\n"
            f"100644 {blob} 2\tinput.bin\n"
        ),
    ):
        with pytest.raises(
            builder.TerminalG9CB3Failure,
            match="stage zero|exactly one",
        ):
            builder._parse_stage_zero_binding(output, "input.bin")


def test_nonstage_index_entries_fail_full_preflight_before_byte_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "input.bin").write_bytes(b"input")
    blob = "a" * 40

    def fake_git_process(
        _root: Path, *arguments: str
    ) -> subprocess.CompletedProcess[bytes]:
        if arguments[:2] == ("ls-files", "--stage"):
            output = (
                f"100644 {blob} 1\tinput.bin\n"
                f"100644 {blob} 2\tinput.bin\n"
            ).encode()
        elif arguments[0] == "ls-tree":
            output = f"100644 blob {blob}\tinput.bin\n".encode()
        else:
            output = b"input.bin\n"
        return subprocess.CompletedProcess(arguments, 0, output, b"")

    monkeypatch.setattr(builder, "_git_process", fake_git_process)
    monkeypatch.setattr(
        builder,
        "_read_bound_regular_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("bound bytes must not be read")
        ),
    )
    with pytest.raises(
        builder.TerminalG9CB3Failure,
        match="exactly one bound input index entry",
    ):
        builder._validate_regular_hashed_inputs(
            tmp_path,
            {
                "binding": {
                    "path": "input.bin",
                    "sha256": hashlib.sha256(b"input").hexdigest(),
                    "git_blob": blob,
                    "git_mode": "100644",
                }
            },
        )


@pytest.mark.parametrize(
    ("binding", "message"),
    [
        (
            {
                "path": "input.bin",
                "logical_path": "other.bin",
                "sha256": "a" * 64,
            },
            "conflicting path aliases",
        ),
        (
            {"path": 3, "sha256": "a" * 64},
            "non-string path alias",
        ),
        (
            {"path": "./input.bin", "sha256": "a" * 64},
            "not normalized",
        ),
    ],
)
def test_recursive_hashed_inputs_reject_malformed_path_bindings(
    tmp_path: Path,
    binding: dict[str, object],
    message: str,
) -> None:
    (tmp_path / "input.bin").write_bytes(b"input")
    (tmp_path / "other.bin").write_bytes(b"other")
    with pytest.raises(builder.TerminalG9CB3Failure, match=message):
        builder._validate_regular_hashed_inputs(
            tmp_path,
            {"nested": [binding]},
            verify_git=False,
        )


def test_bytecode_preflight_accepts_a_clean_noncanonical_root(
    tmp_path: Path,
) -> None:
    (tmp_path / "results").mkdir()
    builder._validate_bytecode_preflight(tmp_path)


@pytest.mark.parametrize("cache_kind", ["directory", "file", "fixed-prefix"])
def test_bytecode_preflight_rejects_repository_cache_artifacts(
    tmp_path: Path,
    cache_kind: str,
) -> None:
    (tmp_path / "results").mkdir()
    if cache_kind == "directory":
        (tmp_path / "pkg" / "__pycache__").mkdir(parents=True)
    elif cache_kind == "file":
        (tmp_path / "orphan.pyc").write_bytes(b"malicious")
    else:
        (tmp_path / "results" / ".g9cb3-bytecode-cache-disabled").mkdir()
    with pytest.raises(
        builder.TerminalG9CB3Failure,
        match="bytecode",
    ):
        builder._validate_bytecode_preflight(tmp_path)


def test_anonymous_pipe_capability_has_exact_schema_and_is_consumed_once(
    tmp_path: Path,
) -> None:
    (tmp_path / "results").mkdir()
    capability = builder._prepare_worker_capability(
        root=tmp_path,
        output_dir=(
            tmp_path / "results/.gross9-structural-clock-g9cb3-worker-slot1"
        ),
        slot=1,
        parent_pid=os.getpid(),
    )
    try:
        assert tuple(capability["row"]) == CAPABILITY_KEYS
        assert capability["row"]["carrier_kind"] == "anonymous_pipe_v1"
        token = builder._consume_worker_capability(
            capability["read_fd"],
            capability["row"],
        )
        capability["read_fd"] = -1
        assert hashlib.sha256(token).hexdigest() == (
            capability["row"]["token_sha256"]
        )
        builder._zero_token(token)
        assert token == bytearray(32)
    finally:
        descriptor = capability["read_fd"]
        if descriptor >= 0:
            os.close(descriptor)
        builder._zero_token(capability["token"])


def test_guarded_metadata_validation_reaches_capability_without_child_process(
    tmp_path: Path,
) -> None:
    _preregistration, binding, parent_authentication = (
        _prepare_guarded_metadata_repository(tmp_path)
    )
    own = "results/.gross9-structural-clock-g9cb3-worker-guarded-own"
    other = "results/.gross9-structural-clock-g9cb3-worker-guarded-other"
    (tmp_path / own).mkdir()
    script = tmp_path / "guarded_metadata.py"
    script.write_text(
        textwrap.dedent(
            """
            import hashlib
            import json
            import os
            from pathlib import Path

            from training import build_gross9_structural_clock_bundle as b

            root = Path(os.environ["G9CB_TEST_ROOT"])
            parent = json.loads(os.environ["G9CB_PARENT_AUTH"])
            claim_binding = json.loads(os.environ["G9CB_CLAIM_PREREG"])
            own = "results/.gross9-structural-clock-g9cb3-worker-guarded-own"
            other = "results/.gross9-structural-clock-g9cb3-worker-guarded-other"
            token = b"g" * 32
            read_fd, write_fd = os.pipe()
            os.write(write_fd, token)
            os.close(write_fd)
            info = os.fstat(read_fd)
            capability = {
                "slot": 1,
                "parent_pid": os.getpid(),
                "stage_directory": own,
                "carrier_kind": "anonymous_pipe_v1",
                "carrier_device": info.st_dev,
                "carrier_inode": info.st_ino,
                "token_sha256": hashlib.sha256(token).hexdigest(),
                "consumed_ledger_path": b.WORKER_LEDGER_PATHS[0].as_posix(),
            }
            traps = []
            def forbidden(name):
                def invoke(*args, **kwargs):
                    traps.append(name)
                    raise AssertionError(name)
                return invoke
            b._git = forbidden("_git")
            b._git_process = forbidden("_git_process")
            b.prereg._run_git = forbidden("_run_git")
            b.prereg._git_result = forbidden("_git_result")
            guard = b._WorkerIsolationGuard(
                root=root,
                own_stage=own,
                other_stage=other,
                ledger_paths=b.WORKER_LEDGER_PATHS,
            )
            guard.install()
            metadata = b._authenticate_guarded_worker_metadata(
                root,
                parent,
                claim_binding,
            )
            consumed = b._consume_worker_capability(read_fd, capability)
            print(json.dumps({
                "binding": metadata["preregistration_binding"],
                "authentication": metadata["authentication"],
                "token": bytes(consumed).decode("ascii"),
                "traps": traps,
                "counters": guard.counters(),
            }, sort_keys=True))
            """
        ),
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.update(
        {
            "G9CB_TEST_ROOT": tmp_path.as_posix(),
            "G9CB_PARENT_AUTH": json.dumps(
                parent_authentication, sort_keys=True, separators=(",", ":")
            ),
            "G9CB_CLAIM_PREREG": json.dumps(
                binding, sort_keys=True, separators=(",", ":")
            ),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": builder.REPOSITORY_ROOT.as_posix(),
        }
    )
    completed = subprocess.run(
        [sys.executable, "-B", str(script)],
        cwd=builder.REPOSITORY_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])
    assert result["binding"] == binding
    assert result["authentication"] == parent_authentication
    assert result["token"] == "g" * 32
    assert result["traps"] == []
    assert result["counters"]["child_process_creation_events"] == 0
    assert result["counters"]["unauthorized_write_or_ipc_events"] == 0


HISTORICAL_METADATA_CASES = (
    "g9cb1-v1",
    "g9cb1-v2",
    "g9cb2-authority-decision",
    "g9cb2-preregistration",
    "g9cb2-access-claim",
    "g9cb2-attempt-sentinel",
)

HISTORICAL_METADATA_MUTATION_CASES = tuple(
    (case, mutation)
    for case in HISTORICAL_METADATA_CASES
    for mutation in (
        ("mode", "sha256", "derived-blob")
        if case == "g9cb2-authority-decision"
        else ("mode", "canonical", "internal-hash", "sha256", "derived-blob")
    )
)


def _historical_metadata_binding(
    preregistration: dict[str, Any],
    case: str,
) -> dict[str, Any]:
    bindings = preregistration["bindings"]
    if case == "g9cb1-v1":
        return bindings["failed_predecessor_preregistrations"][0]
    if case == "g9cb1-v2":
        return bindings["failed_predecessor_preregistrations"][1]
    key = {
        "g9cb2-authority-decision": "authority_decision",
        "g9cb2-preregistration": "preregistration",
        "g9cb2-access-claim": "access_claim",
        "g9cb2-attempt-sentinel": "attempt_sentinel",
    }[case]
    return bindings["failed_predecessor_attempts"][0][key]


@pytest.mark.parametrize(
    ("case", "mutation"),
    HISTORICAL_METADATA_MUTATION_CASES,
)
def test_actual_guarded_metadata_routine_rejects_historical_file_tampering(
    tmp_path: Path,
    case: str,
    mutation: str,
) -> None:
    _preregistration, _binding, parent_authentication = (
        _prepare_guarded_metadata_repository(tmp_path)
    )
    active_path = tmp_path / builder.PREREGISTRATION_PATH
    active = json.loads(active_path.read_bytes())
    historical = _historical_metadata_binding(active, case)
    target = tmp_path / historical["path"]
    if mutation == "mode":
        os.chmod(
            target,
            0o555 if case == "g9cb2-authority-decision" else 0o644,
        )
    elif mutation == "sha256":
        os.chmod(target, 0o644)
        target.write_bytes(target.read_bytes() + b" ")
        os.chmod(target, 0o444)
    elif mutation == "derived-blob":
        historical["git_blob"] = "f" * 40
    else:
        payload = json.loads(target.read_bytes())
        if mutation == "canonical":
            raw = json.dumps(payload, sort_keys=True, indent=2).encode() + b"\n"
        else:
            payload["reviewer_tamper"] = True
            raw = builder._canonical_json_bytes(payload)
        os.chmod(target, 0o644)
        target.write_bytes(raw)
        os.chmod(target, 0o444)
        historical["sha256"] = hashlib.sha256(raw).hexdigest()
        historical["git_blob"] = builder._git_blob_id(raw)
        if "size_bytes" in historical:
            historical["size_bytes"] = len(raw)
    if mutation in {"canonical", "internal-hash", "derived-blob"}:
        active["manifest_hash"] = builder._object_hash(
            active,
            "manifest_hash",
        )
        active_raw = builder._canonical_json_bytes(active)
        os.chmod(active_path, 0o644)
        active_path.write_bytes(active_raw)
        os.chmod(active_path, 0o444)
        active_binding = {
            "path": builder.PREREGISTRATION_PATH.as_posix(),
            "sha256": hashlib.sha256(active_raw).hexdigest(),
            "manifest_hash": active["manifest_hash"],
        }
        parent_authentication["preregistration_authentication"] = {
            **active_binding,
            "protocol_implementation_commit": active[
                "protocol_implementation_commit"
            ],
        }
    else:
        active_binding = {
            "path": builder.PREREGISTRATION_PATH.as_posix(),
            "sha256": hashlib.sha256(active_path.read_bytes()).hexdigest(),
            "manifest_hash": active["manifest_hash"],
        }
    own = "results/.gross9-structural-clock-g9cb3-worker-mutation-own"
    other = "results/.gross9-structural-clock-g9cb3-worker-mutation-other"
    (tmp_path / own).mkdir()
    script = tmp_path / "guarded_mutation.py"
    script.write_text(
        textwrap.dedent(
            """
            import json
            import os
            from pathlib import Path

            from training import build_gross9_structural_clock_bundle as b

            root = Path(os.environ["G9CB_TEST_ROOT"])
            parent = json.loads(os.environ["G9CB_PARENT_AUTH"])
            claim_binding = json.loads(os.environ["G9CB_CLAIM_PREREG"])
            active = json.loads((root / b.PREREGISTRATION_PATH).read_bytes())
            b.prereg.expected_failed_predecessor_preregistration_bindings = (
                lambda: active["bindings"][
                    "failed_predecessor_preregistrations"
                ]
            )
            b.prereg.expected_failed_predecessor_attempts = (
                lambda: active["bindings"]["failed_predecessor_attempts"]
            )
            traps = []
            def forbidden(name):
                def invoke(*args, **kwargs):
                    traps.append(name)
                    raise AssertionError(name)
                return invoke
            b._git = forbidden("_git")
            b._git_process = forbidden("_git_process")
            b.prereg._run_git = forbidden("_run_git")
            b.prereg._git_result = forbidden("_git_result")
            guard = b._WorkerIsolationGuard(
                root=root,
                own_stage=(
                    "results/.gross9-structural-clock-g9cb3-"
                    "worker-mutation-own"
                ),
                other_stage=(
                    "results/.gross9-structural-clock-g9cb3-"
                    "worker-mutation-other"
                ),
                ledger_paths=b.WORKER_LEDGER_PATHS,
            )
            guard.install()
            try:
                b._authenticate_guarded_worker_metadata(
                    root,
                    parent,
                    claim_binding,
                )
            except b.TerminalG9CB3Failure as exc:
                print(json.dumps({
                    "status": "terminal",
                    "error": str(exc),
                    "traps": traps,
                    "counters": guard.counters(),
                }, sort_keys=True))
            else:
                raise AssertionError("historical mutation was accepted")
            """
        ),
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.update(
        {
            "G9CB_TEST_ROOT": tmp_path.as_posix(),
            "G9CB_PARENT_AUTH": json.dumps(
                parent_authentication,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "G9CB_CLAIM_PREREG": json.dumps(
                active_binding,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": builder.REPOSITORY_ROOT.as_posix(),
        }
    )
    completed = subprocess.run(
        [sys.executable, "-B", str(script)],
        cwd=builder.REPOSITORY_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])
    assert result["status"] == "terminal"
    assert result["traps"] == []
    assert result["counters"]["child_process_creation_events"] == 0
    assert result["counters"]["unauthorized_write_or_ipc_events"] == 0


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("delete", "schema"),
        ("add", "schema"),
        ("path-type", "field shape"),
        ("sha-malformed", "field shape"),
        ("manifest-malformed", "field shape"),
        ("commit-malformed", "field shape"),
        ("path-drift", "field shape|field binding"),
        ("sha-drift", "field binding"),
        ("manifest-drift", "field binding"),
        ("commit-drift", "field binding"),
    ),
)
def test_preregistration_authentication_rejects_every_field_before_capability(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    preregistration, binding, parent_authentication = (
        _prepare_guarded_metadata_repository(tmp_path)
    )
    record = parent_authentication["preregistration_authentication"]
    assert isinstance(record, dict)
    if mutation == "delete":
        record.pop("sha256")
    elif mutation == "add":
        record["extra"] = "forbidden"
    elif mutation == "path-type":
        record["path"] = 1
    elif mutation == "sha-malformed":
        record["sha256"] = "A" * 64
    elif mutation == "manifest-malformed":
        record["manifest_hash"] = "0" * 63
    elif mutation == "commit-malformed":
        record["protocol_implementation_commit"] = "0" * 39
    elif mutation == "path-drift":
        record["path"] = "results/other.json"
    elif mutation == "sha-drift":
        record["sha256"] = "f" * 64
    elif mutation == "manifest-drift":
        record["manifest_hash"] = "f" * 64
    else:
        record["protocol_implementation_commit"] = "f" * 40
    with pytest.raises(builder.TerminalG9CB3Failure, match=match):
        builder.validate_preregistration(
            tmp_path,
            validation_mode="guarded_worker",
            parent_authentication=parent_authentication,
            claim_preregistration=binding,
        )
    assert preregistration["identity"] == "G9CB-3"


def test_capability_preparation_orders_pipe_identity_before_mutable_token() -> None:
    source = textwrap.dedent(
        inspect.getsource(builder._prepare_worker_capability)
    )
    tree = ast.parse(source)
    calls: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and isinstance(
            node.func.value, ast.Name
        ):
            name = f"{node.func.value.id}.{node.func.attr}"
        elif isinstance(node.func, ast.Name):
            name = node.func.id
        else:
            continue
        calls.setdefault(name, node.lineno)
        assert not (
            name == "bytes"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "token"
        )
    assert (
        calls["os.pipe2"]
        < calls["os.fstat"]
        < calls["_fill_random_token"]
        < calls["_write_all"]
    )
    module_tree = ast.parse(
        Path(builder.__file__).read_text(encoding="utf-8")
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "bytes"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "token"
        for node in ast.walk(module_tree)
    )


def test_worker_invocation_has_exact_environment_and_no_legacy_transport(
    tmp_path: Path,
) -> None:
    (tmp_path / "training").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / builder.BUILDER_PATH).write_text("", encoding="utf-8")
    capability = {
        "row": _capability_row(parent_pid=os.getpid()),
        "read_fd": 91,
    }
    invocation = builder._prepare_worker(
        root=tmp_path,
        capability=capability,
        other_stage_directory=(
            "results/.gross9-structural-clock-g9cb3-worker-slot2"
        ),
        synthetic_input=None,
        parent_authentication=_parent_authentication(tmp_path),
    )
    assert invocation["command"][:4] == [
        sys.executable,
        "-B",
        str(tmp_path / builder.BUILDER_PATH),
        "--internal-worker",
    ]
    assert tuple(invocation["environment"]) == WORKER_ENVIRONMENT_KEYS
    assert len(invocation["environment"]) == 18
    assert invocation["environment"] == (
        builder.prereg.worker_process_environment(tmp_path)
    )
    joined = "\0".join(invocation["command"])
    descriptor_index = invocation["command"].index("--worker-capability-fd")
    assert invocation["command"][descriptor_index + 1] == "91"
    assert "--worker-token" not in joined
    assert "G9CB_WORKER_TOKEN" not in invocation["environment"]
    assert "PYTHONSTARTUP" not in invocation["environment"]


def test_internal_worker_bootstrap_installs_parent_death_guard_before_argparse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[object] = []

    class FakeGuard:
        def __init__(self, **kwargs: object) -> None:
            events.append(("guard", kwargs))

        def install(self) -> None:
            events.append("install")

    monkeypatch.setattr(
        builder,
        "_establish_parent_death_contract",
        lambda pid: events.append(("pdeath", pid)),
    )
    monkeypatch.setattr(builder, "_WorkerIsolationGuard", FakeGuard)
    arguments = [
        "--internal-worker",
        "--repository-root",
        str(tmp_path),
        "--output-dir",
        "results/.gross9-structural-clock-g9cb3-worker-one",
        "--other-stage-directory",
        "results/.gross9-structural-clock-g9cb3-worker-two",
        "--worker-capability-fd",
        "7",
        "--expected-parent-pid",
        "1234",
    ]
    guard = builder._early_worker_bootstrap(arguments)
    assert isinstance(guard, FakeGuard)
    assert events[0] == ("pdeath", 1234)
    assert events[1][0] == "guard"
    assert events[2] == "install"
    main_source = inspect.getsource(builder.main)
    assert main_source.index("_early_worker_bootstrap") < main_source.index(
        "_parser().parse_args"
    )
    assert "_establish_parent_death_contract" not in inspect.getsource(
        builder._worker_main
    )


def test_worker_popen_passes_only_the_bound_descriptor_and_closes_before_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    class FakeProcess:
        pid = 4321

        def wait(self) -> int:
            events.append("wait")
            return 0

        def poll(self) -> int:
            return 0

        def terminate(self) -> None:
            events.append("terminate")

    def fake_popen(
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        close_fds: bool,
        pass_fds: tuple[int, ...],
    ) -> FakeProcess:
        events.append(
            ("popen", command, cwd, env, close_fds, pass_fds)
        )
        return FakeProcess()

    monkeypatch.setattr(builder.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        builder.os,
        "close",
        lambda descriptor: events.append(("close", descriptor)),
    )
    capability = {"read_fd": 17}
    pid = builder._execute_prepared_worker(
        {
            "command": ["python", "-B", "builder.py", "--internal-worker"],
            "cwd": Path("/tmp"),
            "environment": {"ONLY": "BOUND"},
            "capability": capability,
        }
    )
    assert pid == 4321
    assert events[0][-2:] == (True, (17,))
    assert events[1:] == [("close", 17), "wait"]
    assert capability["read_fd"] == -1


def test_worker_ledger_has_exact_14_field_schema() -> None:
    amendments = builder._expected_authority_amendment_bindings()
    payload = builder._worker_ledger_payload(
        binding=_capability_row(),
        claim={"claim_hash": "a" * 64},
        preregistration={"manifest_hash": "b" * 64},
        sentinel={"manifest_hash": "c" * 64},
        authority_amendments=amendments,
    )
    assert tuple(payload) == LEDGER_KEYS
    assert len(payload) == 14
    assert payload["authority_amendments"] == amendments
    assert payload["status"] == (
        "consumed_before_runtime_or_value_access"
    )


def test_worker_receipt_has_exact_hmac_and_hash_schema() -> None:
    token = bytearray(range(32))
    payload = builder._worker_receipt_payload(
        binding=_capability_row(),
        worker_pid=5678,
        ledger_sha256="a" * 64,
        rebuild_invocations_started=1,
        rebuild_invocations_completed=1,
        guard_counters={
            "child_process_creation_events": 0,
            "other_stage_access_events": 0,
            "other_stage_absence_checks": 1,
            "other_slot_ledger_access_events": 0,
            "unauthorized_write_or_ipc_events": 0,
        },
        csv_gzip_sha256="b" * 64,
        per_pass_core_sha256="c" * 64,
        token=token,
    )
    completion = {
        key: payload[key] for key in RECEIPT_KEYS[:17]
    }
    assert tuple(payload) == RECEIPT_KEYS
    assert len(payload) == 19
    assert payload["completion_hmac_sha256"] == hmac.new(
        bytes(token),
        builder._canonical_json_bytes(completion, trailing_lf=False),
        hashlib.sha256,
    ).hexdigest()
    assert payload["receipt_hash"] == hashlib.sha256(
        builder._canonical_json_bytes(
            {key: payload[key] for key in RECEIPT_KEYS[:18]},
            trailing_lf=False,
        )
    ).hexdigest()


def test_final_manifest_requires_20_field_receipt_rows() -> None:
    receipt = {key: index for index, key in enumerate(RECEIPT_KEYS)}
    first = {**receipt, "slot": 1, "pass_receipt_sha256": "a" * 64}
    second = {**receipt, "slot": 2, "pass_receipt_sha256": "b" * 64}
    core = {
        "manifest_hash": "c" * 64,
        "evidence_boundary": {
            "prohibited_output_counters": builder._prohibited_assertions()
        },
    }
    consumption = [
        {
            "slot": slot,
            "parent_pid": 1234,
            "path": builder.WORKER_LEDGER_PATHS[slot - 1].as_posix(),
            "sha256": "d" * 64,
            "carrier_kind": "anonymous_pipe_v1",
            "carrier_device": slot,
            "carrier_inode": slot + 100,
            "token_sha256": "e" * 64,
        }
        for slot in (1, 2)
    ]
    manifest = builder._final_manifest(
        core,
        consumption,
        [first, second],
    )
    assert all(len(row) == 20 for row in manifest["rebuild_receipts"])
    assert [row["slot"] for row in manifest["rebuild_receipts"]] == [1, 2]


@pytest.mark.parametrize(
    "path",
    [
        "/proc/self/fd/9",
        "/proc/self/fdinfo/9",
        "/proc/self/task/7/fd/9",
        "/proc/thread-self/fd/9",
        "/proc/123/task/7/fdinfo/9",
        "/dev/fd/9",
    ],
)
def test_guard_rejects_procfd_and_devfd_namespaces(
    tmp_path: Path,
    path: str,
) -> None:
    guard = _guard(tmp_path)
    with pytest.raises(
        builder.TerminalG9CB3Failure,
        match="descriptor namespace",
    ):
        guard._checked_path(path)


def test_guard_process_and_ipc_inventories_are_exact_literals() -> None:
    inventories = _literal_string_tuples(
        builder._WorkerIsolationGuard._install_process_and_ipc_guards
    )
    assert (
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
    ) in inventories
    assert ("Popen", "run", "call", "check_call", "check_output") in inventories
    assert (
        "pipe",
        "pipe2",
        "openpty",
        "mkfifo",
        "mknod",
        "memfd_create",
        "eventfd",
        "pidfd_open",
    ) in inventories
    assert ("socket", "socketpair", "fromfd") in inventories
    assert (
        "Pipe",
        "Queue",
        "SimpleQueue",
        "JoinableQueue",
        "Manager",
    ) in inventories


def test_guard_path_and_mutation_inventories_are_exact_literals() -> None:
    path_inventories = _literal_string_tuples(
        builder._WorkerIsolationGuard._install_path_guards
    )
    mutation_inventories = _literal_string_tuples(
        builder._WorkerIsolationGuard._install_mutation_guards
    )
    assert (
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
    ) in path_inventories
    assert (
        "write_text",
        "write_bytes",
        "mkdir",
        "touch",
        "chmod",
        "unlink",
        "symlink_to",
        "hardlink_to",
    ) in path_inventories
    assert (
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
    ) in mutation_inventories
    assert (
        "TemporaryFile",
        "NamedTemporaryFile",
        "SpooledTemporaryFile",
        "mkstemp",
        "mkdtemp",
    ) in mutation_inventories


@pytest.mark.parametrize("callable_name", PROCESS_DESCRIPTOR_CALLABLES)
def test_guard_invokes_every_process_and_descriptor_callable(
    tmp_path: Path,
    callable_name: str,
) -> None:
    result = _run_isolation_call(tmp_path, "process", callable_name)
    absent_names = {callable_name}
    if callable_name.startswith("fcntl.fcntl:"):
        absent_names.add(f"fcntl.{callable_name.split(':', 1)[1]}")
    _assert_terminal_or_exact_absence(result, absent_names=absent_names)


@pytest.mark.parametrize("callable_name", PATH_OBSERVATION_CALLABLES)
def test_guard_invokes_every_path_observation_callable(
    tmp_path: Path,
    callable_name: str,
) -> None:
    result = _run_isolation_call(tmp_path, "path", callable_name)
    _assert_terminal_or_exact_absence(
        result,
        absent_names={callable_name},
    )


@pytest.mark.parametrize("callable_name", FILESYSTEM_MUTATION_CALLABLES)
def test_guard_invokes_every_filesystem_mutation_callable(
    tmp_path: Path,
    callable_name: str,
) -> None:
    result = _run_isolation_call(tmp_path, "mutation", callable_name)
    absent_names = {callable_name}
    if callable_name.startswith("os.open:"):
        absent_names.add(f"os.{callable_name.split(':', 1)[1]}")
    _assert_terminal_or_exact_absence(result, absent_names=absent_names)


@pytest.mark.parametrize("callable_name", IPC_CALLABLES)
def test_guard_invokes_every_ipc_callable(
    tmp_path: Path,
    callable_name: str,
) -> None:
    result = _run_isolation_call(tmp_path, "ipc", callable_name)
    _assert_terminal_or_exact_absence(
        result,
        absent_names={callable_name},
    )


@pytest.mark.parametrize("event_name", AUDIT_EVENTS)
def test_guard_invokes_every_mandatory_audit_event(
    tmp_path: Path,
    event_name: str,
) -> None:
    result = _run_isolation_call(tmp_path, "audit", event_name)
    _assert_terminal_or_exact_absence(
        result,
        absent_names={event_name},
    )


@pytest.mark.parametrize("callable_name", RETAINED_REFERENCE_CASES)
def test_guard_rejects_preinstall_retained_reference_bypass(
    tmp_path: Path,
    callable_name: str,
) -> None:
    result = _run_isolation_call(tmp_path, "retained", callable_name)
    _assert_terminal_or_exact_absence(
        result,
        absent_names={callable_name},
    )


def test_isolated_runtime_ast_has_no_no_audit_alias_or_capture() -> None:
    repository_root = Path(builder.__file__).resolve().parents[1]
    closure_paths = builder._discover_import_closure(
        repository_root,
        (
            "execution/gross9_rank7_clock_runtime.py",
            "training/gross9_structural_clock_primitives.py",
        ),
    )
    assert {
        path.as_posix() for path in closure_paths
    }.issuperset(
        {
        "execution/gross9_rank7_clock_runtime.py",
        "training/gross9_structural_clock_primitives.py",
        }
    )
    assert [
        violation
        for relative in closure_paths
        for violation in _no_audit_prebind_violations(
            repository_root / relative
        )
    ] == []


@pytest.mark.parametrize(("callable_name", "keyword"), DIR_FD_CASES)
def test_guard_rejects_dir_fd_on_every_supporting_callable(
    tmp_path: Path,
    callable_name: str,
    keyword: str,
) -> None:
    result = _run_isolation_call(
        tmp_path,
        "dir-fd",
        f"{callable_name}:{keyword}",
    )
    _assert_terminal_or_exact_absence(
        result,
        absent_names={callable_name},
    )


@pytest.mark.parametrize(
    "case",
    (
        "1:stat",
        "2:read",
        "2:stat",
        "2:list",
        "2:write",
    ),
)
def test_guard_rejects_both_slot_cross_ledger_cases(
    tmp_path: Path,
    case: str,
) -> None:
    result = _run_isolation_call(tmp_path, "cross-ledger", case)
    _assert_terminal_or_exact_absence(result, absent_names=set())
    assert result["counters"]["other_slot_ledger_access_events"] == 1


@pytest.mark.parametrize(
    "case",
    (
        "repository:read",
        "repository:write",
        "repository:retained-read",
        "repository:retained-write",
        "fixed-prefix:read",
        "fixed-prefix:write",
        "fixed-prefix:retained-read",
        "fixed-prefix:retained-write",
    ),
)
def test_guard_unconditionally_rejects_pycache_reads_and_writes(
    tmp_path: Path,
    case: str,
) -> None:
    result = _run_isolation_call(tmp_path, "pycache", case)
    _assert_terminal_or_exact_absence(result, absent_names=set())
    assert result["counters"]["unauthorized_write_or_ipc_events"] == 1


@pytest.mark.parametrize(
    "case",
    (
        "reentrant:read",
        "reentrant:write",
        "concurrent:read",
        "concurrent:write",
    ),
)
def test_guard_rejects_pycache_injection_during_legitimate_source_load(
    tmp_path: Path,
    case: str,
) -> None:
    result = _run_isolation_call(
        tmp_path,
        "pycache-source-load",
        case,
    )
    _assert_terminal_or_exact_absence(result, absent_names=set())
    assert result["counters"]["unauthorized_write_or_ipc_events"] == 1


@pytest.mark.parametrize(
    ("phase", "forbidden_name"),
    tuple(
        (phase, forbidden_name)
        for phase in ("pre-metadata", "pre-capability-read")
        for forbidden_name in FORBIDDEN_PATH_CASES
    ),
)
def test_guard_rejects_procfd_devfd_and_fifo_at_both_worker_phases(
    tmp_path: Path,
    phase: str,
    forbidden_name: str,
) -> None:
    result = _run_isolation_call(
        tmp_path,
        "forbidden-path",
        f"{phase}:{forbidden_name}",
    )
    _assert_terminal_or_exact_absence(result, absent_names=set())
    assert result["counters"]["unauthorized_write_or_ipc_events"] == 1


@pytest.mark.parametrize(
    "scenario",
    IMPORT_RECORDER_TERMINAL_SCENARIOS,
)
def test_import_recorder_rejects_invalid_lifecycle_transition(
    tmp_path: Path,
    scenario: str,
) -> None:
    result = _run_import_recorder_case(tmp_path, scenario)
    assert result["status"] == "terminal", result
    assert result["error_type"] == "TerminalG9CB3Failure"


def test_import_recorder_accepts_authenticated_new_sources_then_freezes(
    tmp_path: Path,
) -> None:
    result = _run_import_recorder_case(tmp_path, "new-and-freeze")
    expected = [
        "execution/gross9_rank7_clock_runtime.py",
        "training/gross9_structural_clock_primitives.py",
    ]
    assert result == {
        "status": "passed",
        "scenario": "new-and-freeze",
        "reconciled": expected,
        "frozen": expected,
    }


@pytest.mark.parametrize(
    ("scenario", "expected_path"),
    IMPORT_RECORDER_VALID_PRELOADS,
)
def test_import_recorder_accepts_only_authorized_preloads(
    tmp_path: Path,
    scenario: str,
    expected_path: str,
) -> None:
    result = _run_import_recorder_case(tmp_path, scenario)
    assert result["status"] == "passed", result
    assert expected_path in result["preloaded"]


def test_guarded_actual_runtime_roots_import_from_authenticated_source(
    tmp_path: Path,
) -> None:
    repository_root = Path(builder.__file__).resolve().parents[1]
    script = tmp_path / "guarded_runtime_import.py"
    script.write_text(
        textwrap.dedent(
            """
            import json
            import os
            from pathlib import Path

            from training import build_gross9_structural_clock_bundle as b
            from training import preregister_gross9_structural_clock_bundle as p

            root = Path(os.environ["G9CB_TEST_ROOT"])
            preregistration = {
                "bindings": {
                    "protocol": [
                        {"path": path.as_posix()}
                        for path in p.PROTOCOL_PATHS
                    ]
                }
            }
            closure = [
                {
                    "path": path.as_posix(),
                    "sha256": b._sha256_file(root / path),
                }
                for path in p.discover_import_closure(
                    p.RUNTIME_IMPORT_ROOTS,
                    root,
                )
            ]
            suffix = str(os.getpid())
            guard = b._WorkerIsolationGuard(
                root=root,
                own_stage=(
                    "results/.gross9-structural-clock-g9cb3-worker-import-own-"
                    + suffix
                ),
                other_stage=(
                    "results/.gross9-structural-clock-g9cb3-worker-import-other-"
                    + suffix
                ),
                ledger_paths=b.WORKER_LEDGER_PATHS,
            )
            guard.install()
            b._preload_runtime_package_initializers(
                root,
                preregistration,
            )
            recorder = b._RuntimeImportRecorder(
                root=root,
                preregistration=preregistration,
                runtime_closure=closure,
            )
            recorder.install()
            modules = b._import_authenticated_modules(root.as_posix())
            print(
                json.dumps(
                    {
                        "modules": sorted(modules),
                        "runtime": recorder.freeze(),
                        "preloaded": sorted(
                            recorder.preloaded_repository_paths
                        ),
                    },
                    sort_keys=True,
                )
            )
            """
        ),
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.update(
        {
            "G9CB_TEST_ROOT": str(repository_root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(repository_root),
            "PYTHONPYCACHEPREFIX": str(
                repository_root
                / "results/.g9cb3-bytecode-cache-disabled"
            ),
        }
    )
    completed = subprocess.run(
        [sys.executable, "-B", str(script)],
        cwd=repository_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.splitlines()[-1])
    assert payload["modules"] == list(builder.RUNTIME_IMPORT_MODULES)
    assert payload["runtime"] == [
        "execution/gross9_rank7_clock_runtime.py",
        "training/gross9_structural_clock_primitives.py",
    ]
    assert payload["preloaded"] == [
        "execution/__init__.py",
        "training/__init__.py",
        "training/build_gross9_structural_clock_bundle.py",
        "training/preregister_gross9_structural_clock_bundle.py",
    ]
    assert not (
        repository_root / "results/.g9cb3-bytecode-cache-disabled"
    ).exists()


def test_parent_death_contract_kills_worker_before_handoff() -> None:
    repository_root = Path(builder.__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(repository_root),
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import os;"
                "from training import "
                "build_gross9_structural_clock_bundle as b;"
                "b._establish_parent_death_contract(os.getppid() + 100000)"
            ),
        ],
        cwd=repository_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )
    assert completed.returncode == -signal.SIGKILL


@pytest.mark.parametrize(
    ("phase", "expected_bytes_read"),
    (
        ("after-handoff-before-read", 0),
        ("partial-read", 7),
        ("complete-consumption", 32),
    ),
)
def test_parent_death_kills_worker_at_every_post_handoff_phase(
    tmp_path: Path,
    phase: str,
    expected_bytes_read: int,
) -> None:
    marker, worker_pid = _run_parent_death_phase(tmp_path, phase)
    assert marker == {
        "phase": phase,
        "worker_pid": worker_pid,
        "bytes_read": expected_bytes_read,
    }
    assert _process_is_dead_or_zombie(worker_pid)


def test_guard_rejects_cross_stage_observation(tmp_path: Path) -> None:
    guard = _guard(tmp_path)
    with pytest.raises(
        builder.TerminalG9CB3Failure,
        match="other worker stage",
    ):
        guard._checked_path(guard.other_stage / "probe")
    assert guard.other_stage_access_events == 1


def test_guard_rejects_path_open_of_fifo(tmp_path: Path) -> None:
    guard = _guard(tmp_path)
    fifo = tmp_path / "named.fifo"
    os.mkfifo(fifo)
    with pytest.raises(
        builder.TerminalG9CB3Failure,
        match="FIFO",
    ):
        guard._checked_path(fifo, fifo_open=True)


@pytest.mark.parametrize("keyword", ["dir_fd", "src_dir_fd", "dst_dir_fd"])
def test_guard_rejects_every_dir_fd_variant(keyword: str) -> None:
    with pytest.raises(
        builder.TerminalG9CB3Failure,
        match=keyword,
    ):
        builder._WorkerIsolationGuard._reject_dir_fds({keyword: -100})


def test_atomic_link_write_once_is_mode_0444_and_never_overwrites(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.json"
    builder._atomic_link_write_once(path, b"first\n")
    assert path.read_bytes() == b"first\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    with pytest.raises(FileExistsError):
        builder._atomic_link_write_once(path, b"second\n")
    assert path.read_bytes() == b"first\n"


def test_synthetic_two_pass_publication_consumes_pipes_and_publishes_exactly_five(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (
        preregistration,
        preregistration_binding,
        claim,
        claim_binding,
        synthetic_input,
    ) = _prepare_synthetic_worker_repository(tmp_path)
    parent_authentication = _parent_authentication(tmp_path)
    assert set(parent_authentication) == {
        "environment",
        "hashed_inputs",
        "preregistration_authentication",
        "runtime_import_closure",
    }
    assert set(parent_authentication["preregistration_authentication"]) == {
        "manifest_hash",
        "path",
        "protocol_implementation_commit",
        "sha256",
    }
    lifecycle: list[tuple[str, str]] = []
    original_create = builder._create_stage_directory
    original_cleanup = builder._cleanup_successful_stage

    def recorded_create(stage: Path, results: Path) -> None:
        lifecycle.append(("create", stage.name))
        original_create(stage, results)

    def recorded_cleanup(stage: Path, results: Path) -> None:
        lifecycle.append(("cleanup", stage.name))
        original_cleanup(stage, results)

    monkeypatch.setattr(
        builder,
        "_validate_claim_commit",
        lambda root: (
            claim,
            claim_binding,
            preregistration,
            preregistration_binding,
        ),
    )
    monkeypatch.setattr(
        builder,
        "_validate_environment",
        lambda prereg, root: parent_authentication["environment"],
    )
    monkeypatch.setattr(
        builder,
        "_validate_regular_hashed_inputs",
        lambda root, prereg, **kwargs: [],
    )
    monkeypatch.setattr(
        builder,
        "_validate_static_closures",
        lambda root, prereg, **kwargs: {"runtime": []},
    )
    monkeypatch.setattr(builder, "_create_stage_directory", recorded_create)
    monkeypatch.setattr(builder, "_cleanup_successful_stage", recorded_cleanup)

    result = builder.produce_one_shot(
        tmp_path,
        synthetic_input=synthetic_input,
    )

    publication_paths = (
        builder.SENTINEL_PATH,
        *builder.WORKER_LEDGER_PATHS,
        builder.CSV_PATH,
        builder.MANIFEST_PATH,
    )
    assert result["identity"] == builder.IDENTITY
    assert [path.as_posix() for path in publication_paths] == [
        "results/gross9_structural_clock_bundle_g9cb3_attempt_consumed_2026-07-31.json",
        (
            "results/"
            "gross9_structural_clock_bundle_g9cb3_"
            "worker_capability_consumed_pass1_2026-07-31.json"
        ),
        (
            "results/"
            "gross9_structural_clock_bundle_g9cb3_"
            "worker_capability_consumed_pass2_2026-07-31.json"
        ),
        "results/gross9_structural_clock_bundle_g9cb3_2026-07-31.csv.gz",
        "results/gross9_structural_clock_bundle_g9cb3_manifest_2026-07-31.json",
    ]
    assert all((tmp_path / path).is_file() for path in publication_paths)
    assert lifecycle[0][0] == "create"
    assert lifecycle[1][0] == "cleanup"
    assert lifecycle[2][0] == "create"
    assert lifecycle[0][1] == lifecycle[1][1]
    assert lifecycle[2][1] != lifecycle[0][1]
    assert lifecycle[-1] == ("cleanup", lifecycle[2][1])

    sentinel = json.loads((tmp_path / builder.SENTINEL_PATH).read_bytes())
    manifest = json.loads((tmp_path / builder.MANIFEST_PATH).read_bytes())
    assert [row["slot"] for row in sentinel["worker_capabilities"]] == [1, 2]
    assert all(
        set(row) == set(CAPABILITY_KEYS) and len(row) == 8
        for row in sentinel["worker_capabilities"]
    )
    expected_parent_hash = hashlib.sha256(
        builder._canonical_json_bytes(
            parent_authentication,
            trailing_lf=False,
        )
    ).hexdigest()
    assert sentinel["parent_authentication_sha256"] == expected_parent_hash
    assert manifest["parent_authentication_sha256"] == expected_parent_hash
    assert manifest["parent_authentication"] == parent_authentication
    assert [row["slot"] for row in manifest["rebuild_receipts"]] == [1, 2]
    assert all(len(row) == 20 for row in manifest["rebuild_receipts"])
    assert (
        manifest["rebuild_receipts"][0]["worker_pid"]
        != manifest["rebuild_receipts"][1]["worker_pid"]
    )
    assert [
        row["slot"] for row in manifest["worker_capability_consumption"]
    ] == [1, 2]
    assert all(
        len(row) == 8
        for row in manifest["worker_capability_consumption"]
    )
    csv_raw = (tmp_path / builder.CSV_PATH).read_bytes()
    rows = builder.validate_csv_gzip(
        csv_raw,
        require_all_sleeves=True,
    )
    sentinel_raw = (tmp_path / builder.SENTINEL_PATH).read_bytes()
    sentinel_binding = {
        "path": builder.SENTINEL_PATH.as_posix(),
        "sha256": hashlib.sha256(sentinel_raw).hexdigest(),
        "manifest_hash": sentinel["manifest_hash"],
    }
    capabilities = builder._normalized_worker_capabilities(
        sentinel["worker_capabilities"]
    )
    ledger_hashes: dict[int, str] = {}
    expected_consumption: list[dict[str, object]] = []
    for path in builder.WORKER_LEDGER_PATHS:
        ledger_path = tmp_path / path
        ledger_raw = ledger_path.read_bytes()
        ledger = json.loads(ledger_raw)
        assert tuple(ledger) == tuple(sorted(LEDGER_KEYS))
        assert set(ledger) == set(LEDGER_KEYS)
        assert len(ledger) == 14
        assert stat.S_IMODE(ledger_path.stat().st_mode) == 0o444
        slot = int(ledger["slot"])
        capability = capabilities[slot - 1]
        ledger_sha256 = hashlib.sha256(ledger_raw).hexdigest()
        ledger_hashes[slot] = ledger_sha256
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
    amendments = builder._expected_authority_amendment_bindings()
    builder._validate_final_manifest_contract(
        manifest,
        csv_raw=csv_raw,
        rows=rows,
        synthetic=True,
        prereg_binding=preregistration_binding,
        claim_binding=claim_binding,
        sentinel_binding=sentinel_binding,
        authority_amendments=amendments,
        capabilities=capabilities,
        expected_consumption=expected_consumption,
        ledger_hashes=ledger_hashes,
        expected_parent_authentication=parent_authentication,
        sentinel_parent_authentication_sha256=expected_parent_hash,
    )

    forged_parent = copy.deepcopy(manifest)
    forged_parent["parent_authentication"]["hashed_inputs"] = [
        {"path": "forged.bin", "sha256": "f" * 64, "size_bytes": 1}
    ]
    forged_parent["parent_authentication_sha256"] = hashlib.sha256(
        builder._canonical_json_bytes(
            forged_parent["parent_authentication"],
            trailing_lf=False,
        )
    ).hexdigest()
    forged_parent["manifest_hash"] = builder._object_hash(
        forged_parent,
        "manifest_hash",
    )
    with pytest.raises(
        builder.TerminalG9CB3Failure,
        match="parent authentication binding",
    ):
        builder._validate_final_manifest_contract(
            forged_parent,
            csv_raw=csv_raw,
            rows=rows,
            synthetic=True,
            prereg_binding=preregistration_binding,
            claim_binding=claim_binding,
            sentinel_binding=sentinel_binding,
            authority_amendments=amendments,
            capabilities=capabilities,
            expected_consumption=expected_consumption,
            ledger_hashes=ledger_hashes,
            expected_parent_authentication=parent_authentication,
            sentinel_parent_authentication_sha256=expected_parent_hash,
        )

    drifted_counter = copy.deepcopy(manifest)
    drifted_counter["access_counters"]["rows_decoded"]["market_5m"] += 1
    drifted_counter["manifest_hash"] = builder._object_hash(
        drifted_counter,
        "manifest_hash",
    )
    with pytest.raises(
        builder.TerminalG9CB3Failure,
        match="receipt binding|counter|core contract",
    ):
        builder._validate_final_manifest_contract(
            drifted_counter,
            csv_raw=csv_raw,
            rows=rows,
            synthetic=True,
            prereg_binding=preregistration_binding,
            claim_binding=claim_binding,
            sentinel_binding=sentinel_binding,
            authority_amendments=amendments,
            capabilities=capabilities,
            expected_consumption=expected_consumption,
            ledger_hashes=ledger_hashes,
            expected_parent_authentication=parent_authentication,
            sentinel_parent_authentication_sha256=expected_parent_hash,
        )

    drifted_receipt = copy.deepcopy(manifest)
    receipt = drifted_receipt["rebuild_receipts"][0]
    receipt["per_pass_core_sha256"] = "0" * 64
    receipt_core = {
        key: value
        for key, value in receipt.items()
        if key not in {"receipt_hash", "pass_receipt_sha256"}
    }
    receipt["receipt_hash"] = hashlib.sha256(
        builder._canonical_json_bytes(
            receipt_core,
            trailing_lf=False,
        )
    ).hexdigest()
    receipt_without_pass = {
        key: value
        for key, value in receipt.items()
        if key != "pass_receipt_sha256"
    }
    receipt["pass_receipt_sha256"] = hashlib.sha256(
        builder._canonical_json_bytes(receipt_without_pass)
    ).hexdigest()
    drifted_receipt["manifest_hash"] = builder._object_hash(
        drifted_receipt,
        "manifest_hash",
    )
    with pytest.raises(
        builder.TerminalG9CB3Failure,
        match="receipt binding",
    ):
        builder._validate_final_manifest_contract(
            drifted_receipt,
            csv_raw=csv_raw,
            rows=rows,
            synthetic=True,
            prereg_binding=preregistration_binding,
            claim_binding=claim_binding,
            sentinel_binding=sentinel_binding,
            authority_amendments=amendments,
            capabilities=capabilities,
            expected_consumption=expected_consumption,
            ledger_hashes=ledger_hashes,
            expected_parent_authentication=parent_authentication,
            sentinel_parent_authentication_sha256=expected_parent_hash,
        )
    assert not list(
        (tmp_path / "results").glob(
            ".gross9-structural-clock-g9cb3-worker-*"
        )
    )
    assert not list(
        (tmp_path / "results").glob(
            "gross9_structural_clock_bundle_pass_receipt.json"
        )
    )
    assert not list(tmp_path.rglob("*.pyc"))
    assert not list(tmp_path.rglob("__pycache__"))


def test_source_has_no_pre2025_parser_or_legacy_worker_transport(
    tmp_path: Path,
) -> None:
    source = Path(builder.__file__).read_text(encoding="utf-8")
    prohibited_definitions = (
        "def _load_pre2025",
        "def _parse_pre2025",
        "def _candidate_adapter",
        "def _run_candidate",
    )
    assert not any(definition in source for definition in prohibited_definitions)
    assert "G9CB_SYNTHETIC_TEST_ROOT" not in source
    bindings = _synthetic_preregistration(tmp_path)["bindings"]
    assert "adapter_import_roots" not in bindings
    assert "adapter_import_closure" not in bindings


def test_rank7_signal_counter_increments_at_each_anchor_evaluation() -> None:
    tree = ast.parse(
        textwrap.dedent(
            inspect.getsource(builder._direct_generic_adapter_impl)
        )
    )
    loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "rank7_signals"
    ]
    assert len(loops) == 1
    loop = loops[0]
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "np"
        and node.func.attr == "count_nonzero"
        for node in ast.walk(loop)
    )
    increments = [
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.AugAssign)
        and "signal_rows_evaluated" in ast.unparse(node.target)
    ]
    assert len(increments) == 1
    assert isinstance(increments[0].op, ast.Add)
    assert isinstance(increments[0].value, ast.Constant)
    assert increments[0].value.value == 1


def test_rank7_parity_counter_increments_at_each_comparison() -> None:
    tree = ast.parse(
        textwrap.dedent(
            inspect.getsource(
                builder._rank7_bundle_activation_with_parity
            )
        )
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "np"
        and node.func.attr in {"array_equal", "count_nonzero"}
        for node in ast.walk(tree)
    )
    loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "enumerate"
        and node.iter.args
        and isinstance(node.iter.args[0], ast.Name)
        and node.iter.args[0].id == "valid"
    ]
    assert len(loops) == 1
    increments = [
        node
        for node in ast.walk(loops[0])
        if isinstance(node, ast.AugAssign)
        and "rank7_bundle_parity_rows_compared"
        in ast.unparse(node.target)
    ]
    assert len(increments) == 1
    assert isinstance(increments[0].op, ast.Add)
    assert isinstance(increments[0].value, ast.Constant)
    assert increments[0].value.value == 1
