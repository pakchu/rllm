from __future__ import annotations

import ast
import copy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import errno
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
import threading
import time
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytest

from training import build_gross9_structural_clock_bundle as builder
from training import gross9_structural_clock_primitives as primitives


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
    "ledger_carrier_kind",
    "ledger_device",
    "ledger_inode",
    "ledger_initial_type",
    "ledger_initial_mode",
    "ledger_initial_size",
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


def _reconstruct(
    bars: list[dict[str, object]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return builder.reconstruct_intervals(bars, domain_end=_time(len(bars)))


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
    failed_prepublication_closures = [
        {
            "identity": identity,
            "permanently_absent_outputs": [
                f"results/synthetic-{suffix}-reserved.json"
            ],
            "residue": {
                "bytecode_cache": {
                    "path": f"results/.synthetic-{suffix}-pycache",
                    "state": "absent",
                },
                "publication_stages": {
                    "glob": f"results/.synthetic-{suffix}-publish-*",
                    "state": "absent",
                },
                "worker_stages": {
                    "glob": f"results/.synthetic-{suffix}-worker-*",
                    "state": "absent",
                },
                **(
                    {
                        "capability_probes": {
                            "glob": "results/.synthetic-g9cb6-probe-*",
                            "state": "absent",
                        }
                    }
                    if identity == "G9CB-6"
                    else {}
                ),
            },
        }
        for identity, suffix in (("G9CB-5", "g9cb5"), ("G9CB-6", "g9cb6"))
    ]
    failed_pre_sentinel_closures = [
        {
            "identity": "G9CB-7",
            "bytecode_incident": {"directories": [], "files": []},
            "permanently_absent_outputs": [
                "results/synthetic-g9cb7-reserved.json"
            ],
            "residue": {
                "bytecode_cache": {
                    "path": "results/.synthetic-g9cb7-pycache",
                    "state": "absent",
                },
                "capability_probes": {
                    "glob": "results/.synthetic-g9cb7-probe-*",
                    "state": "absent",
                },
                "publication_stages": {
                    "glob": "results/.synthetic-g9cb7-publish-*",
                    "state": "absent",
                },
                "worker_stages": {
                    "glob": "results/.synthetic-g9cb7-worker-*",
                    "state": "absent",
                },
            },
        }
    ]
    return builder._with_hash(
        {
            "protocol_version": (
                "gross9_structural_clock_bundle_g9cb8_preregistration_v1"
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
                "failed_predecessor_prepublication_closures": (
                    failed_prepublication_closures
                ),
                "failed_predecessor_pre_sentinel_closures": (
                    failed_pre_sentinel_closures
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


def _synthetic_protocol_bindings(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for relative in (builder.BUILDER_PATH, builder.BUILDER_TEST_PATH):
        raw = (root / relative).read_bytes()
        rows.append(
            {
                "path": relative.as_posix(),
                "path_type": "regular_file",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "git_blob": builder._git_blob_id(raw),
                "git_mode": "100644",
            }
        )
    return rows


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
    builder_test = root / builder.BUILDER_TEST_PATH
    builder_test.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        builder.REPOSITORY_ROOT / builder.BUILDER_TEST_PATH,
        builder_test,
    )

    preregistration = _synthetic_preregistration(root)
    preregistration_bindings = preregistration["bindings"]
    assert isinstance(preregistration_bindings, dict)
    preregistration_bindings["protocol"] = _synthetic_protocol_bindings(root)
    preregistration = builder._with_hash(
        preregistration,
        "manifest_hash",
    )
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
    os.chmod(root / builder.CLAIM_PATH, 0o444)
    claim_binding = {
        "path": builder.CLAIM_PATH.as_posix(),
        "sha256": hashlib.sha256(claim_raw).hexdigest(),
        "claim_hash": str(claim["claim_hash"]),
        "protocol_parent_commit": "1" * 40,
        "claim_commit": "2" * 40,
    }
    synthetic_input = fixtures / "structural-bars.json"
    synthetic_bars = _all_sleeve_bars()
    synthetic_input.write_bytes(
        builder._canonical_json_bytes(
            {
                "bars": synthetic_bars,
                "domain_end": _time(len(synthetic_bars)),
            }
        )
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
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> tuple[dict[str, object], dict[str, str], dict[str, object]]:
    results = root / "results"
    results.mkdir()
    preregistration = _synthetic_preregistration(root)
    source_root = builder.prereg.REPOSITORY_ROOT
    attempts = builder.prereg.expected_failed_predecessor_attempts()
    attempt = attempts[0]
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
    builder_test = root / builder.BUILDER_TEST_PATH
    builder_test.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_root / builder.BUILDER_TEST_PATH, builder_test)
    preregistration_bindings = preregistration["bindings"]
    assert isinstance(preregistration_bindings, dict)
    preregistration_bindings["protocol"] = _synthetic_protocol_bindings(root)
    for row in attempts:
        slot1 = root / row["residue"]["slot1_stage"]["path"]
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
    if monkeypatch is not None:
        expected_prepublication_closures = copy.deepcopy(
            preregistration_bindings[
                "failed_predecessor_prepublication_closures"
            ]
        )
        expected_pre_sentinel_closures = copy.deepcopy(
            preregistration_bindings[
                "failed_predecessor_pre_sentinel_closures"
            ]
        )
        monkeypatch.setattr(
            builder.prereg,
            "expected_failed_predecessor_prepublication_closures",
            lambda: copy.deepcopy(expected_prepublication_closures),
        )
        monkeypatch.setattr(
            builder.prereg,
            "expected_failed_predecessor_pre_sentinel_closures",
            lambda: copy.deepcopy(expected_pre_sentinel_closures),
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
            f"results/.gross9-structural-clock-g9cb8-worker-slot{slot}"
        ),
        "carrier_kind": "anonymous_pipe_v1",
        "carrier_device": slot,
        "carrier_inode": slot + 100,
        "token_sha256": hashlib.sha256(bytes([slot]) * 32).hexdigest(),
        "consumed_ledger_path": (
            builder.WORKER_LEDGER_PATHS[slot - 1].as_posix()
        ),
        "ledger_carrier_kind": "unnamed_otmpfile_v1",
        "ledger_device": slot + 200,
        "ledger_inode": slot + 300,
        "ledger_initial_type": "regular_file",
        "ledger_initial_mode": "0600",
        "ledger_initial_size": 0,
    }


def _guard(root: Path) -> builder._WorkerIsolationGuard:
    (root / "results").mkdir(exist_ok=True)
    own = "results/.gross9-structural-clock-g9cb8-worker-own"
    other = "results/.gross9-structural-clock-g9cb8-worker-other"
    (root / own).mkdir(exist_ok=True)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    repository_fd = os.open(root, directory_flags)
    results_fd = os.open("results", directory_flags, dir_fd=repository_fd)
    filesystem_root_fd = os.open("/", directory_flags)
    ledger_fd = os.open(
        ".",
        os.O_RDWR
        | getattr(os, "O_TMPFILE", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
        dir_fd=results_fd,
    )
    return builder._WorkerIsolationGuard(
        root=root.resolve(),
        own_stage=own,
        other_stage=other,
        ledger_paths=builder.WORKER_LEDGER_PATHS,
        repository_fd=repository_fd,
        results_fd=results_fd,
        filesystem_root_fd=filesystem_root_fd,
        ledger_fd=ledger_fd,
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
own_stage = results / ".gross9-structural-clock-g9cb8-worker-own"
other_stage = results / ".gross9-structural-clock-g9cb8-worker-other"
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
        target = case_root / "results" / ".g9cb8-bytecode-cache-disabled" / "evil.pyc"
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
        / ".g9cb8-bytecode-cache-disabled"
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
        else case_root / "results" / ".g9cb8-bytecode-cache-disabled"
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
        / ".g9cb8-bytecode-cache-disabled"
    )
    pycache.mkdir(parents=True)
    (pycache / "injected.pyc").write_bytes(b"malicious-bytecode")
if family == "forbidden-path" and case.split(":", 1)[1] == "named-fifo":
    os.mkfifo(scratch / "named.fifo")

guard = builder._WorkerIsolationGuard(
    root=case_root.resolve(),
    own_stage=own_stage.relative_to(case_root).as_posix(),
    other_stage=other_stage.relative_to(case_root).as_posix(),
    ledger_paths=builder.WORKER_LEDGER_PATHS,
    repository_fd=(
        repository_fd := os.open(
            case_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
    ),
    results_fd=(
        results_fd := os.open(
            "results",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=repository_fd,
        )
    ),
    filesystem_root_fd=os.open(
        "/",
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    ),
    ledger_fd=os.open(
        ".",
        os.O_RDWR | os.O_TMPFILE | os.O_CLOEXEC,
        0o600,
        dir_fd=results_fd,
    ),
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
except builder.TerminalG9CB8Failure as exc:
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
    assert result["error_type"] == "TerminalG9CB8Failure"


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
except builder.TerminalG9CB8Failure as exc:
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
ledger_descriptor = int(sys.argv[5])
builder._establish_parent_death_contract(expected_parent_pid)
builder._validate_inherited_worker_descriptors(
    descriptor,
    ledger_descriptor,
)
observed = {
    "phase": phase,
    "worker_pid": os.getpid(),
    "bytes_read": 0,
    "descriptor_count": 2,
    "descriptors_distinct": descriptor != ledger_descriptor,
}
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
ledger_fd = os.open(
    marker_directory,
    os.O_RDWR | os.O_TMPFILE | os.O_CLOEXEC,
    0o600,
)
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
        str(ledger_fd),
    ],
    cwd=repository_root,
    env=environment,
    close_fds=True,
    pass_fds=(read_fd, ledger_fd),
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
        builder.TerminalG9CB8Failure,
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
        builder.TerminalG9CB8Failure,
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
        builder.TerminalG9CB8Failure,
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
        builder.TerminalG9CB8Failure,
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
        builder.TerminalG9CB8Failure,
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
    assert builder.IDENTITY == "G9CB-8"
    assert builder.PROTOCOL_VERSION == (
        "gross9_structural_clock_bundle_g9cb8_v1"
    )
    assert builder.TERMINAL_ACTION == (
        "TERMINAL_G9CB8_ATTEMPT_CONSUMED_NO_RETRY"
    )
    assert builder._PYCACHE_PREFIX_RELATIVE == Path(
        "results/.g9cb8-bytecode-cache-disabled"
    )
    assert builder.PREREGISTRATION_PATH == (
        Path(
            "results/"
            "gross9_structural_clock_bundle_g9cb8_preregistration_2026-07-31.json"
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
            "identity": "G9CB-8",
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


def test_g9cb8_stage_prefix_is_distinct_from_preserved_g9cb2_residue(
    tmp_path: Path,
) -> None:
    (tmp_path / "results").mkdir()
    active = Path(
        "results/.gross9-structural-clock-g9cb8-worker-active"
    )
    historical = Path(
        "results/.gross9-structural-clock-worker-ca9ca670ffb0d1b377ed6aef"
    )
    assert builder._worker_stage_path(tmp_path, active) == active.as_posix()
    with pytest.raises(builder.TerminalG9CB8Failure, match="name differs"):
        builder._worker_stage_path(tmp_path, historical)
    (tmp_path / historical).mkdir(mode=0o700)
    assert list(
        (tmp_path / "results").glob(
            ".gross9-structural-clock-g9cb8-worker-*"
        )
    ) == []


def test_five_sleeves_preserve_schedule_side_and_barrier_geometry() -> None:
    bars = _all_sleeve_bars()
    bars[1]["high"] = "105"
    bars[1]["low"] = "94"
    rows, counters = _reconstruct(bars)
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
    with pytest.raises(builder.TerminalG9CB8Failure, match="exclusive"):
        _reconstruct(bars)


def test_reconstruction_rejects_zero_side() -> None:
    bars = _bars()
    bars[0]["decisions"] = {
        "cand_rex_veto_7": {"active": True, "side": 0}
    }
    with pytest.raises(builder.TerminalG9CB8Failure, match="forbidden side"):
        _reconstruct(bars)


def test_per_sleeve_nonoverlap_preserves_cross_sleeve_overlap() -> None:
    bars = _bars()
    for index in (0, 1, 145):
        bars[index]["decisions"] = {
            "cand_rex_veto_7": {"active": True, "side": 1},
            "rex_taker_low_range_position": {"active": True, "side": -1},
        }
    rows, _ = _reconstruct(bars)
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
    rows, counters = _reconstruct(bars)
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


def test_market_grid_retains_warmup_and_derives_n_plus_one_boundaries() -> None:
    dates = pd.to_datetime([_time(index) for index in range(5)])
    market = pd.DataFrame({"date": dates})
    values, boundaries = builder._market_value_opens_and_boundaries(
        market,
        domain_start=_time(1),
        domain_end=_time(5),
    )
    assert values == [int(date.timestamp()) for date in dates]
    assert boundaries == [*values, int(pd.Timestamp(_time(5)).timestamp())]
    assert values[0] < int(pd.Timestamp(_time(1)).timestamp())
    assert len(boundaries) == len(values) + 1


@pytest.mark.parametrize(
    ("dates", "match"),
    [
        (
            [_time(0), _time(1), _time(2), _time(3), _time(4)],
            "domain-end value",
        ),
        ([_time(0), _time(1), _time(3)], "complete unique"),
        ([_time(0), _time(1), _time(1), _time(2), _time(3)], "complete unique"),
        (
            [_time(0), "2023-06-01T00:07:00Z", _time(2), _time(3)],
            "complete unique|off",
        ),
        ([_time(0), _time(1), _time(2)], "ends before"),
    ],
)
def test_market_grid_rejects_physical_end_gap_duplicate_offgrid_and_early_end(
    dates: list[str],
    match: str,
) -> None:
    market = pd.DataFrame({"date": pd.to_datetime(dates)})
    with pytest.raises(builder.TerminalG9CB8Failure, match=match):
        builder._market_value_opens_and_boundaries(
            market,
            domain_start=_time(0),
            domain_end=_time(4),
        )


def test_raw_duplicate_normalization_preserves_sorted_keep_last_policy() -> None:
    raw = pd.DataFrame(
        {
            "date": [_time(1), _time(0), _time(1), _time(2), _time(3)],
            "open": [11, 10, 99, 12, 13],
        }
    )
    normalized = primitives.normalise_market(
        raw,
        exclude_from=pd.Timestamp(_time(4)).tz_localize(None),
    )
    assert list(normalized["open"]) == [10, 99, 12, 13]
    values, boundaries = builder._market_value_opens_and_boundaries(
        normalized,
        domain_start=_time(0),
        domain_end=_time(4),
    )
    assert len(boundaries) == len(values) + 1


def test_structural_horizon_n_is_geometry_only_but_rank7_cap_still_needs_price() -> None:
    dates = pd.date_range("2023-06-01", periods=4, freq="5min")
    market = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0] * 4,
            "high": [100.0] * 4,
            "low": [100.0] * 4,
        }
    )
    counters = builder._empty_counters()
    structural = builder._StructuralTradeEngine(
        market, counters, "fresh_kimchi_fx"
    )
    trade = structural.trade_at(0, 1, 3, 1_000_000, 1_000_000)
    assert trade is not None
    assert trade.exit_position == len(market)
    assert counters["rows_used"]["outcome_dependent_ohlc_rows_examined"] == 3
    assert structural.trade_at(0, 1, 4, 1_000_000, 1_000_000) is None

    funding = pd.DataFrame({"date": [], "funding_rate": []})
    label_engine = builder._Rank7LabelEngine(market, funding, counters, np)
    assert label_engine.trade_at(0, 1, 3, 1_000_000, 1_000_000) is None
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(builder._StructuralTradeEngine.trade_at))
    )
    assert not any(
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Name)
        and node.slice.id == "horizon"
        for node in ast.walk(tree)
    )


def test_split_eligibility_uses_physical_hits_and_only_virtual_terminal_fixed() -> None:
    mask = np.asarray([True, True, False], dtype=bool)
    boundaries = [0, 300, 600, 900]
    assert not builder._structural_exit_is_eligible(
        mask, boundaries, 2, "fixed", 0, 600
    )
    assert builder._structural_exit_is_eligible(
        mask, boundaries, 1, "stop", 0, 600
    )
    assert boundaries[1 + 1] == 600
    assert not builder._structural_exit_is_eligible(
        mask, boundaries, 2, "stop", 0, 900
    )
    assert builder._structural_exit_is_eligible(
        mask, boundaries, 3, "fixed", 0, 1000
    )
    assert not builder._structural_exit_is_eligible(
        mask, boundaries, 3, "fixed", 0, 900
    )


def test_reference_fixed_and_barrier_terminal_exits_use_derived_boundary() -> None:
    fixed = _bars(145)
    fixed[0]["decisions"] = {
        "cand_rex_veto_7": {"active": True, "side": 1}
    }
    fixed_rows, _ = builder.reconstruct_intervals(
        fixed, domain_end=_time(len(fixed))
    )
    assert fixed_rows[0]["exit_time_utc"] == _time(len(fixed))
    fixed_market = pd.DataFrame(fixed)
    fixed_engine = builder._StructuralTradeEngine(
        fixed_market, builder._empty_counters(), "cand_rex_veto_7"
    )
    fixed_trade = fixed_engine.trade_at(
        0, 1, 144, 1_000_000, 1_000_000
    )
    assert fixed_trade is not None
    assert fixed_trade.exit_position == len(fixed)

    barrier = _bars(289)
    barrier[0]["decisions"] = {
        "fresh_kimchi_fx": {
            "active": True,
            "side": 1,
            "long_gate": True,
            "short_gate": False,
        }
    }
    barrier[-1]["low"] = "90"
    barrier_rows, counters = builder.reconstruct_intervals(
        barrier, domain_end=_time(len(barrier))
    )
    assert barrier_rows[0]["exit_time_utc"] == _time(len(barrier))
    assert counters["per_sleeve"]["fresh_kimchi_fx"]["stop_exits"] == 1
    barrier_market = pd.DataFrame(barrier)
    direct_counters = builder._empty_counters()
    barrier_engine = builder._StructuralTradeEngine(
        barrier_market, direct_counters, "fresh_kimchi_fx"
    )
    barrier_trade = barrier_engine.trade_at(0, 1, 288, 400, 250)
    assert barrier_trade is not None
    assert barrier_trade.exit_kind == "stop"
    assert barrier_trade.exit_position + 1 == len(barrier)
    assert _time(barrier_trade.exit_position + 1) == (
        barrier_rows[0]["exit_time_utc"]
    )
    assert direct_counters["rows_used"][
        "outcome_dependent_ohlc_rows_examined"
    ] == 288


def test_reference_fixed_scheduler_never_calls_barrier_or_reads_ohlc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = _bars(145)
    bars[0]["decisions"] = {
        "cand_rex_veto_7": {"active": True, "side": 1}
    }
    for row in bars:
        row["open"] = object()
        row["high"] = object()
        row["low"] = object()
    monkeypatch.setattr(
        builder,
        "_barrier_exit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fixed reference entered barrier replay")
        ),
    )
    rows, counters = builder.reconstruct_intervals(
        bars,
        domain_end=_time(len(bars)),
    )
    assert rows[0]["exit_time_utc"] == _time(len(bars))
    assert counters["rows_used"][
        "outcome_dependent_ohlc_rows_examined"
    ] == 0
    assert counters["per_sleeve"]["cand_rex_veto_7"] == {
        "signal_rows_evaluated": 1,
        "intervals_emitted": 1,
        "long_intervals": 1,
        "short_intervals": 0,
        "fixed_horizon_exits": 1,
        "take_exits": 0,
        "stop_exits": 0,
        "outcome_dependent_ohlc_rows_examined": 0,
    }


def _direct_structural_case(
    bars: list[dict[str, object]],
    sleeve: str,
    *,
    split_start: int,
    split_end: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    counters = builder._empty_counters()
    intervals = {row["name"]: [] for row in builder.SLEEVES}
    value_opens = [
        builder._parse_timestamp(str(row["time_utc"]))
        for row in bars
    ]
    boundaries = [*value_opens, value_opens[-1] + 300]
    mask = np.asarray(
        [
            split_start <= second < split_end
            for second in value_opens
        ],
        dtype=bool,
    )
    decision = bars[0]["decisions"][sleeve]
    counters["per_sleeve"][sleeve]["signal_rows_evaluated"] += 1
    spec = builder.SLEEVE_BY_NAME[sleeve]
    if spec["kind"] == "fixed":
        entry_position = 1
        exit_position = entry_position + int(spec["hold_bars"])
        builder._schedule_direct_fixed_interval(
            intervals,
            counters,
            sleeve,
            value_opens,
            boundaries,
            mask,
            entry_position,
            exit_position,
            int(decision["side"]),
            split_start,
            split_end,
        )
    else:
        engine = builder._StructuralTradeEngine(
            pd.DataFrame(bars),
            counters,
            sleeve,
        )
        take_bps = int(spec["take_bps"])
        stop_bps = int(spec["stop_bps"])
        trade = engine.trade_at(
            0,
            int(decision["side"]),
            int(spec["hold_bars"]),
            take_bps,
            stop_bps,
        )
        if trade is not None and builder._structural_exit_is_eligible(
            mask,
            boundaries,
            trade.exit_position,
            trade.exit_kind,
            split_start,
            split_end,
        ):
            boundary_position = trade.exit_position + (
                0 if trade.exit_kind == "fixed" else 1
            )
            builder._append_direct_interval(
                intervals,
                counters,
                sleeve,
                value_opens[trade.entry_position],
                boundaries[boundary_position],
                trade.side,
                trade.exit_kind,
            )
    return builder._materialize_direct_rows(intervals, counters), counters


@pytest.mark.parametrize(
    ("sleeve", "length", "split_end_index", "barrier_hit_index", "emits"),
    (
        ("cand_rex_veto_7", 145, 146, None, True),
        ("fresh_kimchi_fx", 289, 290, 288, True),
        ("cand_rex_veto_7", 200, 145, None, False),
        ("fresh_kimchi_fx", 300, 145, 144, True),
    ),
)
def test_direct_and_reference_structural_rows_and_complete_counters_match(
    sleeve: str,
    length: int,
    split_end_index: int,
    barrier_hit_index: int | None,
    emits: bool,
) -> None:
    bars = _bars(length)
    decision: dict[str, object] = {"active": True, "side": 1}
    if sleeve == "fresh_kimchi_fx":
        decision.update({"long_gate": True, "short_gate": False})
    bars[0]["decisions"] = {sleeve: decision}
    if barrier_hit_index is not None:
        bars[barrier_hit_index]["low"] = "90"
    split_start = builder._parse_timestamp(_time(0))
    split_end = builder._parse_timestamp(_time(split_end_index))
    reference_rows, reference_counters = builder.reconstruct_intervals(
        bars,
        domain_end=_time(length),
        split_bounds=[(_time(0), _time(split_end_index))],
    )
    direct_rows, direct_counters = _direct_structural_case(
        bars,
        sleeve,
        split_start=split_start,
        split_end=split_end,
    )
    assert direct_rows == reference_rows
    assert direct_counters == reference_counters
    assert bool(reference_rows) is emits
    if barrier_hit_index is not None and emits:
        assert reference_rows[0]["exit_time_utc"] == _time(
            barrier_hit_index + 1
        )


def test_markov_and_fresh_tail_ranges_preserve_interior_and_add_only_terminal() -> None:
    source = inspect.getsource(builder._direct_generic_adapter_impl)
    assert source.count("_schedule_direct_fixed_interval(") == 2
    assert "len(market) - markov_hold" in source
    assert 'len(market) - int(fresh_cfg["hold_bars"])' in source
    for length, hold, start, stride in (
        (500, 144, 143, 24),
        (600, 288, 143, 6),
    ):
        corrected = list(range(start, max(0, length - hold), stride))
        legacy_value_count = length + 1
        legacy = list(
            range(
                start,
                max(0, legacy_value_count - hold - 2),
                stride,
            )
        )
        assert [
            position
            for position in corrected
            if position + 1 + hold < length
        ] == legacy
        assert all(
            position + 1 + hold <= length
            for position in corrected
        )


def test_grid_validation_precedes_feature_model_and_schedule_access() -> None:
    source = inspect.getsource(builder._direct_generic_adapter_impl)
    validation = source.index("_market_value_opens_and_boundaries")
    assert validation < source.index("build_market_feature_frame")
    assert validation < source.index("_StructuralTradeEngine")


def test_csv_reparse_rejects_noncanonical_gzip_header() -> None:
    rows, _ = _reconstruct(_all_sleeve_bars())
    raw = builder.compress_csv(builder.serialize_csv(rows))
    assert len(builder.validate_csv_gzip(raw)) == 5
    changed = bytearray(raw)
    changed[9] = 3
    with pytest.raises(builder.TerminalG9CB8Failure, match="prefix"):
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
    with pytest.raises(builder.TerminalG9CB8Failure, match="counter names"):
        builder._validate_counter_contract(counters)


def test_counted_csv_reader_counts_physical_decoder_reads(tmp_path: Path) -> None:
    source = tmp_path / "market.csv"
    source.write_bytes(
        b"date,b\n2023-06-01T00:00:00Z,2\n"
        b"2023-06-01T00:05:00Z,4\n"
    )
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


def test_raw_physical_domain_end_is_rejected_before_normalization(
    tmp_path: Path,
) -> None:
    source = tmp_path / "market.csv"
    source.write_text(
        "date,open,high,low,close,volume\n"
        "2026-05-31T23:55:00Z,100,100,100,100,1\n"
        "2026-06-01T00:00:00Z,101,101,101,101,1\n",
        encoding="utf-8",
    )
    counters = builder._empty_counters()
    original = builder._install_counted_csv_reader(
        pd,
        tmp_path,
        {"market_5m": str(source)},
        counters,
    )
    try:
        with pytest.raises(
            builder.TerminalG9CB8Failure,
            match="physical domain-end",
        ):
            primitives.load_market(
                source,
                exclude_from=pd.Timestamp(builder.DOMAIN_END).tz_localize(None),
            )
    finally:
        pd.read_csv = original
    assert counters["rows_decoded"]["market_5m"] == 2
    assert counters["rows_used"]["causal_feature_rows_by_source"][
        "market_5m"
    ] == 0
    assert counters["rows_used"]["prediction_rows_scored"] == 0


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
        builder.TerminalG9CB8Failure,
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
            builder.TerminalG9CB8Failure,
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
    rows, counters = _reconstruct(_all_sleeve_bars())
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
    rows, counters = _reconstruct(_all_sleeve_bars())
    compressed = builder.compress_csv(builder.serialize_csv(rows))
    amendments = builder._expected_authority_amendment_bindings()
    amendments[1]["authority_commit"] = "0" * 40
    with pytest.raises(
        builder.TerminalG9CB8Failure,
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
        builder.TerminalG9CB8Failure,
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
    git("config", "user.email", "g9cb8-test@example.invalid")
    git("config", "user.name", "G9CB-8 Test")
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
    with pytest.raises(builder.TerminalG9CB8Failure, match="Git metadata"):
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
        builder.TerminalG9CB8Failure,
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
            builder.TerminalG9CB8Failure,
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
        builder.TerminalG9CB8Failure,
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
    git("config", "user.email", "g9cb8-test@example.invalid")
    git("config", "user.name", "G9CB-8 Test")
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
        with pytest.raises(builder.TerminalG9CB8Failure, match=message):
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
        builder.TerminalG9CB8Failure,
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
    git("config", "user.email", "g9cb8-test@example.invalid")
    git("config", "user.name", "G9CB-8 Test")
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
        builder.TerminalG9CB8Failure,
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
    git("config", "user.email", "g9cb8-test@example.invalid")
    git("config", "user.name", "G9CB-8 Test")
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
        builder.TerminalG9CB8Failure,
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
    _materialize_closed_phase_state(tmp_path)

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
    git("config", "user.email", "g9cb8-test@example.invalid")
    git("config", "user.name", "G9CB-8 Test")
    source = tmp_path / "tracked.bin"
    source.write_bytes(b"head")
    protocol_rows = []
    for relative in (builder.BUILDER_PATH, builder.BUILDER_TEST_PATH):
        candidate = tmp_path / relative
        candidate.parent.mkdir(parents=True, exist_ok=True)
        raw = f"# synthetic {relative.name}\n".encode()
        candidate.write_bytes(raw)
        protocol_rows.append(
            {
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "git_blob": builder._git_blob_id(raw),
                "git_mode": "100644",
            }
        )
    preregistration = {
        "bindings": {
            "inputs": [
                {
                    "path": "tracked.bin",
                    "sha256": hashlib.sha256(b"worktree").hexdigest(),
                    "git_blob": builder._git_blob_id(b"head"),
                    "git_mode": "100644",
                }
            ],
            "protocol": protocol_rows,
        }
    }
    preregistration_path = tmp_path / builder.PREREGISTRATION_PATH
    preregistration_path.parent.mkdir(parents=True, exist_ok=True)
    preregistration_path.write_bytes(
        builder._canonical_json_bytes(preregistration)
    )
    git("add", ".")
    git("commit", "-m", "head")
    source.write_bytes(b"worktree")
    active_path = ""
    opened: list[str] = []
    reads: list[tuple[str, int]] = []
    original_open_initial = builder._SecureBoundSnapshot.open_initial
    original_pread = builder.os.pread

    def recorded_open_initial(
        snapshot: builder._SecureBoundSnapshot,
        path_text: str,
        repository_relative: bool,
    ) -> tuple[bytes, os.stat_result]:
        nonlocal active_path
        active_path = path_text
        opened.append(path_text)
        return original_open_initial(snapshot, path_text, repository_relative)

    def recorded_pread(fd: int, size: int, offset: int) -> bytes:
        reads.append((active_path, fd))
        return original_pread(fd, size, offset)

    monkeypatch.setattr(
        builder._SecureBoundSnapshot,
        "open_initial",
        recorded_open_initial,
    )
    monkeypatch.setattr(builder.os, "pread", recorded_pread)
    with pytest.raises(
        builder.TerminalG9CB8Failure,
        match="worktree Git blob mismatch",
    ):
        builder._preauthenticate_parent_snapshot(
            tmp_path,
            preregistration,
            content_mode=builder._PREREGISTRATION_ONLY,
        )
    assert opened.count("tracked.bin") == 1
    tracked_reads = [fd for path_text, fd in reads if path_text == "tracked.bin"]
    assert len(tracked_reads) == 1


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
        builder.TerminalG9CB8Failure,
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
        builder.TerminalG9CB8Failure,
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


def test_parent_preclassifies_all_git_pairs_before_one_open_two_same_fd_reads(
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
    git("config", "user.email", "g9cb8-test@example.invalid")
    git("config", "user.name", "G9CB-8 Test")
    git("add", ".")
    git("commit", "-m", "fixture")
    events: list[tuple[str, str]] = []
    active_path = ""
    read_fds: dict[str, list[int]] = {}
    descriptor_paths: dict[int, str] = {}
    original_git = builder._git_process
    original_open_initial = builder._SecureBoundSnapshot.open_initial
    original_pread = builder.os.pread

    def recorded_git(
        root: Path,
        *arguments: str,
    ) -> subprocess.CompletedProcess[bytes]:
        events.append(("git", arguments[-1]))
        return original_git(root, *arguments)

    def recorded_open_initial(
        snapshot: builder._SecureBoundSnapshot,
        path_text: str,
        repository_relative: bool,
    ) -> tuple[bytes, os.stat_result]:
        nonlocal active_path
        active_path = path_text
        events.append(("open", path_text))
        result = original_open_initial(snapshot, path_text, repository_relative)
        descriptor_paths[snapshot.file_descriptors[path_text]] = path_text
        return result

    def recorded_pread(fd: int, size: int, offset: int) -> bytes:
        path_text = descriptor_paths.get(fd, active_path)
        events.append(("read", path_text))
        read_fds.setdefault(path_text, []).append(fd)
        return original_pread(fd, size, offset)

    monkeypatch.setattr(builder, "_git_process", recorded_git)
    monkeypatch.setattr(
        builder._SecureBoundSnapshot,
        "open_initial",
        recorded_open_initial,
    )
    monkeypatch.setattr(builder.os, "pread", recorded_pread)
    snapshot, _pairs = builder._preauthenticate_parent_snapshot(
        tmp_path,
        preregistration,
        content_mode=builder._PREREGISTRATION_ONLY,
    )
    try:
        snapshot.verify_final()
        expected_paths = set(snapshot.file_descriptors)
    finally:
        snapshot.close()
    attempts = builder.prereg.expected_failed_predecessor_attempts()
    current_paths = [
        attempts[0][key]["path"]
        for key in (
            "authority_decision",
            "preregistration",
            "access_claim",
            "attempt_sentinel",
        )
    ] + [
        attempts[1][key]["path"]
        for key in ("authority_decision", "preregistration", "access_claim")
    ] + [
        binding["path"]
        for binding in attempts[1]["terminal_evidence"].values()
    ]
    first_open = next(
        index for index, event in enumerate(events) if event[0] == "open"
    )
    assert expected_paths
    for path_text in expected_paths:
        git_events = [
            index
            for index, event in enumerate(events)
            if event == ("git", path_text)
        ]
        assert len(git_events) == 3
        assert max(git_events) < first_open
        assert events.count(("open", path_text)) == 1
        assert events.count(("read", path_text)) == 2
        assert len(set(read_fds[path_text])) == 1
    for path_text in current_paths:
        assert path_text in expected_paths
    for binding in (
        builder.prereg.expected_failed_predecessor_preregistration_bindings()
    ):
        assert binding["path"] in expected_paths


def test_claim_parent_entry_preclassifies_all_pairs_before_one_open_two_reads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    chain = _prepare_synthetic_a5_q5_p5_chain(tmp_path)
    root = chain["root"]
    events: list[tuple[str, str]] = []
    active_path = ""
    descriptor_paths: dict[int, str] = {}
    read_fds: dict[str, list[int]] = {}
    original_git_process = builder._git_process
    original_open_initial = builder._SecureBoundSnapshot.open_initial
    original_pread = builder.os.pread

    def recorded_git(
        repository: Path,
        *arguments: str,
    ) -> subprocess.CompletedProcess[bytes]:
        events.append(("pair", arguments[-1]))
        return original_git_process(repository, *arguments)

    def recorded_open_initial(
        snapshot: builder._SecureBoundSnapshot,
        path_text: str,
        repository_relative: bool,
    ) -> tuple[bytes, os.stat_result]:
        nonlocal active_path
        active_path = path_text
        events.append(("open", path_text))
        result = original_open_initial(
            snapshot,
            path_text,
            repository_relative,
        )
        descriptor_paths[snapshot.file_descriptors[path_text]] = path_text
        return result

    def recorded_pread(fd: int, size: int, offset: int) -> bytes:
        path_text = descriptor_paths.get(fd, active_path)
        events.append(("read", path_text))
        read_fds.setdefault(path_text, []).append(fd)
        return original_pread(fd, size, offset)

    monkeypatch.setattr(builder, "_git_process", recorded_git)
    monkeypatch.setattr(
        builder._SecureBoundSnapshot,
        "open_initial",
        recorded_open_initial,
    )
    monkeypatch.setattr(builder.os, "pread", recorded_pread)
    binding = builder.create_claim_only(
        root,
        expected_security_profile=_synthetic_security_profile(chain),
    )
    claim = _read_synthetic_claim(chain)
    assert binding == _synthetic_claim_binding(chain, claim)
    assert claim["identity"] == builder.IDENTITY
    assert (root / builder.CLAIM_PATH).is_file()
    expected_paths = set(descriptor_paths.values())
    first_open = next(
        index for index, event in enumerate(events) if event[0] == "open"
    )
    assert expected_paths
    for path_text in expected_paths:
        classifications = [
            index
            for index, event in enumerate(events)
            if event == ("pair", path_text)
        ]
        assert len(classifications) == 3, (
            path_text,
            classifications,
            first_open,
        )
        assert max(classifications) < first_open
        assert events.count(("open", path_text)) == 1
        assert events.count(("read", path_text)) == 2
        assert len(set(read_fds[path_text])) == 1


def _write_guarded_active_metadata(
    root: Path,
    preregistration_binding: Mapping[str, str],
) -> None:
    claim = builder._claim_payload(
        "1" * 40,
        preregistration_binding,
        builder._expected_authority_amendment_bindings(),
        [],
        [],
    )
    claim_path = root / builder.CLAIM_PATH
    claim_path.write_bytes(builder._canonical_json_bytes(claim))
    claim_path.chmod(0o444)
    sentinel = builder._with_hash(
        {"identity": builder.IDENTITY, "status": "authenticated"},
        "manifest_hash",
    )
    sentinel_path = root / builder.SENTINEL_PATH
    sentinel_path.write_bytes(builder._canonical_json_bytes(sentinel))
    sentinel_path.chmod(0o444)


def test_worker_auth_entry_reads_each_bound_path_once_and_uses_active_fstats(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    preregistration, binding, parent_authentication = (
        _prepare_guarded_metadata_repository(tmp_path, monkeypatch)
    )
    _write_guarded_active_metadata(tmp_path, binding)
    read_fds: list[int] = []
    original_pread = builder.os.pread
    original_lstat = builder.os.lstat
    active_paths = {
        builder.PREREGISTRATION_PATH.as_posix(),
        builder.CLAIM_PATH.as_posix(),
        builder.SENTINEL_PATH.as_posix(),
    }

    def recorded_pread(fd: int, size: int, offset: int) -> bytes:
        read_fds.append(fd)
        return original_pread(fd, size, offset)

    def guarded_lstat(
        path: object,
        *args: object,
        **kwargs: object,
    ) -> os.stat_result:
        try:
            relative = Path(path).relative_to(tmp_path).as_posix()
        except (TypeError, ValueError):
            relative = ""
        if relative in active_paths:
            raise AssertionError(f"active metadata lstat reread: {relative}")
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(builder.os, "pread", recorded_pread)
    monkeypatch.setattr(builder.os, "lstat", guarded_lstat)
    result = builder._authenticate_worker_metadata_entry(
        tmp_path,
        parent_authentication,
        synthetic=False,
    )
    snapshot = result["snapshot"]
    try:
        expected_paths = {
            builder._binding_path(row)
            for row in builder._iter_bindings(preregistration)
        } | active_paths
        assert set(result["raw_cache"]) == expected_paths
        assert set(snapshot.file_descriptors) == expected_paths
        snapshot.verify_final()
        assert all(
            read_fds.count(descriptor) == 2
            for descriptor in snapshot.file_descriptors.values()
        )
        assert len(set(snapshot.file_descriptors.values())) == len(
            expected_paths
        )
        assert not any(
            path.as_posix() in snapshot.file_descriptors
            for path in builder.WORKER_LEDGER_PATHS
        )
    finally:
        snapshot.close()


@pytest.mark.parametrize(
    ("relative", "message"),
    (
        (
            builder.PREREGISTRATION_PATH,
            "preregistration filesystem mode differs",
        ),
        (builder.CLAIM_PATH, "active claim filesystem mode differs"),
        (builder.SENTINEL_PATH, "active sentinel filesystem mode differs"),
    ),
)
def test_worker_auth_entry_rejects_active_mode_mutation_from_open_fstat(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relative: Path,
    message: str,
) -> None:
    _preregistration, binding, parent_authentication = (
        _prepare_guarded_metadata_repository(tmp_path, monkeypatch)
    )
    _write_guarded_active_metadata(tmp_path, binding)
    (tmp_path / relative).chmod(0o644)
    with pytest.raises(builder.TerminalG9CB8Failure, match=message):
        builder._authenticate_worker_metadata_entry(
            tmp_path,
            parent_authentication,
            synthetic=False,
        )


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
        _prepare_guarded_metadata_repository(tmp_path, monkeypatch)
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
        **_kwargs: object,
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
    with pytest.raises(builder.TerminalG9CB8Failure, match=match):
        builder.produce_one_shot(tmp_path)
    assert sentinel_writes == []


def test_duplicate_accepted_bindings_use_one_open_two_same_fd_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "input.bin").write_bytes(b"input")
    reads: list[int] = []
    original_pread = builder.os.pread

    def recorded_pread(fd: int, size: int, offset: int) -> bytes:
        reads.append(fd)
        return original_pread(fd, size, offset)

    monkeypatch.setattr(builder.os, "pread", recorded_pread)
    digest = hashlib.sha256(b"input").hexdigest()
    snapshot = builder._SecureBoundSnapshot(tmp_path)
    try:
        snapshot.open_initial("input.bin", True)
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
            raw_cache=snapshot,
        ) == [
            {
                "path": "input.bin",
                "sha256": digest,
                "size_bytes": len(b"input"),
            }
        ]
        snapshot.verify_final()
        assert len(snapshot.file_descriptors) == 1
        assert len(reads) == 2
        assert len(set(reads)) == 1
    finally:
        snapshot.close()


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
            builder.TerminalG9CB8Failure,
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
        builder.TerminalG9CB8Failure,
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
    with pytest.raises(builder.TerminalG9CB8Failure, match=message):
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


@pytest.mark.parametrize(
    "cache_kind", ["directory", "pyc", "pyo", "fixed-prefix"]
)
def test_bytecode_preflight_rejects_repository_cache_artifacts(
    tmp_path: Path,
    cache_kind: str,
) -> None:
    (tmp_path / "results").mkdir()
    if cache_kind == "directory":
        (tmp_path / "pkg" / "__pycache__").mkdir(parents=True)
    elif cache_kind in {"pyc", "pyo"}:
        (tmp_path / f"orphan.{cache_kind}").write_bytes(b"malicious")
    else:
        (tmp_path / "results" / ".g9cb8-bytecode-cache-disabled").mkdir()
    with pytest.raises(
        builder.TerminalG9CB8Failure,
        match="bytecode",
    ):
        builder._validate_bytecode_preflight(tmp_path)


@pytest.mark.parametrize("entry_name", ["create_claim_only", "produce_one_shot"])
def test_c8_and_d8_bytecode_gate_precedes_git_snapshot_open_and_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_name: str,
) -> None:
    events: list[str] = []
    traversal_error = PermissionError("synthetic traversal failure")

    def failing_walk(
        root: Path,
        *,
        topdown: bool,
        onerror: Any,
        followlinks: bool,
    ) -> Any:
        events.append("bytecode")
        assert root == tmp_path.resolve()
        assert topdown is True
        assert followlinks is False
        assert callable(onerror)
        onerror(traversal_error)
        yield root, [], []

    def forbidden(name: str) -> Any:
        def call(*_args: Any, **_kwargs: Any) -> Any:
            events.append(name)
            raise AssertionError(f"{name} ran after bytecode rejection")

        return call

    monkeypatch.setattr(builder.prereg.os, "walk", failing_walk)
    monkeypatch.setattr(
        builder, "validate_claim_preflight", forbidden("claim-preflight")
    )
    monkeypatch.setattr(
        builder, "_validate_claim_commit", forbidden("claim-commit")
    )
    monkeypatch.setattr(
        builder, "_validate_git_pair_preflight", forbidden("git-pair")
    )
    monkeypatch.setattr(
        builder, "_read_bound_regular_bytes", forbidden("open")
    )
    monkeypatch.setattr(
        builder, "_PublicationContext", forbidden("publication-context")
    )
    with pytest.raises(
        builder.TerminalG9CB8Failure,
        match="bytecode traversal failed",
    ) as caught:
        getattr(builder, entry_name)(tmp_path)
    assert isinstance(caught.value.__cause__, ValueError)
    assert caught.value.__cause__.__cause__ is traversal_error
    assert events == ["bytecode"]


def test_anonymous_pipe_capability_has_exact_schema_and_is_consumed_once(
    tmp_path: Path,
) -> None:
    (tmp_path / "results").mkdir()
    capability = builder._prepare_worker_capability(
        root=tmp_path,
        output_dir=(
            tmp_path / "results/.gross9-structural-clock-g9cb8-worker-slot1"
        ),
        slot=1,
        parent_pid=os.getpid(),
    )
    try:
        assert tuple(capability["row"]) == CAPABILITY_KEYS
        assert capability["row"]["carrier_kind"] == "anonymous_pipe_v1"
        assert capability["read_fd"] != capability["ledger_fd"]
        assert stat.S_ISREG(os.fstat(capability["ledger_fd"]).st_mode)
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
        ledger_descriptor = capability["ledger_fd"]
        if ledger_descriptor >= 0:
            os.close(ledger_descriptor)
        builder._zero_token(capability["token"])


def test_guarded_metadata_validation_reaches_capability_in_two_fd_child(
    tmp_path: Path,
) -> None:
    _preregistration, binding, parent_authentication = (
        _prepare_guarded_metadata_repository(tmp_path)
    )
    own = "results/.gross9-structural-clock-g9cb8-worker-guarded-own"
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
            own = "results/.gross9-structural-clock-g9cb8-worker-guarded-own"
            other = "results/.gross9-structural-clock-g9cb8-worker-guarded-other"
            read_fd = int(os.environ["G9CB_CAPABILITY_FD"])
            ledger_fd = int(os.environ["G9CB_LEDGER_FD"])
            capability = json.loads(os.environ["G9CB_CAPABILITY_ROW"])
            assert read_fd != ledger_fd
            b._validate_inherited_worker_descriptors(read_fd, ledger_fd)
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
                root=root.resolve(),
                own_stage=own,
                other_stage=other,
                ledger_paths=b.WORKER_LEDGER_PATHS,
                repository_fd=(
                    repository_fd := os.open(
                        root,
                        os.O_RDONLY
                        | os.O_DIRECTORY
                        | os.O_NOFOLLOW
                        | os.O_CLOEXEC,
                    )
                ),
                results_fd=(
                    results_fd := os.open(
                        "results",
                        os.O_RDONLY
                        | os.O_DIRECTORY
                        | os.O_NOFOLLOW
                        | os.O_CLOEXEC,
                        dir_fd=repository_fd,
                    )
                ),
                filesystem_root_fd=os.open(
                    "/",
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | os.O_NOFOLLOW
                    | os.O_CLOEXEC,
                ),
                ledger_fd=ledger_fd,
            )
            active = json.loads(
                (root / b.PREREGISTRATION_PATH).read_bytes()
            )
            b.prereg.expected_failed_predecessor_prepublication_closures = (
                lambda: active["bindings"][
                    "failed_predecessor_prepublication_closures"
                ]
            )
            b.prereg.expected_failed_predecessor_pre_sentinel_closures = (
                lambda: active["bindings"][
                    "failed_predecessor_pre_sentinel_closures"
                ]
            )
            snapshot = b._SecureBoundSnapshot(
                root,
                repository_fd=guard.repository_fd,
                filesystem_root_fd=guard.filesystem_root_fd,
                opener=guard._original_os_open,
                register_descriptor=guard.register_snapshot_descriptor,
            )
            prepared = {b.PREREGISTRATION_PATH.as_posix(): True}
            for declared in b._iter_bindings(active):
                path_text = b._binding_path(declared)
                _candidate, repository_relative = b._bound_regular_path(
                    root,
                    path_text,
                )
                prepared[path_text] = repository_relative
            for path_text, repository_relative in sorted(prepared.items()):
                snapshot.open_initial(path_text, repository_relative)
            guard.install()
            metadata = b._authenticate_guarded_worker_metadata(
                root,
                parent,
                claim_binding,
                raw_cache=snapshot,
            )
            consumed = b._consume_worker_capability(read_fd, capability)
            print(json.dumps({
                "binding": metadata["preregistration_binding"],
                "authentication": metadata["authentication"],
                "token_sha256": hashlib.sha256(consumed).hexdigest(),
                "descriptor_count": 2,
                "descriptors_distinct": read_fd != ledger_fd,
                "traps": traps,
                "counters": guard.counters(),
            }, sort_keys=True))
            """
        ),
        encoding="utf-8",
    )
    capability = builder._prepare_worker_capability(
        root=tmp_path,
        output_dir=tmp_path / own,
        slot=1,
        parent_pid=os.getpid(),
    )
    read_fd = int(capability["read_fd"])
    ledger_fd = int(capability["ledger_fd"])
    assert read_fd != ledger_fd
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
            "G9CB_CAPABILITY_FD": str(read_fd),
            "G9CB_LEDGER_FD": str(ledger_fd),
            "G9CB_CAPABILITY_ROW": json.dumps(
                capability["row"], sort_keys=True, separators=(",", ":")
            ),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": builder.REPOSITORY_ROOT.as_posix(),
        }
    )
    process = subprocess.Popen(
        [sys.executable, "-B", str(script)],
        cwd=builder.REPOSITORY_ROOT,
        env=environment,
        pass_fds=(read_fd, ledger_fd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    os.close(read_fd)
    capability["read_fd"] = -1
    try:
        stdout, stderr = process.communicate(timeout=30)
    finally:
        os.close(ledger_fd)
        capability["ledger_fd"] = -1
        builder._zero_token(capability["token"])
    completed = subprocess.CompletedProcess(
        process.args,
        process.returncode,
        stdout,
        stderr,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])
    assert result["binding"] == binding
    assert result["authentication"] == parent_authentication
    assert result["token_sha256"] == capability["row"]["token_sha256"]
    assert result["descriptor_count"] == 2
    assert result["descriptors_distinct"] is True
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
    "g9cb3-authority-decision",
    "g9cb3-preregistration",
    "g9cb3-access-claim",
    "g9cb3-attempt-sentinel",
    "g9cb3-pass1-ledger",
)

HISTORICAL_METADATA_MUTATION_CASES = tuple(
    (case, mutation)
    for case in HISTORICAL_METADATA_CASES
    for mutation in (
        ("mode", "sha256", "derived-blob")
        if case in {
            "g9cb2-authority-decision",
            "g9cb3-authority-decision",
        }
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
    g9cb2_key = {
        "g9cb2-authority-decision": "authority_decision",
        "g9cb2-preregistration": "preregistration",
        "g9cb2-access-claim": "access_claim",
        "g9cb2-attempt-sentinel": "attempt_sentinel",
    }.get(case)
    if g9cb2_key is not None:
        return bindings["failed_predecessor_attempts"][0][g9cb2_key]
    g9cb3 = bindings["failed_predecessor_attempts"][1]
    if case == "g9cb3-attempt-sentinel":
        return g9cb3["terminal_evidence"]["attempt_sentinel"]
    if case == "g9cb3-pass1-ledger":
        return g9cb3["terminal_evidence"]["pass1_worker_ledger"]
    return g9cb3[
        {
            "g9cb3-authority-decision": "authority_decision",
            "g9cb3-preregistration": "preregistration",
            "g9cb3-access-claim": "access_claim",
        }[case]
    ]


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
            0o555 if case.endswith("authority-decision") else 0o644,
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
    own = "results/.gross9-structural-clock-g9cb8-worker-mutation-own"
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
            b.prereg.expected_failed_predecessor_prepublication_closures = (
                lambda: active["bindings"][
                    "failed_predecessor_prepublication_closures"
                ]
            )
            b.prereg.expected_failed_predecessor_pre_sentinel_closures = (
                lambda: active["bindings"][
                    "failed_predecessor_pre_sentinel_closures"
                ]
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
                root=root.resolve(),
                own_stage=(
                    "results/.gross9-structural-clock-g9cb8-"
                    "worker-mutation-own"
                ),
                other_stage=(
                    "results/.gross9-structural-clock-g9cb8-"
                    "worker-mutation-other"
                ),
                ledger_paths=b.WORKER_LEDGER_PATHS,
                repository_fd=(
                    repository_fd := os.open(
                        root,
                        os.O_RDONLY
                        | os.O_DIRECTORY
                        | os.O_NOFOLLOW
                        | os.O_CLOEXEC,
                    )
                ),
                results_fd=(
                    results_fd := os.open(
                        "results",
                        os.O_RDONLY
                        | os.O_DIRECTORY
                        | os.O_NOFOLLOW
                        | os.O_CLOEXEC,
                        dir_fd=repository_fd,
                    )
                ),
                filesystem_root_fd=os.open(
                    "/",
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | os.O_NOFOLLOW
                    | os.O_CLOEXEC,
                ),
                ledger_fd=os.open(
                    ".",
                    os.O_RDWR | os.O_TMPFILE | os.O_CLOEXEC,
                    0o600,
                    dir_fd=results_fd,
                ),
            )
            snapshot = b._SecureBoundSnapshot(
                root,
                repository_fd=guard.repository_fd,
                filesystem_root_fd=guard.filesystem_root_fd,
                opener=guard._original_os_open,
                register_descriptor=guard.register_snapshot_descriptor,
            )
            prepared = {b.PREREGISTRATION_PATH.as_posix(): True}
            for declared in b._iter_bindings(active):
                path_text = b._binding_path(declared)
                _candidate, repository_relative = b._bound_regular_path(
                    root,
                    path_text,
                )
                prepared[path_text] = repository_relative
            for path_text, repository_relative in sorted(prepared.items()):
                snapshot.open_initial(path_text, repository_relative)
            guard.install()
            try:
                b._authenticate_guarded_worker_metadata(
                    root,
                    parent,
                    claim_binding,
                    raw_cache=snapshot,
                )
            except b.TerminalG9CB8Failure as exc:
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
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    preregistration, binding, parent_authentication = (
        _prepare_guarded_metadata_repository(tmp_path, monkeypatch)
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
    with pytest.raises(builder.TerminalG9CB8Failure, match=match):
        builder.validate_preregistration(
            tmp_path,
            validation_mode="guarded_worker",
            parent_authentication=parent_authentication,
            claim_preregistration=binding,
        )
    assert preregistration["identity"] == "G9CB-8"


def test_capability_preparation_orders_pipe_identity_before_mutable_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "results").mkdir()
    events: list[tuple[str, int | None]] = []
    pipe_fds: set[int] = set()
    original_pipe2 = builder.os.pipe2
    original_fstat = builder.os.fstat
    original_fill = builder._fill_random_token
    original_write = builder._write_all

    def recorded_pipe2(flags: int) -> tuple[int, int]:
        pair = original_pipe2(flags)
        pipe_fds.update(pair)
        events.append(("pipe", pair[0]))
        return pair

    def recorded_fstat(descriptor: int) -> os.stat_result:
        if descriptor in pipe_fds:
            events.append(("pipe-fstat", descriptor))
        return original_fstat(descriptor)

    def recorded_fill(token: bytearray) -> None:
        events.append(("fill", None))
        original_fill(token)

    def recorded_write(descriptor: int, raw: bytes | bytearray) -> None:
        if descriptor in pipe_fds:
            events.append(("pipe-write", descriptor))
            assert isinstance(raw, bytearray)
        original_write(descriptor, raw)

    monkeypatch.setattr(builder.os, "pipe2", recorded_pipe2)
    monkeypatch.setattr(builder.os, "fstat", recorded_fstat)
    monkeypatch.setattr(builder, "_fill_random_token", recorded_fill)
    monkeypatch.setattr(builder, "_write_all", recorded_write)
    capability = builder._prepare_worker_capability(
        root=tmp_path,
        output_dir=(
            tmp_path
            / "results/.gross9-structural-clock-g9cb8-worker-order"
        ),
        slot=1,
        parent_pid=os.getpid(),
    )
    try:
        names = [name for name, _descriptor in events]
        assert names.index("pipe") < names.index("pipe-fstat")
        assert names.index("pipe-fstat") < names.index("fill")
        assert names.index("fill") < names.index("pipe-write")
        assert isinstance(capability["token"], bytearray)
        assert capability["read_fd"] != capability["ledger_fd"]
    finally:
        os.close(capability["read_fd"])
        os.close(capability["ledger_fd"])
        builder._zero_token(capability["token"])
        assert capability["token"] == bytearray(32)


def test_worker_invocation_has_exact_environment_and_no_legacy_transport(
    tmp_path: Path,
) -> None:
    (tmp_path / "training").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / builder.BUILDER_PATH).write_text("", encoding="utf-8")
    capability = builder._prepare_worker_capability(
        root=tmp_path,
        output_dir=(
            tmp_path / "results/.gross9-structural-clock-g9cb8-worker-slot1"
        ),
        slot=1,
        parent_pid=os.getpid(),
    )
    try:
        invocation = builder._prepare_worker(
            root=tmp_path,
            capability=capability,
            other_stage_directory=(
                "results/.gross9-structural-clock-g9cb8-worker-slot2"
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
        capability_index = invocation["command"].index(
            "--worker-capability-fd"
        )
        ledger_index = invocation["command"].index("--worker-ledger-fd")
        assert int(invocation["command"][capability_index + 1]) == (
            capability["read_fd"]
        )
        assert int(invocation["command"][ledger_index + 1]) == (
            capability["ledger_fd"]
        )
        assert capability["read_fd"] != capability["ledger_fd"]
        assert "--worker-token" not in joined
        assert "G9CB_WORKER_TOKEN" not in invocation["environment"]
        assert "PYTHONSTARTUP" not in invocation["environment"]
    finally:
        os.close(capability["read_fd"])
        os.close(capability["ledger_fd"])
        builder._zero_token(capability["token"])


def test_stale_run_worker_wrapper_is_absent() -> None:
    assert "_run_worker" not in vars(builder)


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
    monkeypatch.setattr(
        builder,
        "_validate_inherited_worker_descriptors",
        lambda capability_fd, ledger_fd: events.append(
            ("validate-fds", capability_fd, ledger_fd)
        ),
    )
    monkeypatch.setattr(builder, "_WorkerIsolationGuard", FakeGuard)
    (tmp_path / "results").mkdir()
    monkeypatch.chdir(tmp_path)
    arguments = [
        "--internal-worker",
        "--repository-root",
        str(tmp_path),
        "--output-dir",
        "results/.gross9-structural-clock-g9cb8-worker-one",
        "--other-stage-directory",
        "results/.gross9-structural-clock-g9cb8-worker-two",
        "--worker-capability-fd",
        "7",
        "--worker-ledger-fd",
        "9",
        "--expected-parent-pid",
        "1234",
    ]
    guard = builder._early_worker_bootstrap(arguments)
    assert isinstance(guard, FakeGuard)
    assert events[0] == ("pdeath", 1234)
    assert events[1] == ("validate-fds", 7, 9)
    guard_event = events[2]
    assert isinstance(guard_event, tuple)
    assert guard_event[0] == "guard"
    guard_arguments = guard_event[1]
    assert isinstance(guard_arguments, dict)
    assert guard_arguments["ledger_fd"] == 9
    assert events[3] == "install"

    main_events: list[str] = []

    class FakeParser:
        def parse_args(self, _arguments: list[str]) -> SimpleNamespace:
            main_events.append("parse")
            return SimpleNamespace(
                internal_worker=False,
                output_dir=None,
                other_stage_directory=None,
                worker_capability_fd=None,
                worker_ledger_fd=None,
                expected_parent_pid=None,
                synthetic_input=None,
                create_claim=True,
                verify_publication=False,
                repository_root=tmp_path.as_posix(),
            )

    monkeypatch.setattr(
        builder,
        "_early_worker_bootstrap",
        lambda _arguments: main_events.append("bootstrap"),
    )
    monkeypatch.setattr(builder, "_parser", lambda: FakeParser())
    monkeypatch.setattr(
        builder,
        "create_claim_only",
        lambda _root: main_events.append("claim"),
    )
    assert builder.main(["--create-claim"]) == 0
    assert main_events == ["bootstrap", "parse", "claim"]


def test_worker_popen_passes_two_bound_descriptors_and_closes_pipe_before_wait(
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
    capability = {"read_fd": 17, "ledger_fd": 19}
    pid = builder._execute_prepared_worker(
        {
            "command": ["python", "-B", "builder.py", "--internal-worker"],
            "cwd": Path("/tmp"),
            "environment": {"ONLY": "BOUND"},
            "capability": capability,
        }
    )
    assert pid == 4321
    popen_event = events[0]
    assert isinstance(popen_event, tuple)
    assert popen_event[-2:] == (True, (17, 19))
    assert events[1:] == [("close", 17), "wait"]
    assert capability["read_fd"] == -1
    assert capability["ledger_fd"] == 19


@pytest.mark.parametrize(
    "capability",
    [
        {"read_fd": 17},
        {"read_fd": 17, "ledger_fd": 17},
    ],
    ids=["ledgerless", "aliased"],
)
def test_worker_execution_rejects_non_two_descriptor_invocation_before_spawn(
    monkeypatch: pytest.MonkeyPatch,
    capability: dict[str, int],
) -> None:
    monkeypatch.setattr(
        builder.subprocess,
        "Popen",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("invalid descriptor handoff reached process spawn")
        ),
    )
    with pytest.raises(
        builder.TerminalG9CB8Failure,
        match="ledger descriptor is absent|not distinct",
    ):
        builder._execute_prepared_worker(
            {
                "command": [
                    "python",
                    "-B",
                    "builder.py",
                    "--internal-worker",
                ],
                "cwd": Path("/tmp"),
                "environment": {"ONLY": "BOUND"},
                "capability": capability,
            }
        )


def test_worker_inherited_fd_table_rejects_real_extra_fd_above_65535(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    script = textwrap.dedent(
        """
        import fcntl
        import os
        from pathlib import Path
        import resource
        import sys

        from training import build_gross9_structural_clock_bundle as builder

        results = Path(sys.argv[1])
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        required = 70001
        if soft < required:
            if hard != resource.RLIM_INFINITY and hard < required:
                raise SystemExit(90)
            resource.setrlimit(resource.RLIMIT_NOFILE, (required, hard))
        capability_fd, write_fd = os.pipe2(os.O_CLOEXEC)
        os.close(write_fd)
        ledger_fd = os.open(
            results,
            os.O_RDWR | os.O_TMPFILE | os.O_CLOEXEC,
            0o600,
        )
        source_fd = os.open(os.devnull, os.O_RDONLY | os.O_CLOEXEC)
        extra_fd = fcntl.fcntl(
            source_fd,
            fcntl.F_DUPFD_CLOEXEC,
            70000,
        )
        os.close(source_fd)
        assert extra_fd >= 70000
        try:
            builder._validate_inherited_worker_descriptors(
                capability_fd,
                ledger_fd,
            )
        except builder.TerminalG9CB8Failure as exc:
            if "descriptor table" not in str(exc):
                raise
            raise SystemExit(0)
        raise SystemExit(91)
        """
    )
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": builder.REPOSITORY_ROOT.as_posix(),
        }
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script, results.as_posix()],
        cwd=builder.REPOSITORY_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode != 90, "RLIMIT_NOFILE cannot represent FD 70000"
    assert completed.returncode == 0, (
        f"extra FD 70000 was accepted; rc={completed.returncode}; "
        f"stderr={completed.stderr}"
    )


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
        builder.TerminalG9CB8Failure,
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
    assert result["error_type"] == "TerminalG9CB8Failure"


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
                root=root.resolve(),
                own_stage=(
                    "results/.gross9-structural-clock-g9cb8-worker-import-own-"
                    + suffix
                ),
                other_stage=(
                    "results/.gross9-structural-clock-g9cb8-worker-import-other-"
                    + suffix
                ),
                ledger_paths=b.WORKER_LEDGER_PATHS,
                repository_fd=(
                    repository_fd := os.open(
                        root,
                        os.O_RDONLY
                        | os.O_DIRECTORY
                        | os.O_NOFOLLOW
                        | os.O_CLOEXEC,
                    )
                ),
                results_fd=(
                    results_fd := os.open(
                        "results",
                        os.O_RDONLY
                        | os.O_DIRECTORY
                        | os.O_NOFOLLOW
                        | os.O_CLOEXEC,
                        dir_fd=repository_fd,
                    )
                ),
                filesystem_root_fd=os.open(
                    "/",
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | os.O_NOFOLLOW
                    | os.O_CLOEXEC,
                ),
                ledger_fd=os.open(
                    ".",
                    os.O_RDWR | os.O_TMPFILE | os.O_CLOEXEC,
                    0o600,
                    dir_fd=results_fd,
                ),
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
                / "results/.g9cb8-bytecode-cache-disabled"
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
        repository_root / "results/.g9cb8-bytecode-cache-disabled"
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
        "descriptor_count": 2,
        "descriptors_distinct": True,
    }
    assert _process_is_dead_or_zombie(worker_pid)


def test_guard_rejects_cross_stage_observation(tmp_path: Path) -> None:
    guard = _guard(tmp_path)
    with pytest.raises(
        builder.TerminalG9CB8Failure,
        match="other worker stage",
    ):
        guard._checked_path(guard.other_stage / "probe")
    assert guard.other_stage_access_events == 1


def test_guard_rejects_path_open_of_fifo(tmp_path: Path) -> None:
    guard = _guard(tmp_path)
    fifo = tmp_path / "named.fifo"
    os.mkfifo(fifo)
    with pytest.raises(
        builder.TerminalG9CB8Failure,
        match="FIFO",
    ):
        guard._checked_path(fifo, fifo_open=True)


@pytest.mark.parametrize("keyword", ["dir_fd", "src_dir_fd", "dst_dir_fd"])
def test_guard_rejects_every_dir_fd_variant(keyword: str) -> None:
    with pytest.raises(
        builder.TerminalG9CB8Failure,
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
    chain = _prepare_synthetic_a5_q5_p5_chain(tmp_path)
    root = chain["root"]
    preregistration = chain["preregistration"]
    preregistration_binding = chain["preregistration_binding"]
    security_profile = _synthetic_security_profile(chain)
    binding = builder.create_claim_only(
        root,
        expected_security_profile=security_profile,
    )
    claim = _read_synthetic_claim(chain)
    assert binding == _synthetic_claim_binding(chain, claim)
    c5 = _commit_synthetic_claim(chain, claim)
    security_profile["claim_commit"] = c5
    claim_raw = (root / builder.CLAIM_PATH).read_bytes()
    claim_binding = {
        "path": builder.CLAIM_PATH.as_posix(),
        "sha256": hashlib.sha256(claim_raw).hexdigest(),
        "claim_hash": claim["claim_hash"],
        "protocol_parent_commit": chain["p5"],
        "claim_commit": c5,
    }
    synthetic_input = _write_synthetic_chain_input(chain)
    lifecycle: list[tuple[str, str]] = []
    checkpoints: list[str] = []
    worker_pids: list[int] = []
    original_create = builder._create_stage_directory
    original_cleanup = builder._cleanup_successful_stage
    original_namespace = builder._validate_production_namespace
    original_execute = builder._execute_prepared_worker

    def recorded_create(
        stage: Path,
        results: Path,
        **kwargs: Any,
    ) -> None:
        lifecycle.append(("create", stage.name))
        original_create(stage, results, **kwargs)

    def recorded_cleanup(
        stage: Path,
        results: Path,
        **kwargs: Any,
    ) -> None:
        lifecycle.append(("cleanup", stage.name))
        original_cleanup(stage, results, **kwargs)

    def recorded_namespace(
        context: builder._PublicationContext,
        checkpoint: str,
        stage_one: Path,
        stage_two: Path,
    ) -> None:
        original_namespace(context, checkpoint, stage_one, stage_two)
        checkpoints.append(checkpoint)

    def recorded_execute(invocation: dict[str, Any]) -> int:
        pid = original_execute(invocation)
        worker_pids.append(pid)
        return pid

    monkeypatch.setattr(builder, "_create_stage_directory", recorded_create)
    monkeypatch.setattr(builder, "_cleanup_successful_stage", recorded_cleanup)
    monkeypatch.setattr(
        builder,
        "_validate_production_namespace",
        recorded_namespace,
    )
    monkeypatch.setattr(
        builder,
        "_execute_prepared_worker",
        recorded_execute,
    )

    result = builder.produce_one_shot(
        root,
        synthetic_input=synthetic_input,
        expected_security_profile=security_profile,
    )

    publication_paths = (
        builder.SENTINEL_PATH,
        *builder.WORKER_LEDGER_PATHS,
        builder.CSV_PATH,
        builder.MANIFEST_PATH,
    )
    assert result["identity"] == builder.IDENTITY
    assert [path.as_posix() for path in publication_paths] == [
        "results/gross9_structural_clock_bundle_g9cb8_attempt_consumed_2026-07-31.json",
        (
            "results/"
            "gross9_structural_clock_bundle_g9cb8_"
            "worker_capability_consumed_pass1_2026-07-31.json"
        ),
        (
            "results/"
            "gross9_structural_clock_bundle_g9cb8_"
            "worker_capability_consumed_pass2_2026-07-31.json"
        ),
        "results/gross9_structural_clock_bundle_g9cb8_2026-07-31.csv.gz",
        "results/gross9_structural_clock_bundle_g9cb8_manifest_2026-07-31.json",
    ]
    assert all((root / path).is_file() for path in publication_paths)
    assert checkpoints == list(builder.PRODUCTION_CHECKPOINTS)
    assert len(worker_pids) == 2
    assert worker_pids[0] != worker_pids[1]
    assert lifecycle[0][0] == "create"
    assert lifecycle[1][0] == "cleanup"
    assert lifecycle[2][0] == "create"
    assert lifecycle[0][1] == lifecycle[1][1]
    assert lifecycle[2][1] != lifecycle[0][1]
    assert lifecycle[-1] == ("cleanup", lifecycle[2][1])

    sentinel = json.loads((root / builder.SENTINEL_PATH).read_bytes())
    manifest = json.loads((root / builder.MANIFEST_PATH).read_bytes())
    parent_authentication = manifest["parent_authentication"]
    assert set(parent_authentication) == {
        "environment",
        "hashed_inputs",
        "preregistration_authentication",
        "runtime_import_closure",
    }
    assert [row["slot"] for row in sentinel["worker_capabilities"]] == [1, 2]
    assert all(
        set(row) == set(CAPABILITY_KEYS) and len(row) == 14
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
    csv_raw = (root / builder.CSV_PATH).read_bytes()
    rows = builder.validate_csv_gzip(
        csv_raw,
        require_all_sleeves=True,
    )
    sentinel_raw = (root / builder.SENTINEL_PATH).read_bytes()
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
        ledger_path = root / path
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
    amendments = claim["authority_amendments"]
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
        builder.TerminalG9CB8Failure,
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
        builder.TerminalG9CB8Failure,
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
        builder.TerminalG9CB8Failure,
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
        (root / "results").glob(
            ".gross9-structural-clock-g9cb8-worker-*"
        )
    )
    assert not list(
        (root / "results").glob(
            "gross9_structural_clock_bundle_pass_receipt.json"
        )
    )
    assert not list(root.rglob("*.pyc"))
    assert not list(root.rglob("__pycache__"))


def test_parent_advances_ledger_linked_while_worker_alive_and_stage_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    chain = _prepare_synthetic_a5_q5_p5_chain(tmp_path)
    security_profile = _synthetic_security_profile(chain)
    binding = builder.create_claim_only(
        chain["root"],
        expected_security_profile=security_profile,
    )
    claim = _read_synthetic_claim(chain)
    assert binding == _synthetic_claim_binding(chain, claim)
    claim_commit = _commit_synthetic_claim(chain, claim)
    security_profile["claim_commit"] = claim_commit
    root = chain["root"]
    worker_probe = tmp_path / "worker_checkpoint_probe.py"
    worker_probe.write_text(
        textwrap.dedent(
            """
            import sys
            import time

            from training import build_gross9_structural_clock_bundle as b

            checkpoint_seen = False
            original_checkpoint = b._worker_ledger_linked_checkpoint
            original_synthetic_read = b._read_synthetic_worker_input

            def recorded_checkpoint(*args, **kwargs):
                global checkpoint_seen
                result = original_checkpoint(*args, **kwargs)
                checkpoint_seen = True
                time.sleep(0.25)
                return result

            def guarded_synthetic_read(*args, **kwargs):
                if not checkpoint_seen:
                    raise b.TerminalG9CB8Failure(
                        "runtime/value access preceded worker ledger checkpoint"
                    )
                return original_synthetic_read(*args, **kwargs)

            b._worker_ledger_linked_checkpoint = recorded_checkpoint
            b._read_synthetic_worker_input = guarded_synthetic_read
            raise SystemExit(b.main(sys.argv[1:]))
            """
        ),
        encoding="utf-8",
    )
    original_prepare = builder._prepare_worker
    original_popen = builder.subprocess.Popen
    original_advance = builder._ProductionStateMachine.advance
    workers: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []

    def probed_prepare(*args: Any, **kwargs: Any) -> dict[str, Any]:
        invocation = original_prepare(*args, **kwargs)
        command = list(invocation["command"])
        command[2] = worker_probe.as_posix()
        invocation["command"] = command
        return invocation

    def recorded_popen(
        command: Any,
        *args: Any,
        **kwargs: Any,
    ) -> subprocess.Popen[Any]:
        process: subprocess.Popen[Any] = original_popen(
            command,
            *args,
            **kwargs,
        )
        command_parts = [str(part) for part in command]
        if "--internal-worker" in command_parts:
            stage_argument = Path(
                command_parts[command_parts.index("--output-dir") + 1]
            )
            stage = (
                stage_argument
                if stage_argument.is_absolute()
                else root / stage_argument
            )
            workers.append(
                {
                    "command": command_parts,
                    "pass_fds": tuple(kwargs.get("pass_fds", ())),
                    "process": process,
                    "stage": stage,
                    "stdio": tuple(
                        kwargs.get(name) for name in ("stdin", "stdout", "stderr")
                    ),
                }
            )
        return process

    def recorded_advance(
        state: builder._ProductionStateMachine,
        checkpoint: str,
        validator: Any,
    ) -> None:
        slot_by_checkpoint = {
            "PASS1_LEDGER_LINKED": 0,
            "PASS1_OUTPUT_READY": 0,
            "PASS2_LEDGER_LINKED": 1,
            "PASS2_OUTPUT_READY": 1,
        }
        if checkpoint in slot_by_checkpoint:
            worker = workers[slot_by_checkpoint[checkpoint]]
            checkpoints.append(
                {
                    "checkpoint": checkpoint,
                    "stage_entries": tuple(
                        sorted(path.name for path in worker["stage"].iterdir())
                    ),
                    "worker_alive": worker["process"].poll() is None,
                }
            )
        original_advance(state, checkpoint, validator)

    monkeypatch.setattr(builder, "_prepare_worker", probed_prepare)
    monkeypatch.setattr(builder.subprocess, "Popen", recorded_popen)
    monkeypatch.setattr(
        builder._ProductionStateMachine,
        "advance",
        recorded_advance,
    )
    result = builder.produce_one_shot(
        root,
        synthetic_input=_write_synthetic_chain_input(chain),
        expected_security_profile=security_profile,
    )
    assert result["identity"] == builder.IDENTITY
    assert [row["checkpoint"] for row in checkpoints] == [
        "PASS1_LEDGER_LINKED",
        "PASS1_OUTPUT_READY",
        "PASS2_LEDGER_LINKED",
        "PASS2_OUTPUT_READY",
    ]
    staged_outputs = tuple(
        sorted(
            (
                builder._STAGED_CORE_NAME,
                builder._STAGED_CSV_NAME,
                builder._STAGED_RECEIPT_NAME,
            )
        )
    )
    assert [
        (row["worker_alive"], row["stage_entries"])
        for row in checkpoints[::2]
    ] == [(True, ()), (True, ())]
    for output_checkpoint in checkpoints[1::2]:
        assert output_checkpoint["stage_entries"] == staged_outputs
    assert len(workers) == 2
    for worker in workers:
        command = worker["command"]
        capability_fd = int(
            command[command.index("--worker-capability-fd") + 1]
        )
        ledger_fd = int(command[command.index("--worker-ledger-fd") + 1])
        assert capability_fd != ledger_fd
        assert worker["pass_fds"] == (capability_fd, ledger_fd)
        assert worker["stdio"] == (None, None, None)
        assert [part for part in command if part.endswith("-fd")] == [
            "--worker-capability-fd",
            "--worker-ledger-fd",
        ]


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
    assert isinstance(bindings, Mapping)
    assert "adapter_import_roots" not in bindings
    assert "adapter_import_closure" not in bindings


def test_rank7_signal_counter_increments_at_each_evaluated_decision() -> None:
    bars = _bars(4)
    bars[0]["decisions"] = {
        "frozen_annual_rank7": {"active": False}
    }
    bars[1]["decisions"] = {
        "frozen_annual_rank7": {"active": False}
    }
    bars[2]["decisions"] = {
        "frozen_annual_rank7": {"active": "not-a-boolean"}
    }
    counters = builder._empty_counters()
    with pytest.raises(
        builder.TerminalG9CB8Failure,
        match="nonboolean active flag",
    ):
        builder.reconstruct_intervals(
            bars,
            counters=counters,
            domain_end=_time(len(bars)),
        )
    assert counters["per_sleeve"]["frozen_annual_rank7"][
        "signal_rows_evaluated"
    ] == 3


def _git_for_q5_test(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _synthetic_head_binding_repository(root: Path) -> tuple[str, bytes]:
    root.mkdir()
    _git_for_q5_test(root, "init", "-q")
    _git_for_q5_test(root, "config", "user.email", "q5@example.invalid")
    _git_for_q5_test(root, "config", "user.name", "Q8 Synthetic")
    relative = "protocol/binding.txt"
    candidate = root / relative
    candidate.parent.mkdir()
    raw = b"synthetic-q5-binding\n"
    candidate.write_bytes(raw)
    _git_for_q5_test(root, "add", relative)
    _git_for_q5_test(root, "commit", "-qm", "synthetic binding")
    return relative, raw


def _prepare_synthetic_a5_q5_p5_chain(
    base: Path,
) -> dict[str, Any]:
    root = base / "repo"
    remote = base / "remote.git"
    root.mkdir()
    subprocess.run(
        ["git", "init", "--bare", "-q", str(remote)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _git_for_q5_test(
        root,
        "init",
        "-q",
        "-b",
        builder.prereg.EXPECTED_BRANCH,
    )
    _git_for_q5_test(root, "config", "user.email", "q5@example.invalid")
    _git_for_q5_test(root, "config", "user.name", "Q8 Synthetic")
    for relative in sorted(
        builder.prereg.PROTOCOL_PATHS,
        key=lambda path: path.as_posix(),
    ):
        candidate = root / relative
        candidate.parent.mkdir(parents=True, exist_ok=True)
        if relative == Path("training/__init__.py"):
            raw = b""
        elif relative in builder.prereg.RUNTIME_IMPORT_ROOTS:
            raw = b"SYNTHETIC_RUNTIME_VALUE = 1\n"
        else:
            raw = f"synthetic A5 bytes: {relative.as_posix()}\n".encode()
        candidate.write_bytes(raw)
    results = root / "results"
    results.mkdir()
    (root / ".gitignore").write_text(
        "\n".join(
            [
                "data/",
                "fixtures/",
                "results/gross9_structural_clock_bundle_g9cb8_*",
                "results/.gross9-structural-clock-g9cb8-worker-*",
                "results/.g9cb8-bytecode-cache-disabled",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _git_for_q5_test(root, "add", ".")
    _git_for_q5_test(root, "commit", "-qm", "synthetic A5")
    a5 = _git_for_q5_test(root, "rev-parse", "HEAD")
    for entry in builder.prereg.SUCCESSOR_PROTOCOL_DIFF:
        status, path_text = entry.split("\t", 1)
        assert status == "M"
        source = builder.REPOSITORY_ROOT / path_text
        assert source.is_file()
        shutil.copy2(
            source,
            root / path_text,
        )
    _git_for_q5_test(root, "add", ".")
    _git_for_q5_test(root, "commit", "-qm", "synthetic Q8")
    q5 = _git_for_q5_test(root, "rev-parse", "HEAD")
    _git_for_q5_test(root, "remote", "add", "origin", str(remote))
    _git_for_q5_test(
        root,
        "push",
        "-qu",
        "origin",
        builder.prereg.EXPECTED_BRANCH,
    )

    materialized_path = Path("data/synthetic-g9cb6-materialized.bin")
    materialized_raw = b"synthetic opaque G9CB-6 materialization\n"
    (root / materialized_path).parent.mkdir(parents=True, exist_ok=True)
    (root / materialized_path).write_bytes(materialized_raw)
    (root / materialized_path).chmod(0o444)

    for row in builder.prereg.expected_failed_predecessor_attempts():
        slot_one = root / row["residue"]["slot1_stage"]["path"]
        slot_one.mkdir(mode=0o700)
        slot_one.chmod(0o700)

    protocol_bindings = [
        builder._head_blob_binding(root, path.as_posix())
        for path in sorted(
            builder.prereg.PROTOCOL_PATHS,
            key=lambda item: item.as_posix(),
        )
    ]
    protocol_snapshot_bindings = [
        {
            "path": row["path"],
            "path_type": "regular_file",
            "sha256": row["sha256"],
            "git_blob": row["git_blob"],
            "git_mode": row["mode"],
        }
        for row in protocol_bindings
    ]
    protocol_by_path = {
        row["path"]: row for row in protocol_snapshot_bindings
    }
    amendment_paths = (
        builder.prereg.RANK7_AUTHORITY_AMENDMENT_PATH,
        builder.prereg.RUNTIME_ISOLATION_AMENDMENT_PATH,
        builder.prereg.PREREGISTRATION_CORRECTION_AMENDMENT_PATH,
    )
    authority_amendments = [
        {
            "identity": identity,
            **protocol_by_path[path.as_posix()],
            "authority_commit": a5,
        }
        for identity, path in zip(
            ("G9CB-1A", "G9CB-1B", "G9CB-1C"),
            amendment_paths,
            strict=True,
        )
    ]
    predecessor_binding = protocol_by_path[
        builder.prereg.PROTOCOL_PATHS[2].as_posix()
    ]
    predecessor_attempts = {
        row["identity"]: row
        for row in builder.prereg.expected_failed_predecessor_attempts()
    }
    failed_attempts: list[dict[str, Any]] = []
    for identity in ("G9CB-2", "G9CB-3"):
        prefix = identity.lower().replace("-", "")
        expected_residue = predecessor_attempts[identity]["residue"]
        slot_one = expected_residue["slot1_stage"]["path"]
        failed_attempts.append(
            {
                "identity": identity,
                "authority_decision": dict(predecessor_binding),
                "preregistration": dict(predecessor_binding),
                "access_claim": dict(predecessor_binding),
                "attempt_sentinel": dict(predecessor_binding),
                "permanently_absent_outputs": [
                    f"results/synthetic-{prefix}-reserved.json"
                ],
                "residue": {
                    "slot1_stage": {
                        "path": slot_one,
                        "state": "empty_directory",
                        "filesystem_mode_octal": "0700",
                        "committed": False,
                    },
                    "slot2_stage": {
                        "path": expected_residue["slot2_stage"]["path"],
                        "state": "absent",
                        "committed": False,
                    },
                },
            }
        )
    failed_closure = {
        "identity": "G9CB-4",
        "authority_decision": dict(predecessor_binding),
        "preregistration": dict(predecessor_binding),
        "permanently_absent_outputs": [
            "results/synthetic-g9cb4-reserved.json"
        ],
        "residue": {
            "bytecode_cache": {
                "path": "results/.synthetic-g9cb4-pycache",
                "state": "absent",
            },
            "publication_stages": {
                "glob": "results/.synthetic-g9cb4-publish-*",
                "state": "absent",
            },
            "worker_stages": {
                "glob": "results/.synthetic-g9cb4-worker-*",
                "state": "absent",
            },
        },
    }
    g9cb5_prepublication_closure = {
        "identity": "G9CB-5",
        "authority_decision": dict(predecessor_binding),
        "permanently_absent_outputs": [
            "results/synthetic-g9cb5-reserved.json"
        ],
        "residue": {
            "bytecode_cache": {
                "path": "results/.synthetic-g9cb5-pycache",
                "state": "absent",
            },
            "publication_stages": {
                "glob": "results/.synthetic-g9cb5-publish-*",
                "state": "absent",
            },
            "worker_stages": {
                "glob": "results/.synthetic-g9cb5-worker-*",
                "state": "absent",
            },
        },
    }
    g9cb6_prepublication_closure = {
        "identity": "G9CB-6",
        "authority_decision": dict(predecessor_binding),
        "input_materialization": {
            "destination": {
                "path": materialized_path.as_posix(),
                "path_type": "regular_file",
                "sha256": hashlib.sha256(materialized_raw).hexdigest(),
                "size_bytes": len(materialized_raw),
                "git_blob": None,
                "git_mode": None,
                "mode_octal": "0444",
            }
        },
        "permanently_absent_outputs": [
            "results/synthetic-g9cb6-reserved.json"
        ],
        "residue": {
            "bytecode_cache": {
                "path": "results/.synthetic-g9cb6-pycache",
                "state": "absent",
            },
            "capability_probes": {
                "glob": "results/.synthetic-g9cb6-probe-*",
                "state": "absent",
            },
            "publication_stages": {
                "glob": "results/.synthetic-g9cb6-publish-*",
                "state": "absent",
            },
            "worker_stages": {
                "glob": "results/.synthetic-g9cb6-worker-*",
                "state": "absent",
            },
        },
    }
    g9cb7_pre_sentinel_closure = {
        "identity": "G9CB-7",
        "authority_decision": dict(predecessor_binding),
        "preregistration": dict(predecessor_binding),
        "access_claim": dict(predecessor_binding),
        "bytecode_incident": {"directories": [], "files": []},
        "permanently_absent_outputs": [
            "results/synthetic-g9cb7-reserved.json"
        ],
        "residue": {
            "bytecode_cache": {
                "path": "results/.synthetic-g9cb7-pycache",
                "state": "absent",
            },
            "capability_probes": {
                "glob": "results/.synthetic-g9cb7-probe-*",
                "state": "absent",
            },
            "publication_stages": {
                "glob": "results/.synthetic-g9cb7-publish-*",
                "state": "absent",
            },
            "worker_stages": {
                "glob": "results/.synthetic-g9cb7-worker-*",
                "state": "absent",
            },
        },
    }
    runtime_paths = [
        f"{module.replace('.', '/')}.py"
        for module in builder.RUNTIME_IMPORT_MODULES
    ]
    runtime_closure = [
        {
            **protocol_by_path[path],
            "package_initializer": False,
        }
        for path in sorted(runtime_paths)
    ]
    environment = builder._environment_record()
    environment["worker_process_environment"] = (
        builder.prereg.worker_process_environment(root)
    )
    preregistration = {
        "protocol_version": builder.prereg.PROTOCOL_VERSION,
        "identity": builder.IDENTITY,
        "protocol_implementation_commit": q5,
        "git_seal": {
            "expected_branch": builder.prereg.EXPECTED_BRANCH,
        },
        "bindings": {
            "protocol": protocol_snapshot_bindings,
            "authority_amendments": authority_amendments,
            "failed_predecessor_preregistrations": [],
            "failed_predecessor_attempts": failed_attempts,
            "failed_predecessor_closures": [failed_closure],
            "failed_predecessor_prepublication_closures": [
                g9cb5_prepublication_closure,
                g9cb6_prepublication_closure,
            ],
            "failed_predecessor_pre_sentinel_closures": [
                g9cb7_pre_sentinel_closure,
            ],
            "successor_preregistrations": [
                {
                    "identity": identity,
                    "preregistration": dict(predecessor_binding),
                }
                for identity in ("G9CB-2", "G9CB-3", "G9CB-4")
            ],
            "direct_authority": [],
            "config_metadata_evidence": [],
            "runtime_import_roots": runtime_paths,
            "runtime_import_closure": runtime_closure,
            "rank7_bundle": {"declared_files": []},
            "source_manifest_ordered_inventory": [],
            "environment": environment,
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
        "candidate_independence": {
            "candidate_identity_present": False,
            "candidate_artifacts_opened": False,
            "comparator_clock_rows_opened": 0,
            "comparator_clocks_preseen_by_research_program": True,
        },
    }
    preregistration["manifest_hash"] = builder._object_hash(
        preregistration,
        "manifest_hash",
    )
    preregistration_raw = builder._canonical_json_bytes(preregistration)
    preregistration_path = root / builder.PREREGISTRATION_PATH
    preregistration_path.write_bytes(preregistration_raw)
    _git_for_q5_test(
        root,
        "add",
        "-f",
        builder.PREREGISTRATION_PATH.as_posix(),
    )
    _git_for_q5_test(root, "commit", "-qm", "synthetic P8")
    p5 = _git_for_q5_test(root, "rev-parse", "HEAD")
    _git_for_q5_test(root, "push", "-q")
    preregistration_path.chmod(0o444)
    preregistration_binding = {
        "path": builder.PREREGISTRATION_PATH.as_posix(),
        "sha256": hashlib.sha256(preregistration_raw).hexdigest(),
        "manifest_hash": preregistration["manifest_hash"],
    }
    declared_residue_paths = {
        row["residue"]["slot1_stage"]["path"]
        for row in preregistration["bindings"]["failed_predecessor_attempts"]
    }
    materialized_residue_paths = {
        path.relative_to(root).as_posix()
        for path in results.iterdir()
        if path.is_dir()
    }
    assert materialized_residue_paths == declared_residue_paths
    assert not (results / ".synthetic-baseline").exists()
    return {
        "root": root,
        "a5": a5,
        "q5": q5,
        "p5": p5,
        "preregistration": preregistration,
        "preregistration_raw": preregistration_raw,
        "preregistration_binding": preregistration_binding,
        "protocol_bindings": protocol_bindings,
    }


def _synthetic_security_profile(
    chain: Mapping[str, Any],
    *,
    claim_commit: str | None = None,
    publication_commit: str | None = None,
) -> dict[str, Any]:
    bindings = chain["preregistration"]["bindings"]
    profile: dict[str, Any] = {
        "identity": builder.IDENTITY,
        "expected_branch": builder.prereg.EXPECTED_BRANCH,
        "authority_commit": chain["a5"],
        "protocol_implementation_commit": chain["q5"],
        "preregistration_seal_commit": chain["p5"],
        "failed_predecessor_preregistrations": copy.deepcopy(
            bindings["failed_predecessor_preregistrations"]
        ),
        "failed_predecessor_attempts": copy.deepcopy(
            bindings["failed_predecessor_attempts"]
        ),
        "failed_predecessor_closures": copy.deepcopy(
            bindings["failed_predecessor_closures"]
        ),
        "failed_predecessor_prepublication_closures": copy.deepcopy(
            bindings["failed_predecessor_prepublication_closures"]
        ),
        "failed_predecessor_pre_sentinel_closures": copy.deepcopy(
            bindings["failed_predecessor_pre_sentinel_closures"]
        ),
        "successor_preregistrations": copy.deepcopy(
            bindings["successor_preregistrations"]
        ),
        "authority_amendments": copy.deepcopy(
            bindings["authority_amendments"]
        ),
        "protocol_paths": sorted(
            path.as_posix() for path in builder.prereg.PROTOCOL_PATHS
        ),
        "protocol_diff": list(builder.prereg.SUCCESSOR_PROTOCOL_DIFF),
        "preregistration_diff": list(builder.ACTIVE_PREREGISTRATION_DIFF),
        "claim_diff": list(builder.CLAIM_DIFF),
        "publication_diff": list(builder.PUBLICATION_DIFF),
    }
    if claim_commit is not None:
        profile["claim_commit"] = claim_commit
    if publication_commit is not None:
        profile["publication_commit"] = publication_commit
    return profile


def _read_synthetic_claim(chain: Mapping[str, Any]) -> dict[str, Any]:
    raw = (Path(chain["root"]) / builder.CLAIM_PATH).read_bytes()
    payload = json.loads(raw)
    assert raw == builder._canonical_json_bytes(payload)
    return payload


def _synthetic_claim_binding(
    chain: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    raw = (Path(chain["root"]) / builder.CLAIM_PATH).read_bytes()
    return {
        "path": builder.CLAIM_PATH.as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "claim_hash": payload["claim_hash"],
        "protocol_parent_commit": chain["p5"],
    }


def _publish_synthetic_claim_from_p5(
    chain: dict[str, Any],
    *,
    events: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = chain["root"]
    preregistration = chain["preregistration"]
    snapshot, pairs = builder._preauthenticate_parent_snapshot(
        root,
        preregistration,
        content_mode=builder._PREREGISTRATION_ONLY,
    )
    context = builder._PublicationContext(root)
    try:
        builder._validate_closed_entry_phase(
            context,
            builder.P8_CLAIM_PREFLIGHT,
        )
        context.probe()
        snapshot.rebaseline_directory_timestamps(
            matching_identity=context.results_token
        )
        payload = builder._claim_payload(
            chain["p5"],
            chain["preregistration_binding"],
            preregistration["bindings"]["authority_amendments"],
            chain["protocol_bindings"],
            [],
        )
        raw = builder._canonical_json_bytes(payload)

        def final_recheck() -> None:
            if events is not None:
                events.append("recheck")
            builder._final_parent_snapshot_recheck(
                root,
                snapshot,
                pairs,
                context,
                builder.P8_CLAIM_PREFLIGHT,
                builder.prereg.EXPECTED_BRANCH,
            )

        if events is not None:
            events.append("prepare")
        publication = context.publish(
            builder.CLAIM_PATH,
            raw,
            prelink_recheck=final_recheck,
        )
        if events is not None:
            events.append("linked")
        return payload, publication
    finally:
        snapshot.close()
        context.close()


def _commit_synthetic_claim(
    chain: dict[str, Any],
    claim: Mapping[str, Any],
) -> str:
    root = chain["root"]
    path = root / builder.CLAIM_PATH
    _git_for_q5_test(root, "add", "-f", builder.CLAIM_PATH.as_posix())
    _git_for_q5_test(root, "commit", "-qm", "synthetic C8")
    c5 = _git_for_q5_test(root, "rev-parse", "HEAD")
    _git_for_q5_test(root, "push", "-q")
    path.chmod(0o444)
    assert json.loads(path.read_bytes()) == claim
    return c5


def _write_synthetic_chain_input(chain: Mapping[str, Any]) -> Path:
    root = Path(chain["root"])
    synthetic_input = root / "fixtures/structural-bars.json"
    synthetic_input.parent.mkdir(parents=True, exist_ok=True)
    bars = _all_sleeve_bars()
    synthetic_input.write_bytes(
        builder._canonical_json_bytes(
            {
                "bars": bars,
                "domain_end": _time(len(bars)),
            }
        )
    )
    return synthetic_input


def _commit_synthetic_d5(chain: dict[str, Any]) -> str:
    root = chain["root"]
    publications = (
        builder.SENTINEL_PATH,
        *builder.WORKER_LEDGER_PATHS,
        builder.CSV_PATH,
        builder.MANIFEST_PATH,
    )
    for relative in publications:
        assert (root / relative).is_file()
        assert stat.S_IMODE((root / relative).stat().st_mode) == 0o444
        _git_for_q5_test(root, "add", "-f", relative.as_posix())
    _git_for_q5_test(root, "commit", "-qm", "synthetic D8")
    d5 = _git_for_q5_test(root, "rev-parse", "HEAD")
    _git_for_q5_test(root, "push", "-q")
    for relative in publications:
        (root / relative).chmod(0o444)
    return d5


def test_tracked_synthetic_baseline_cannot_authorize_residue_aliases(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _git_for_q5_test(root, "init", "-q")
    _git_for_q5_test(root, "config", "user.email", "q5@example.invalid")
    _git_for_q5_test(root, "config", "user.name", "Q8 Synthetic")
    results = root / "results"
    results.mkdir()
    marker = results / ".synthetic-baseline"
    marker.write_bytes(b"must not grant alternate residue authority\n")
    _git_for_q5_test(root, "add", marker.relative_to(root).as_posix())
    _git_for_q5_test(root, "commit", "-qm", "tracked synthetic marker")
    for row in builder.prereg.expected_failed_predecessor_attempts():
        residue = root / row["residue"]["slot1_stage"]["path"]
        residue.mkdir(mode=0o700)
    for leaf in (".synthetic-g9cb2-slot1", ".synthetic-g9cb3-slot1"):
        (results / leaf).mkdir(mode=0o700)
    assert _git_for_q5_test(
        root,
        "ls-files",
        "--",
        marker.relative_to(root).as_posix(),
    ) == marker.relative_to(root).as_posix()

    context = builder._PublicationContext(root)
    try:
        with pytest.raises(
            builder.TerminalG9CB8Failure,
            match="inventory|residue|authority|synthetic",
        ):
            builder._validate_closed_entry_phase(
                context,
                builder.Q8_PREREGISTRATION_PUBLICATION,
            )
    finally:
        context.close()


@pytest.mark.parametrize(
    "entry_name",
    ("validate_claim_preflight", "create_claim_only"),
)
def test_default_claim_entry_rejects_synthetic_authority_but_profile_succeeds(
    tmp_path: Path,
    entry_name: str,
) -> None:
    chain = _prepare_synthetic_a5_q5_p5_chain(tmp_path)
    root = chain["root"]
    security_calls: list[str] = []
    watched_codes = {
        builder._official_expected_security_profile.__code__: "official",
        builder._validated_expected_security_profile.__code__: "validated",
        builder.validate_preregistration.__code__: "preregistration",
    }

    def record_security_call(
        frame: Any,
        event: str,
        _argument: Any,
    ) -> None:
        if event == "call" and frame.f_code in watched_codes:
            security_calls.append(watched_codes[frame.f_code])

    entry = getattr(builder, entry_name)
    prior_profile = sys.getprofile()
    sys.setprofile(record_security_call)
    try:
        with pytest.raises(
            builder.TerminalG9CB8Failure,
            match="official A8/Q8/P8 security topology differs",
        ):
            entry(root)
        assert not (root / builder.CLAIM_PATH).exists()
        result = entry(
            root,
            expected_security_profile=_synthetic_security_profile(chain),
        )
    finally:
        sys.setprofile(prior_profile)

    assert security_calls.count("official") == 1
    assert "validated" in security_calls
    assert "preregistration" in security_calls
    assert not hasattr(builder, "_derived_expected_security_profile")
    if entry_name == "create_claim_only":
        payload = _read_synthetic_claim(chain)
        assert result == _synthetic_claim_binding(chain, payload)
    else:
        assert result["protocol_parent_commit"] == chain["p5"]
        assert not (root / builder.CLAIM_PATH).exists()


def test_default_security_expectations_are_fixed_a8_contract_constants() -> None:
    fixed = builder._fixed_security_expectations()
    assert fixed == {
        "failed_predecessor_preregistrations": (
            builder.prereg.expected_failed_predecessor_preregistration_bindings()
        ),
        "failed_predecessor_attempts": (
            builder.prereg.expected_failed_predecessor_attempts()
        ),
        "failed_predecessor_closures": (
            builder.prereg.expected_failed_predecessor_closures()
        ),
        "failed_predecessor_prepublication_closures": (
            builder.prereg.expected_failed_predecessor_prepublication_closures()
        ),
        "failed_predecessor_pre_sentinel_closures": (
            builder.prereg.expected_failed_predecessor_pre_sentinel_closures()
        ),
        "successor_preregistrations": (
            builder.prereg.expected_successor_preregistration_bindings()
        ),
        "authority_amendments": (
            builder._expected_authority_amendment_bindings()
        ),
        "protocol_paths": sorted(
            path.as_posix() for path in builder.prereg.PROTOCOL_PATHS
        ),
        "protocol_diff": list(builder.prereg.SUCCESSOR_PROTOCOL_DIFF),
        "preregistration_diff": list(builder.ACTIVE_PREREGISTRATION_DIFF),
        "claim_diff": list(builder.CLAIM_DIFF),
        "publication_diff": list(builder.PUBLICATION_DIFF),
    }
    assert builder.prereg.AUTHORITY_DECISION_COMMIT == (
        "33a5aad98c19cec29aba253933145d76b893be93"
    )
    assert len(fixed["failed_predecessor_preregistrations"]) == 2
    assert len(fixed["failed_predecessor_attempts"]) == 2
    assert len(fixed["failed_predecessor_closures"]) == 1
    assert [
        row["identity"]
        for row in fixed["failed_predecessor_prepublication_closures"]
    ] == ["G9CB-5", "G9CB-6"]
    assert [
        row["identity"]
        for row in fixed["failed_predecessor_pre_sentinel_closures"]
    ] == ["G9CB-7"]
    assert [
        row["identity"] for row in fixed["successor_preregistrations"]
    ] == ["G9CB-2", "G9CB-3", "G9CB-4"]
    assert [row["identity"] for row in fixed["authority_amendments"]] == [
        "G9CB-1A",
        "G9CB-1B",
        "G9CB-1C",
    ]
    assert len(fixed["protocol_paths"]) == 20
    assert {
        "authority_commit",
        "protocol_implementation_commit",
        "preregistration_seal_commit",
        "claim_commit",
        "publication_commit",
    }.isdisjoint(fixed)


def test_guarded_prepublication_closure_binding_rejects_nested_tampering(
    tmp_path: Path,
) -> None:
    expected = (
        builder.prereg.expected_failed_predecessor_prepublication_closures()
    )
    payload = {
        "bindings": {
            "failed_predecessor_prepublication_closures": copy.deepcopy(
                expected
            )
        }
    }
    assert (
        builder._validate_failed_predecessor_prepublication_closure_binding(
            payload
        )
        == expected
    )
    assert builder._validate_guarded_prepublication_closure_binding(
        payload, tmp_path
    ) == expected

    payload["bindings"]["failed_predecessor_prepublication_closures"][0][
        "failure"
    ]["bytes_opened"] += 1
    with pytest.raises(
        builder.TerminalG9CB8Failure,
        match="prepublication closure binding differs",
    ):
        builder._validate_guarded_prepublication_closure_binding(payload, tmp_path)


def test_guarded_pre_sentinel_closure_binding_rejects_nested_tampering() -> None:
    expected = builder.prereg.expected_failed_predecessor_pre_sentinel_closures()
    payload = {
        "bindings": {
            "failed_predecessor_pre_sentinel_closures": copy.deepcopy(
                expected
            )
        }
    }
    assert builder._validate_failed_predecessor_pre_sentinel_closure_binding(
        payload
    ) == expected
    payload["bindings"]["failed_predecessor_pre_sentinel_closures"][0][
        "failure"
    ]["wrapper_conformed"] = True
    with pytest.raises(
        builder.TerminalG9CB8Failure,
        match="pre-sentinel closure binding differs",
    ):
        builder._validate_failed_predecessor_pre_sentinel_closure_binding(
            payload
        )


@pytest.mark.parametrize(
    ("profile_class", "error_pattern"),
    (
        ("authority_commit", "A8/Q8/P8 topology differs"),
        (
            "failed_predecessor_preregistrations",
            "binding differs: failed_predecessor_preregistrations",
        ),
        (
            "failed_predecessor_attempts",
            "binding differs: failed_predecessor_attempts",
        ),
        (
            "failed_predecessor_closures",
            "binding differs: failed_predecessor_closures",
        ),
        (
            "failed_predecessor_prepublication_closures",
            "binding differs: failed_predecessor_prepublication_closures",
        ),
        (
            "failed_predecessor_pre_sentinel_closures",
            "binding differs: failed_predecessor_pre_sentinel_closures",
        ),
        (
            "successor_preregistrations",
            "binding differs: successor_preregistrations",
        ),
        (
            "authority_amendments",
            "binding differs: authority_amendments",
        ),
        ("protocol_diff", "path or diff contract differs"),
        ("protocol_implementation_commit", "identity binding differs"),
        ("preregistration_seal_commit", "A8/Q8/P8 topology differs"),
    ),
)
def test_explicit_security_profile_rejects_each_mutated_authority_class(
    tmp_path: Path,
    profile_class: str,
    error_pattern: str,
) -> None:
    chain = _prepare_synthetic_a5_q5_p5_chain(tmp_path)
    profile = _synthetic_security_profile(chain)
    if profile_class == "authority_commit":
        profile[profile_class] = "f" * 40
    elif profile_class == "failed_predecessor_preregistrations":
        profile[profile_class] = [{"identity": "forged-preregistration"}]
    elif profile_class in {
        "failed_predecessor_attempts",
        "successor_preregistrations",
        "authority_amendments",
    }:
        profile[profile_class].reverse()
    elif profile_class in {
        "failed_predecessor_closures",
        "failed_predecessor_prepublication_closures",
        "failed_predecessor_pre_sentinel_closures",
    }:
        profile[profile_class][0]["identity"] = "forged-closure"
    elif profile_class == "protocol_diff":
        profile[profile_class][0] = "M\tforged-protocol-path"
    elif profile_class == "protocol_implementation_commit":
        profile[profile_class] = chain["a5"]
    elif profile_class == "preregistration_seal_commit":
        profile[profile_class] = chain["q5"]
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(profile_class)

    with pytest.raises(
        builder.TerminalG9CB8Failure,
        match=error_pattern,
    ):
        builder.validate_claim_preflight(
            chain["root"],
            expected_security_profile=profile,
        )
    assert not (chain["root"] / builder.CLAIM_PATH).exists()


def test_uncached_head_blob_binding_authenticates_real_head_blob(
    tmp_path: Path,
) -> None:
    relative, raw = _synthetic_head_binding_repository(tmp_path / "repo")
    binding = builder._head_blob_binding(tmp_path / "repo", relative)
    assert binding == {
        "path": relative,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "git_blob": _git_for_q5_test(
            tmp_path / "repo",
            "rev-parse",
            f"HEAD:{relative}",
        ),
        "mode": "100644",
    }


def test_uncached_head_blob_binding_uses_keyword_only_git_pair_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    relative, raw = _synthetic_head_binding_repository(tmp_path / "repo")
    calls: list[tuple[str, bool, bool, bool]] = []
    original = builder._validate_git_pair_preflight

    def keyword_only_pair(
        root: Path,
        path_text: str,
        *,
        repository_relative: bool,
        declaration: Mapping[str, Any],
        verify_git: bool,
        require_tracked: bool = False,
    ) -> tuple[str, str] | None:
        calls.append(
            (
                path_text,
                repository_relative,
                verify_git,
                require_tracked,
            )
        )
        return original(
            root,
            path_text,
            repository_relative=repository_relative,
            declaration=declaration,
            verify_git=verify_git,
            require_tracked=require_tracked,
        )

    monkeypatch.setattr(
        builder,
        "_validate_git_pair_preflight",
        keyword_only_pair,
    )
    binding = builder._head_blob_binding(tmp_path / "repo", relative)
    assert binding["sha256"] == hashlib.sha256(raw).hexdigest()
    assert calls == [(relative, True, True, True)]


def test_parent_snapshot_keeps_ignored_paired_null_distinct_from_required_tracked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    chain = _prepare_synthetic_a5_q5_p5_chain(tmp_path)
    preregistration = chain["preregistration"]
    materialized = preregistration["bindings"][
        "failed_predecessor_prepublication_closures"
    ][1]["input_materialization"]["destination"]["path"]
    calls: dict[str, bool] = {}
    original = builder._validate_git_pair_preflight

    def recorded(
        root: Path,
        path_text: str,
        *,
        repository_relative: bool,
        declaration: Mapping[str, Any],
        verify_git: bool,
        require_tracked: bool = False,
        process_runner: Any | None = None,
    ) -> tuple[str, str] | None:
        calls[path_text] = require_tracked
        return original(
            root,
            path_text,
            repository_relative=repository_relative,
            declaration=declaration,
            verify_git=verify_git,
            require_tracked=require_tracked,
            process_runner=process_runner,
        )

    monkeypatch.setattr(builder, "_validate_git_pair_preflight", recorded)
    snapshot, pairs = builder._preauthenticate_parent_snapshot(
        chain["root"],
        preregistration,
        content_mode=builder._PREREGISTRATION_ONLY,
    )
    snapshot.close()
    assert pairs[materialized] is None
    assert calls[materialized] is False
    assert calls[builder.PREREGISTRATION_PATH.as_posix()] is True
    assert all(
        calls[path.as_posix()] is True
        for path in builder.prereg.PROTOCOL_PATHS
    )

    _git_for_q5_test(chain["root"], "add", "-f", materialized)
    with pytest.raises(
        builder.TerminalG9CB8Failure,
        match="paired-null bound input Git absence proof differs",
    ):
        builder._preauthenticate_parent_snapshot(
            chain["root"],
            preregistration,
            content_mode=builder._PREREGISTRATION_ONLY,
        )


def test_head_index_pairs_are_classified_before_first_bound_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    relative, _raw = _synthetic_head_binding_repository(tmp_path / "repo")
    events: list[str] = []
    original_git = builder._git_process
    original_read = builder._read_bound_regular_bytes

    def recorded_git(root: Path, *arguments: str):
        events.append(f"git:{arguments[0]}")
        return original_git(root, *arguments)

    def recorded_read(path: Path, path_text: str):
        events.append("open")
        return original_read(path, path_text)

    monkeypatch.setattr(builder, "_git_process", recorded_git)
    monkeypatch.setattr(builder, "_read_bound_regular_bytes", recorded_read)
    builder._head_blob_binding(tmp_path / "repo", relative)
    assert events[-1] == "open"
    assert events[:3] == [
        "git:ls-files",
        "git:ls-tree",
        "git:ls-files",
    ]


def test_global_clean_tree_gate_does_not_replace_cached_blob_authentication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    relative, _raw = _synthetic_head_binding_repository(tmp_path / "repo")
    (tmp_path / "repo" / relative).write_bytes(b"dirty synthetic bytes\n")
    monkeypatch.setattr(
        builder,
        "_require_clean_pushed_branch",
        lambda *_a, **_k: _git_for_q5_test(
            tmp_path / "repo", "rev-parse", "HEAD"
        ),
    )
    with pytest.raises(builder.TerminalG9CB8Failure, match="differs from HEAD"):
        builder._head_blob_binding(tmp_path / "repo", relative)


def test_bound_reader_rejects_symlink_in_parent_component(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "leaf").write_bytes(b"synthetic\n")
    (tmp_path / "alias").symlink_to(real, target_is_directory=True)
    with pytest.raises(builder.TerminalG9CB8Failure, match="symlink|component"):
        builder._read_bound_regular_bytes(
            tmp_path / "alias" / "leaf",
            "alias/leaf",
        )


def test_bound_reader_rejects_leaf_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"synthetic\n")
    leaf = tmp_path / "leaf"
    leaf.symlink_to(target)
    with pytest.raises(builder.TerminalG9CB8Failure, match="symlink|open"):
        builder._read_bound_regular_bytes(leaf, "leaf")


@pytest.mark.parametrize(
    "path_text",
    ["a/./b", "a/../b", "a//b"],
    ids=["dot-component", "dotdot-component", "empty-component"],
)
def test_bound_reader_rejects_noncanonical_components(
    tmp_path: Path,
    path_text: str,
) -> None:
    snapshot = builder._SecureBoundSnapshot(tmp_path)
    try:
        with pytest.raises(
            builder.TerminalG9CB8Failure,
            match="component|canonical|parent",
        ):
            snapshot.open_initial(path_text, True)
    finally:
        snapshot.close()


def test_distinct_hardlinked_bound_paths_fail_snapshot_authentication(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_bytes(b"synthetic\n")
    os.link(first, second)
    snapshot = builder._SecureBoundSnapshot(tmp_path)
    try:
        snapshot.open_initial("first", True)
        with pytest.raises(
            builder.TerminalG9CB8Failure,
            match="alias|filesystem object",
        ):
            snapshot.open_initial("second", True)
    finally:
        snapshot.close()


def test_duplicate_bound_declarations_share_one_open_and_two_reads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    leaf = tmp_path / "leaf"
    leaf.write_bytes(b"synthetic\n")
    reads: list[int] = []
    original_pread = builder.os.pread

    def recorded_pread(fd: int, size: int, offset: int) -> bytes:
        reads.append(fd)
        return original_pread(fd, size, offset)

    monkeypatch.setattr(builder.os, "pread", recorded_pread)
    snapshot = builder._SecureBoundSnapshot(tmp_path)
    try:
        first = snapshot.open_initial("leaf", True)
        second = snapshot.open_initial("leaf", True)
        snapshot.verify_final()
        assert first == second
        assert len(snapshot.file_descriptors) == 1
        assert len(reads) == 2
    finally:
        snapshot.close()


def test_conflicting_duplicate_declarations_fail_before_bound_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    leaf = tmp_path / "leaf"
    raw = b"synthetic\n"
    leaf.write_bytes(raw)
    opened: list[str] = []
    original_open = builder._SecureBoundSnapshot.open_initial

    def recorded_open(
        snapshot: builder._SecureBoundSnapshot,
        path_text: str,
        repository_relative: bool,
    ) -> tuple[bytes, os.stat_result]:
        opened.append(path_text)
        return original_open(snapshot, path_text, repository_relative)

    monkeypatch.setattr(
        builder._SecureBoundSnapshot,
        "open_initial",
        recorded_open,
    )
    preregistration = {
        "bindings": {
            "duplicates": [
                {
                    "path": "leaf",
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "git_blob": None,
                    "git_mode": None,
                },
                {
                    "path": "leaf",
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "git_blob": "f" * 40,
                    "git_mode": "100644",
                },
            ]
        }
    }
    with pytest.raises(
        builder.TerminalG9CB8Failure,
        match="conflicting duplicate input metadata",
    ):
        builder._preauthenticate_parent_snapshot(
            tmp_path,
            preregistration,
        )
    assert opened == []


def test_final_same_descriptor_read_detects_bound_byte_drift(
    tmp_path: Path,
) -> None:
    leaf = tmp_path / "leaf"
    leaf.write_bytes(b"initial\n")
    snapshot = builder._SecureBoundSnapshot(tmp_path)
    try:
        snapshot.open_initial("leaf", True)
        leaf.write_bytes(b"changed\n")
        with pytest.raises(
            builder.TerminalG9CB8Failure,
            match="final|snapshot",
        ):
            snapshot.verify_final()
    finally:
        snapshot.close()


def test_snapshot_consumers_use_cached_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = [
        Path("execution/gross9_rank7_clock_runtime.py"),
        Path("training/gross9_structural_clock_primitives.py"),
    ]
    rows: list[dict[str, Any]] = []
    for relative in paths:
        candidate = tmp_path / relative
        candidate.parent.mkdir(parents=True, exist_ok=True)
        raw = b"SYNTHETIC_VALUE = 1\n"
        candidate.write_bytes(raw)
        rows.append(
            {
                "path": relative.as_posix(),
                "path_type": "regular_file",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "git_blob": builder._git_blob_id(raw),
                "git_mode": "100644",
                "package_initializer": False,
            }
        )
    rows.sort(key=lambda row: row["path"])
    preregistration = {
        "bindings": {
            "protocol": [dict(row) for row in rows],
            "runtime_import_roots": [
                f"{module.replace('.', '/')}.py"
                for module in builder.RUNTIME_IMPORT_MODULES
            ],
            "runtime_import_closure": [dict(row) for row in rows],
        }
    }
    snapshot = builder._SecureBoundSnapshot(tmp_path)
    try:
        for row in rows:
            snapshot.open_initial(row["path"], True)

        def forbidden_path_read(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("cached consumer reopened a pathname")

        monkeypatch.setattr(
            builder,
            "_read_bound_regular_bytes",
            forbidden_path_read,
        )
        authenticated = builder._validate_regular_hashed_inputs(
            tmp_path,
            preregistration,
            verify_git=False,
            raw_cache=snapshot,
        )
        closures = builder._validate_static_closures(
            tmp_path,
            preregistration,
            verify_git=False,
            raw_cache=snapshot,
        )
        first = rows[0]
        cached_binding = builder._head_blob_binding(
            tmp_path,
            first["path"],
            raw_cache=snapshot,
            preclassified_pairs={
                first["path"]: (
                    builder._git_blob_id(snapshot[first["path"]][0]),
                    "100644",
                )
            },
        )
        assert [row["path"] for row in authenticated] == [
            row["path"] for row in rows
        ]
        assert closures["runtime"] == rows
        assert cached_binding["sha256"] == first["sha256"]
    finally:
        snapshot.close()


def test_worktree_hash_object_is_absent_from_snapshot_entry_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    chain = _prepare_synthetic_a5_q5_p5_chain(tmp_path)
    commands: list[tuple[str, ...]] = []
    original_git = builder._git_process

    def recorded_git(
        root: Path,
        *arguments: str,
    ) -> subprocess.CompletedProcess[bytes]:
        commands.append(tuple(arguments))
        return original_git(root, *arguments)

    monkeypatch.setattr(builder, "_git_process", recorded_git)
    snapshot, pairs = builder._preauthenticate_parent_snapshot(
        chain["root"],
        chain["preregistration"],
        content_mode=builder._PREREGISTRATION_ONLY,
    )
    context = builder._PublicationContext(chain["root"])
    try:
        builder._validate_closed_entry_phase(
            context,
            builder.P8_CLAIM_PREFLIGHT,
        )
        builder._final_parent_snapshot_recheck(
            chain["root"],
            snapshot,
            pairs,
            context,
            builder.P8_CLAIM_PREFLIGHT,
            builder.prereg.EXPECTED_BRANCH,
        )
    finally:
        snapshot.close()
        context.close()
    assert commands
    assert all(not command or command[0] != "hash-object" for command in commands)


def test_publication_uses_unnamed_otmpfile_without_named_staging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "artifact"
    observations: list[tuple[str, int, tuple[str, ...]]] = []
    original_link = builder._link_unnamed_procfd

    def recorded_link(
        unnamed_fd: int,
        results_fd: int,
        leaf: str,
    ) -> None:
        observations.append(
            (
                leaf,
                os.fstat(unnamed_fd).st_nlink,
                tuple(sorted(os.listdir(results_fd))),
            )
        )
        original_link(unnamed_fd, results_fd, leaf)

    monkeypatch.setattr(builder, "_link_unnamed_procfd", recorded_link)
    publication = builder._atomic_link_write_once(
        target,
        b"synthetic unnamed publication\n",
    )
    assert observations
    assert observations[-1] == ("artifact", 0, ())
    assert all(nlink == 0 and not entries for _leaf, nlink, entries in observations)
    assert all(".stage" not in leaf for leaf, _nlink, _entries in observations)
    assert tuple(path.name for path in tmp_path.iterdir()) == ("artifact",)
    assert publication["canonical_inode"] == publication["unnamed_inode"]


def test_publication_capability_probe_restores_inventory_and_rebaselines_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    context = builder._PublicationContext(tmp_path)
    raw = b"G9CB8 O_TMPFILE capability probe\n"
    links: list[dict[str, Any]] = []
    removals: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    fsyncs: list[int] = []
    rebaselines: list[
        tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...], tuple[int, ...]]
    ] = []
    original_linkat = builder._LIBC.linkat
    original_link = builder._link_unnamed_procfd
    original_unlink = builder.os.unlink
    original_fsync = builder.os.fsync
    original_rebaseline = context._rebaseline

    def force_empty_path_fallback(
        olddirfd: Any,
        oldpath: Any,
        newdirfd: Any,
        newpath: Any,
        flags: Any,
    ) -> int:
        source = getattr(oldpath, "value", oldpath)
        if source in (b"", None):
            builder.ctypes.set_errno(errno.ENOENT)
            return -1
        return int(
            original_linkat(
                olddirfd,
                oldpath,
                newdirfd,
                newpath,
                flags,
            )
        )

    def recorded_link(unnamed_fd: int, results_fd: int, leaf: str) -> None:
        unnamed = os.fstat(unnamed_fd)
        before_entries = builder._directory_entries(results_fd)
        unnamed_raw = builder._pread_complete(unnamed_fd, unnamed.st_size)
        original_link(unnamed_fd, results_fd, leaf)
        after_entries = builder._directory_entries(results_fd)
        canonical = os.stat(leaf, dir_fd=results_fd, follow_symlinks=False)
        canonical_fd = os.open(
            leaf,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=results_fd,
        )
        try:
            canonical_raw = builder._pread_complete(
                canonical_fd,
                canonical.st_size,
            )
        finally:
            os.close(canonical_fd)
        links.append(
            {
                "leaf": leaf,
                "before": before_entries,
                "after": after_entries,
                "unnamed_inode": (unnamed.st_dev, unnamed.st_ino),
                "canonical_inode": (canonical.st_dev, canonical.st_ino),
                "nlink_before": unnamed.st_nlink,
                "mode": stat.S_IMODE(canonical.st_mode),
                "regular": stat.S_ISREG(canonical.st_mode),
                "size": canonical.st_size,
                "unnamed_raw": unnamed_raw,
                "canonical_raw": canonical_raw,
            }
        )

    def recorded_unlink(
        path: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        before_entries = builder._directory_entries(context.results_fd)
        original_unlink(path, *args, **kwargs)
        removals.append(
            (
                before_entries,
                builder._directory_entries(context.results_fd),
            )
        )

    def recorded_fsync(descriptor: int) -> None:
        fsyncs.append(descriptor)
        original_fsync(descriptor)

    def recorded_rebaseline(expected_entries: tuple[str, ...]) -> None:
        before_entries = context.entries
        before_token = context.timestamp_token
        original_rebaseline(expected_entries)
        rebaselines.append(
            (
                before_entries,
                expected_entries,
                before_token,
                context.timestamp_token,
            )
        )

    monkeypatch.setattr(builder._LIBC, "linkat", force_empty_path_fallback)
    monkeypatch.setattr(builder, "_link_unnamed_procfd", recorded_link)
    monkeypatch.setattr(builder.os, "unlink", recorded_unlink)
    monkeypatch.setattr(builder.os, "fsync", recorded_fsync)
    context._rebaseline = recorded_rebaseline
    try:
        prior_timestamp = context.timestamp_token
        context.probe()
        assert tuple(results.iterdir()) == ()
        assert context.probed is True
        assert context.entries == ()
        assert context.timestamp_token != prior_timestamp
        assert len(links) == 1
        link = links[0]
        assert link["before"] == ()
        assert link["after"] == (link["leaf"],)
        assert link["nlink_before"] == 0
        assert link["regular"] is True
        assert link["mode"] == 0o444
        assert link["size"] == len(raw)
        assert link["unnamed_raw"] == raw
        assert link["canonical_raw"] == raw
        assert link["canonical_inode"] == link["unnamed_inode"]
        assert removals == [((link["leaf"],), ())]
        assert fsyncs.count(context.results_fd) == 2
        assert [
            (before, after) for before, after, _old, _new in rebaselines
        ] == [((), (link["leaf"],)), ((link["leaf"],), ())]
        assert all(
            after_token[-1] == before_token[-1] + 1
            and after_token != before_token
            for _before, _after, before_token, after_token in rebaselines
        )
    finally:
        context.close()


def test_unnamed_publication_eexist_never_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "artifact"
    target.write_bytes(b"existing\n")
    before = target.stat()
    with pytest.raises(FileExistsError):
        builder._atomic_link_write_once(target, b"replacement\n")
    after = target.stat()
    assert target.read_bytes() == b"existing\n"
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)


def test_unnamed_publication_preserves_inode_bytes_hash_size_and_mode(
    tmp_path: Path,
) -> None:
    target = tmp_path / "artifact"
    raw = b"synthetic publication\n"
    result = builder._atomic_link_write_once(target, raw)
    info = os.stat(target, follow_symlinks=False)
    assert result["canonical_inode"] == result["unnamed_inode"]
    assert result["sha256"] == hashlib.sha256(raw).hexdigest()
    assert info.st_size == len(raw)
    assert stat.S_ISREG(info.st_mode)
    assert stat.S_IMODE(info.st_mode) == 0o444


@pytest.mark.parametrize(
    "fault",
    [
        "write",
        "fchmod",
        "file_fsync",
        "same_fd_verify",
        "procfd_link",
        "canonical_open",
        "inode_compare",
        "directory_fsync",
    ],
)
def test_unnamed_publication_faults_fail_closed_without_bound_reread(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fault: str,
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    context = builder._PublicationContext(tmp_path)
    context.probed = True
    rechecks: list[str] = []
    original_open = builder.os.open
    original_fstat = builder.os.fstat
    original_fsync = builder.os.fsync

    def injected(*_args, **_kwargs):
        raise OSError(errno.EIO, fault)

    if fault == "write":
        monkeypatch.setattr(builder, "_write_all", injected)
    elif fault == "fchmod":
        monkeypatch.setattr(builder.os, "fchmod", injected)
    elif fault == "file_fsync":
        monkeypatch.setattr(builder.os, "fsync", injected)
    elif fault == "same_fd_verify":
        monkeypatch.setattr(builder, "_pread_complete", lambda *_a: b"drift")
    elif fault == "procfd_link":
        monkeypatch.setattr(builder, "_link_unnamed_procfd", injected)
    elif fault == "canonical_open":
        def fail_canonical_open(path, *args, **kwargs):
            if path == fault:
                return injected()
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(builder.os, "open", fail_canonical_open)
    elif fault == "inode_compare":
        def drift_canonical_fstat(descriptor: int):
            observed = original_fstat(descriptor)
            if descriptor not in {
                context.repository_fd,
                context.results_fd,
            } and stat.S_IMODE(observed.st_mode) == 0o444:
                values = list(observed)
                values[1] += 1
                return os.stat_result(values)
            return observed

        monkeypatch.setattr(builder.os, "fstat", drift_canonical_fstat)
    else:
        def fail_directory_fsync(descriptor: int) -> None:
            if descriptor == context.results_fd:
                injected()
            original_fsync(descriptor)

        monkeypatch.setattr(builder.os, "fsync", fail_directory_fsync)
    try:
        with pytest.raises(
            (OSError, builder.TerminalG9CB8Failure),
            match=fault + "|publication|inode",
        ):
            context.publish(
                Path("results") / fault,
                b"synthetic\n",
                prelink_recheck=lambda: rechecks.append("recheck"),
            )
        assert len(rechecks) <= 1
        assert not any(".stage-" in path.name for path in results.iterdir())
    finally:
        context.close()


def test_results_directory_substitution_before_link_fails_closed(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    context = builder._PublicationContext(tmp_path)
    try:
        context.probe()
        moved = tmp_path / "moved"
        results.rename(moved)
        results.mkdir()
        with pytest.raises(
            builder.TerminalG9CB8Failure,
            match="directory|identity",
        ):
            context.publish(
                Path("results/artifact"),
                b"synthetic\n",
            )
    finally:
        context.close()


def test_absolute_binding_uses_retained_root_and_tmp_anchors(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    candidate = tmp_path / "absolute" / "leaf"
    candidate.parent.mkdir()
    candidate.write_bytes(b"synthetic absolute binding\n")
    path_text = candidate.as_posix()
    snapshot = builder._SecureBoundSnapshot(
        repository,
        absolute_allowlist=frozenset({path_text}),
    )
    try:
        raw, _info = snapshot.open_initial(path_text, False)
        expected_parents = {
            ("absolute", tuple(candidate.parts[1:index]))
            for index in range(1, len(candidate.parts))
        }
        assert ("absolute", ()) in snapshot.directory_descriptors
        assert ("absolute", ("tmp",)) in snapshot.directory_descriptors
        assert expected_parents.issubset(snapshot.directory_descriptors)
        assert raw == b"synthetic absolute binding\n"
        snapshot.verify_final()
    finally:
        snapshot.close()


def test_unlisted_absolute_binding_fails_before_leaf_access(tmp_path: Path) -> None:
    candidate = tmp_path / "unlisted"
    candidate.write_bytes(b"synthetic\n")
    snapshot = builder._SecureBoundSnapshot(tmp_path)
    try:
        with pytest.raises(
            builder.TerminalG9CB8Failure,
            match="absolute|allowlist",
        ):
            snapshot.open_initial(candidate.as_posix(), False)
    finally:
        snapshot.close()


def test_worker_invocation_passes_capability_and_ledger_descriptors(
    tmp_path: Path,
) -> None:
    (tmp_path / "results").mkdir()
    stage = (
        tmp_path
        / "results/.gross9-structural-clock-g9cb8-worker-invocation"
    )
    capability = builder._prepare_worker_capability(
        root=tmp_path,
        output_dir=stage,
        slot=1,
        parent_pid=os.getpid(),
    )
    try:
        invocation = builder._prepare_worker(
            root=tmp_path,
            capability=capability,
            other_stage_directory=(
                "results/.gross9-structural-clock-g9cb8-worker-other"
            ),
            synthetic_input=None,
            parent_authentication={},
        )
        command = invocation["command"]
        capability_fd = int(
            command[command.index("--worker-capability-fd") + 1]
        )
        ledger_fd = int(command[command.index("--worker-ledger-fd") + 1])
        assert (capability_fd, ledger_fd) == (
            capability["read_fd"],
            capability["ledger_fd"],
        )
        assert capability_fd != ledger_fd
    finally:
        for key in ("read_fd", "ledger_fd"):
            descriptor = int(capability[key])
            if descriptor >= 0:
                os.close(descriptor)
                capability[key] = -1
        builder._zero_token(capability["token"])


def test_worker_capability_rows_bind_unnamed_ledger_identity(
    tmp_path: Path,
) -> None:
    (tmp_path / "results").mkdir()
    capability = builder._prepare_worker_capability(
        root=tmp_path,
        output_dir=(
            tmp_path
            / "results/.gross9-structural-clock-g9cb8-worker-capability"
        ),
        slot=1,
        parent_pid=os.getpid(),
    )
    try:
        row = capability["row"]
        ledger = os.fstat(capability["ledger_fd"])
        carrier = os.fstat(capability["read_fd"])
        assert tuple(row) == CAPABILITY_KEYS
        assert row["ledger_carrier_kind"] == "unnamed_otmpfile_v1"
        assert (row["ledger_device"], row["ledger_inode"]) == (
            ledger.st_dev,
            ledger.st_ino,
        )
        assert row["ledger_initial_type"] == "regular_file"
        assert row["ledger_initial_mode"] == "0600"
        assert row["ledger_initial_size"] == 0
        assert (carrier.st_dev, carrier.st_ino) != (
            ledger.st_dev,
            ledger.st_ino,
        )
        assert capability["read_fd"] != capability["ledger_fd"]
    finally:
        for key in ("read_fd", "ledger_fd"):
            descriptor = int(capability[key])
            if descriptor >= 0:
                os.close(descriptor)
                capability[key] = -1
        builder._zero_token(capability["token"])


def test_worker_ledger_uses_sole_prebound_procfd_self_link(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    own = "results/.gross9-structural-clock-g9cb8-worker-ledger-own"
    (tmp_path / own).mkdir(mode=0o700)
    metadata = tmp_path / "synthetic-metadata"
    metadata.write_bytes(b"synthetic metadata\n")
    script = tmp_path / "worker_ledger_harness.py"
    script.write_text(
        textwrap.dedent(
            """
            import json
            import os
            from pathlib import Path

            from training import build_gross9_structural_clock_bundle as b

            root = Path(os.environ["G9CB_TEST_ROOT"]).resolve()
            own = "results/.gross9-structural-clock-g9cb8-worker-ledger-own"
            other = "results/.gross9-structural-clock-g9cb8-worker-ledger-other"
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
            repository_fd = os.open(root, flags)
            results_fd = os.open("results", flags, dir_fd=repository_fd)
            filesystem_root_fd = os.open("/", flags)
            capability = b._prepare_worker_capability(
                root=root,
                output_dir=root / own,
                slot=1,
                parent_pid=os.getpid(),
                results_fd=results_fd,
            )
            distinct = capability["read_fd"] != capability["ledger_fd"]
            token = b._consume_worker_capability(
                capability["read_fd"], capability["row"]
            )
            capability["read_fd"] = -1
            guard = b._WorkerIsolationGuard(
                root=root,
                own_stage=own,
                other_stage=other,
                ledger_paths=b.WORKER_LEDGER_PATHS,
                repository_fd=repository_fd,
                results_fd=results_fd,
                filesystem_root_fd=filesystem_root_fd,
                ledger_fd=capability["ledger_fd"],
            )
            guard.bind_ledger_slot(1)
            guard.authorize_directory_sync(root / own)
            snapshot = b._SecureBoundSnapshot(
                root,
                repository_fd=repository_fd,
                filesystem_root_fd=filesystem_root_fd,
                opener=guard._original_os_open,
                register_descriptor=guard.register_snapshot_descriptor,
            )
            snapshot.open_initial("synthetic-metadata", True)
            guard.install()

            events = []
            link_calls = []
            rechecks = []
            canonical_opens = []
            canonical_fds = set()
            ledger_reads = []
            read_phase = "prelink"
            original_link = b._link_unnamed_procfd
            original_recheck = b._worker_metadata_final_recheck
            original_open = guard._original_os_open
            original_pread_complete = b._pread_complete

            def recorded_link(unnamed_fd, directory_fd, leaf):
                global read_phase
                link_calls.append((unnamed_fd, directory_fd, leaf))
                events.append("procfd_link")
                result = original_link(unnamed_fd, directory_fd, leaf)
                read_phase = "postlink"
                return result

            def recorded_recheck(active_guard, active_snapshot, binding, ledger_fd):
                rechecks.append(ledger_fd)
                events.append("metadata_recheck")
                return original_recheck(
                    active_guard, active_snapshot, binding, ledger_fd
                )

            def recorded_open(path, flags, *args, **kwargs):
                descriptor = original_open(path, flags, *args, **kwargs)
                if path == b.WORKER_LEDGER_PATHS[0].name:
                    canonical_fds.add(descriptor)
                    canonical_opens.append({
                        "dir_fd": kwargs.get("dir_fd"),
                        "fd": descriptor,
                        "flags": flags,
                        "path": path,
                    })
                    events.append("canonical_open")
                return descriptor

            def recorded_pread_complete(descriptor, size):
                if descriptor == capability["ledger_fd"]:
                    ledger_reads.append({
                        "fd_kind": "unnamed",
                        "phase": read_phase,
                    })
                elif descriptor in canonical_fds:
                    ledger_reads.append({
                        "fd_kind": "canonical",
                        "phase": read_phase,
                    })
                return original_pread_complete(descriptor, size)

            b._link_unnamed_procfd = recorded_link
            b._worker_metadata_final_recheck = recorded_recheck
            b._pread_complete = recorded_pread_complete
            guard._original_os_open = recorded_open
            claim = {"claim_hash": "1" * 64}
            preregistration = {"manifest_hash": "2" * 64}
            sentinel = {"manifest_hash": "3" * 64}
            published = b._publish_worker_ledger(
                guard=guard,
                snapshot=snapshot,
                ledger_fd=capability["ledger_fd"],
                binding=capability["row"],
                claim=claim,
                preregistration=preregistration,
                sentinel=sentinel,
                authority_amendments=[],
            )
            b._worker_ledger_linked_checkpoint(
                guard,
                capability["row"],
                published,
            )
            print(json.dumps({
                "canonical_opens": canonical_opens,
                "canonical_distinct": (
                    published["canonical_fd"] != capability["ledger_fd"]
                ),
                "descriptor_count": 2,
                "descriptors_distinct": distinct,
                "events": events,
                "final_verified": snapshot.final_verified,
                "ledger_reads": ledger_reads,
                "link_calls": [
                    [fd, directory_fd, leaf]
                    for fd, directory_fd, leaf in link_calls
                ],
                "rechecks": rechecks,
                "row": capability["row"],
                "sha256": published["sha256"],
            }, sort_keys=True))
            if published["canonical_fd"] != capability["ledger_fd"]:
                os.close(published["canonical_fd"])
            snapshot.close()
            b._zero_token(token)
            b._zero_token(capability["token"])
            os.close(capability["ledger_fd"])
            os.close(guard.stage_fd)
            os.close(filesystem_root_fd)
            os.close(results_fd)
            os.close(repository_fd)
            """
        ),
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.update(
        {
            "G9CB_TEST_ROOT": tmp_path.as_posix(),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": builder.REPOSITORY_ROOT.as_posix(),
        }
    )
    completed = subprocess.run(
        [sys.executable, "-B", str(script)],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout.splitlines()[-1])
    assert observed["descriptor_count"] == 2
    assert observed["descriptors_distinct"] is True
    assert set(observed["row"]) == set(CAPABILITY_KEYS)
    assert len(observed["row"]) == len(CAPABILITY_KEYS)
    assert observed["final_verified"] is True
    assert len(observed["rechecks"]) == 1
    assert len(observed["link_calls"]) == 1
    assert observed["link_calls"][0][2] == builder.WORKER_LEDGER_PATHS[0].name
    assert observed["events"] == [
        "metadata_recheck",
        "procfd_link",
        "canonical_open",
    ]
    assert observed["canonical_distinct"] is True
    assert len(observed["canonical_opens"]) == 1
    canonical_open = observed["canonical_opens"][0]
    assert canonical_open["path"] == builder.WORKER_LEDGER_PATHS[0].name
    assert canonical_open["flags"] == (
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    assert canonical_open["dir_fd"] == observed["link_calls"][0][1]
    assert canonical_open["fd"] != observed["link_calls"][0][0]
    assert observed["ledger_reads"] == [
        {"fd_kind": "unnamed", "phase": "prelink"},
        {"fd_kind": "canonical", "phase": "postlink"},
    ]
    ledger = tmp_path / builder.WORKER_LEDGER_PATHS[0]
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == observed["sha256"]
    assert stat.S_IMODE(ledger.stat().st_mode) == 0o444


def test_worker_canonical_ledger_open_authority_is_one_shot_and_fail_closed(
    tmp_path: Path,
) -> None:
    script = tmp_path / "canonical_ledger_open_authority_harness.py"
    script.write_text(
        textwrap.dedent(
            """
            import errno
            import json
            import os
            from pathlib import Path

            from training import build_gross9_structural_clock_bundle as b

            case = os.environ["G9CB_TEST_CASE"]
            root = Path(os.environ["G9CB_TEST_ROOT"]).resolve()
            own_stage = (
                "results/.gross9-structural-clock-g9cb8-worker-authority-own"
            )
            other_stage = (
                "results/.gross9-structural-clock-g9cb8-worker-authority-other"
            )
            directory_flags = (
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
            )
            repository_fd = os.open(root, directory_flags)
            results_fd = os.open(
                "results", directory_flags, dir_fd=repository_fd
            )
            filesystem_root_fd = os.open("/", directory_flags)
            ledger_fd = os.open(
                ".",
                os.O_RDWR | os.O_TMPFILE | os.O_CLOEXEC,
                0o600,
                dir_fd=results_fd,
            )
            guard = b._WorkerIsolationGuard(
                root=root,
                own_stage=own_stage,
                other_stage=other_stage,
                ledger_paths=b.WORKER_LEDGER_PATHS,
                repository_fd=repository_fd,
                results_fd=results_fd,
                filesystem_root_fd=filesystem_root_fd,
                ledger_fd=ledger_fd,
            )
            guard.bind_ledger_slot(1)
            payload = b"synthetic one-shot canonical ledger\\n"
            b._write_all(ledger_fd, payload)
            os.fchmod(ledger_fd, 0o444)
            os.fsync(ledger_fd)
            leaf = b.WORKER_LEDGER_PATHS[0].name
            if case == "preopen_mismatch":
                mismatch_fd = os.open(
                    leaf,
                    (
                        os.O_WRONLY
                        | os.O_CREAT
                        | os.O_EXCL
                        | os.O_NOFOLLOW
                        | os.O_CLOEXEC
                    ),
                    0o444,
                    dir_fd=results_fd,
                )
                b._write_all(mismatch_fd, b"different canonical inode\\n")
                os.fchmod(mismatch_fd, 0o444)
                os.fsync(mismatch_fd)
                os.close(mismatch_fd)
            else:
                b._link_unnamed_procfd(ledger_fd, results_fd, leaf)
            guard.install()
            exact_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC

            def rejected(operation):
                try:
                    result = operation()
                except b.TerminalG9CB8Failure as exc:
                    return {
                        "message": str(exc),
                        "status": "terminal",
                    }
                if isinstance(result, int):
                    os.close(result)
                elif result is not None and hasattr(result, "close"):
                    result.close()
                return {"status": "accepted"}

            def authority_evidence(calls):
                counters = guard.counters()
                isolation_or_unauthorized_total = sum(
                    counters[name]
                    for name in (
                        "child_process_creation_events",
                        "other_stage_access_events",
                        "other_slot_ledger_access_events",
                        "unauthorized_write_or_ipc_events",
                    )
                )
                return {
                    "calls": list(calls),
                    "count": guard._owned_ledger_open_count,
                    "counters": counters,
                    "isolation_or_unauthorized_total": (
                        isolation_or_unauthorized_total
                    ),
                    "state": guard._owned_ledger_open_state,
                }

            observed = {
                "case": case,
                "ledger_fd": ledger_fd,
                "results_fd": results_fd,
            }
            if case == "success_then_retry":
                underlying_open = guard._original_os_open
                calls = []

                def counted_open(path, flags, *args, **kwargs):
                    descriptor = underlying_open(path, flags, *args, **kwargs)
                    calls.append(
                        {
                            "dir_fd": kwargs.get("dir_fd"),
                            "fd": descriptor,
                            "flags": flags,
                            "path": path,
                        }
                    )
                    return descriptor

                guard._original_os_open = counted_open
                canonical_fd = guard.open_owned_canonical_ledger_once(
                    leaf, exact_flags, dir_fd=results_fd
                )
                canonical_info = os.fstat(canonical_fd)
                unnamed_info = os.fstat(ledger_fd)
                observed.update(
                    {
                        "canonical_distinct": canonical_fd != ledger_fd,
                        "canonical_matches_unnamed": (
                            canonical_info.st_dev,
                            canonical_info.st_ino,
                        )
                        == (unnamed_info.st_dev, unnamed_info.st_ino),
                        "calls": calls,
                        "retry_while_open": rejected(
                            lambda: guard.open_owned_canonical_ledger_once(
                                leaf, exact_flags, dir_fd=results_fd
                            )
                        ),
                    }
                )
                os.close(canonical_fd)
                try:
                    os.fstat(canonical_fd)
                except OSError as exc:
                    observed["closed_errno"] = exc.errno
                observed["retry_after_close"] = rejected(
                    lambda: guard.open_owned_canonical_ledger_once(
                        leaf, exact_flags, dir_fd=results_fd
                    )
                )
                observed["calls_after_retries"] = calls
            elif case == "failed_first_open":
                underlying_open = guard._original_os_open
                attempts = []

                def failed_open(path, flags, *args, **kwargs):
                    attempts.append(
                        {
                            "dir_fd": kwargs.get("dir_fd"),
                            "flags": flags,
                            "path": path,
                        }
                    )
                    raise OSError(errno.EIO, "injected canonical open failure")

                guard._original_os_open = failed_open
                try:
                    guard.open_owned_canonical_ledger_once(
                        leaf, exact_flags, dir_fd=results_fd
                    )
                except OSError as exc:
                    observed["first_failure"] = {
                        "errno": exc.errno,
                        "type": type(exc).__name__,
                    }
                guard._original_os_open = underlying_open
                observed["retry"] = rejected(
                    lambda: guard.open_owned_canonical_ledger_once(
                        leaf, exact_flags, dir_fd=results_fd
                    )
                )
                observed["attempts"] = attempts
            elif case in {
                "absolute_ledger",
                "dot_leaf",
                "other_ledger",
                "preopen_mismatch",
                "wrong_dirfd",
                "wrong_flags",
                "wrong_leaf",
            }:
                canonical = root / b.WORKER_LEDGER_PATHS[0]
                underlying_open = guard._original_os_open
                calls = []

                def counted_invalid_open(path, flags, *args, **kwargs):
                    calls.append(
                        {
                            "dir_fd": kwargs.get("dir_fd"),
                            "flags": flags,
                            "path": path,
                        }
                    )
                    return underlying_open(path, flags, *args, **kwargs)

                guard._original_os_open = counted_invalid_open
                operations = {
                    "dot_leaf": lambda: guard.open_owned_canonical_ledger_once(
                        f"./{leaf}", exact_flags, dir_fd=results_fd
                    ),
                    "absolute_ledger": lambda: (
                        guard.open_owned_canonical_ledger_once(
                            canonical.as_posix(),
                            exact_flags,
                            dir_fd=results_fd,
                        )
                    ),
                    "wrong_flags": lambda: (
                        guard.open_owned_canonical_ledger_once(
                            leaf,
                            exact_flags | os.O_NONBLOCK,
                            dir_fd=results_fd,
                        )
                    ),
                    "wrong_dirfd": lambda: (
                        guard.open_owned_canonical_ledger_once(
                            leaf, exact_flags, dir_fd=repository_fd
                        )
                    ),
                    "other_ledger": lambda: (
                        guard.open_owned_canonical_ledger_once(
                            b.WORKER_LEDGER_PATHS[1].name,
                            exact_flags,
                            dir_fd=results_fd,
                        )
                    ),
                    "wrong_leaf": lambda: (
                        guard.open_owned_canonical_ledger_once(
                            f"wrong-{leaf}",
                            exact_flags,
                            dir_fd=results_fd,
                        )
                    ),
                    "preopen_mismatch": lambda: (
                        guard.open_owned_canonical_ledger_once(
                            leaf,
                            exact_flags,
                            dir_fd=results_fd,
                        )
                    ),
                }
                observed["first_rejection"] = rejected(operations[case])
                observed["after_invalid"] = authority_evidence(calls)
                observed["exact_retry"] = rejected(
                    lambda: guard.open_owned_canonical_ledger_once(
                        leaf, exact_flags, dir_fd=results_fd
                    )
                )
                observed["after_retry"] = authority_evidence(calls)
            else:
                canonical = root / b.WORKER_LEDGER_PATHS[0]
                path_operations = {
                    "path_open": lambda: open(canonical, "rb"),
                    "path_stat": lambda: os.stat(canonical),
                    "path_read": lambda: canonical.read_bytes(),
                }
                observed["rejection"] = rejected(path_operations[case])
            print(json.dumps(observed, sort_keys=True))
            """
        ),
        encoding="utf-8",
    )
    invalid_authority_cases = (
        "dot_leaf",
        "absolute_ledger",
        "wrong_leaf",
        "wrong_flags",
        "wrong_dirfd",
        "other_ledger",
        "preopen_mismatch",
    )
    path_guard_cases = (
        "path_open",
        "path_stat",
        "path_read",
    )
    cases = (
        "success_then_retry",
        *invalid_authority_cases,
        *path_guard_cases,
        "failed_first_open",
    )
    observed: dict[str, dict[str, Any]] = {}
    for case in cases:
        root = tmp_path / case
        (root / "results").mkdir(parents=True)
        (
            root
            / "results/.gross9-structural-clock-g9cb8-worker-authority-own"
        ).mkdir(mode=0o700)
        environment = dict(os.environ)
        environment.update(
            {
                "G9CB_TEST_CASE": case,
                "G9CB_TEST_ROOT": root.as_posix(),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": builder.REPOSITORY_ROOT.as_posix(),
            }
        )
        completed = subprocess.run(
            [sys.executable, "-B", str(script)],
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
        assert completed.returncode == 0, (
            f"{case}: stdout={completed.stdout!r} stderr={completed.stderr!r}"
        )
        observed[case] = json.loads(completed.stdout.splitlines()[-1])

    success = observed["success_then_retry"]
    assert success["canonical_distinct"] is True
    assert success["canonical_matches_unnamed"] is True
    assert success["closed_errno"] == errno.EBADF
    assert success["calls"] == success["calls_after_retries"] == [
        {
            "dir_fd": success["results_fd"],
            "fd": success["calls"][0]["fd"],
            "flags": os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            "path": builder.WORKER_LEDGER_PATHS[0].name,
        }
    ]
    assert success["calls"][0]["fd"] != success["ledger_fd"]
    for retry in (success["retry_while_open"], success["retry_after_close"]):
        assert retry["status"] == "terminal"
        assert "canonical-open authority differs" in retry["message"]

    for case in invalid_authority_cases:
        first = observed[case]["first_rejection"]
        assert first["status"] == "terminal", case
        assert "ledger" in first["message"], case
        immediate = observed[case]["after_invalid"]
        assert immediate["state"] == "failed", case
        assert immediate["count"] == 1, case
        assert immediate["calls"] == [], case
        assert immediate["isolation_or_unauthorized_total"] == 1, case
        assert immediate["counters"]["other_stage_absence_checks"] == 1, case

        retry = observed[case]["exact_retry"]
        assert retry["status"] == "terminal", case
        assert "canonical-open authority differs" in retry["message"], case
        final = observed[case]["after_retry"]
        assert final["state"] == "failed", case
        assert final["count"] == 1, case
        assert final["calls"] == [], case
        assert final["isolation_or_unauthorized_total"] >= 1, case

    for case in path_guard_cases:
        rejection = observed[case]["rejection"]
        assert rejection["status"] == "terminal", case
        assert "ledger" in rejection["message"], case

    failed = observed["failed_first_open"]
    assert failed["first_failure"] == {"errno": errno.EIO, "type": "OSError"}
    assert failed["attempts"] == [
        {
            "dir_fd": failed["results_fd"],
            "flags": os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            "path": builder.WORKER_LEDGER_PATHS[0].name,
        }
    ]
    assert failed["retry"]["status"] == "terminal"
    assert "canonical-open authority differs" in failed["retry"]["message"]


def test_worker_guard_forbids_other_procfs_link_and_unregistered_dirfd(
    tmp_path: Path,
) -> None:
    guard = _guard(tmp_path)
    try:
        with pytest.raises(
            builder.TerminalG9CB8Failure,
            match="descriptor namespace",
        ):
            guard._checked_path(f"/proc/self/fd/{guard.repository_fd}")
        with pytest.raises(
            builder.TerminalG9CB8Failure,
            match="dir_fd",
        ):
            guard._reject_dir_fds({"dir_fd": guard.repository_fd})
        with pytest.raises(
            builder.TerminalG9CB8Failure,
            match="worker ledger",
        ):
            guard._checked_path(guard.ledger_paths[1])
        assert guard.unauthorized_write_or_ipc_events == 1
        assert guard.other_slot_ledger_access_events == 1
    finally:
        for descriptor in (
            guard.ledger_fd,
            guard.filesystem_root_fd,
            guard.results_fd,
            guard.repository_fd,
        ):
            if descriptor is not None:
                os.close(descriptor)


def _materialize_closed_phase_state(
    root: Path,
    present: tuple[Path, ...] = (),
) -> Path:
    results = root / "results"
    results.mkdir(exist_ok=True)
    for row in builder.prereg.expected_failed_predecessor_attempts():
        stage = root / row["residue"]["slot1_stage"]["path"]
        stage.mkdir(mode=0o700, exist_ok=True)
        stage.chmod(0o700)
    for relative in present:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"synthetic closed state: {relative}\n".encode())
        target.chmod(0o444)
    return results


def _rebaseline_restored_results_timestamp(
    context: builder._PublicationContext,
) -> None:
    context.timestamp_token = (
        *builder._descriptor_token(os.fstat(context.results_fd)),
        context.timestamp_token[-1] + 1,
    )


def _register_test_stage(
    context: builder._PublicationContext,
    stage: Path,
) -> int:
    descriptor = os.open(
        stage.name,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=context.results_fd,
    )
    context.stage_descriptors[stage.name] = descriptor
    context.stage_tokens[stage.name] = builder._descriptor_token(
        os.fstat(descriptor)
    )
    context.stage_entries[stage.name] = builder._directory_entries(descriptor)
    return descriptor


def test_production_declares_exact_stable_checkpoint_sequence() -> None:
    assert builder.PRODUCTION_CHECKPOINTS == (
        "C8_PRODUCTION_PREFLIGHT",
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


def test_helper_local_transients_are_not_accepted_as_checkpoints() -> None:
    transients = set(builder.HELPER_LOCAL_TRANSIENT_STATES)
    checkpoints = set(builder.PRODUCTION_CHECKPOINTS)
    assert transients
    assert transients.isdisjoint(checkpoints)
    for transient in transients:
        with pytest.raises(builder.TerminalG9CB8Failure, match="transient"):
            builder._validate_production_checkpoint(transient)


def test_closed_phase_rejects_restored_inventory_timestamp_drift(
    tmp_path: Path,
) -> None:
    results = _materialize_closed_phase_state(tmp_path)
    context = builder._PublicationContext(tmp_path)
    try:
        builder._validate_closed_entry_phase(
            context,
            builder.Q8_PREREGISTRATION_PUBLICATION,
        )
        before = context.timestamp_token
        time.sleep(0.01)
        ephemeral = results / ".g9cb8-unauthorized-ephemeral"
        ephemeral.mkdir()
        ephemeral.rmdir()
        assert tuple(sorted(path.name for path in results.iterdir())) == (
            context.entries
        )
        assert builder._descriptor_token(os.fstat(context.results_fd)) != (
            before[:-1]
        )
        with pytest.raises(
            builder.TerminalG9CB8Failure,
            match="timestamp|drift|results",
        ):
            builder._validate_closed_entry_phase(
                context,
                builder.Q8_PREREGISTRATION_PUBLICATION,
            )
    finally:
        context.close()


@pytest.mark.parametrize("identity", ["G9CB-2", "G9CB-3"])
@pytest.mark.parametrize("replacement_state", ["empty", "nonempty"])
def test_builder_retains_predecessor_residue_edge_through_final_recheck(
    tmp_path: Path,
    identity: str,
    replacement_state: str,
) -> None:
    chain = _prepare_synthetic_a5_q5_p5_chain(tmp_path)
    root = chain["root"]
    snapshot, pairs = builder._preauthenticate_parent_snapshot(
        root,
        chain["preregistration"],
        content_mode=builder._PREREGISTRATION_ONLY,
    )
    context = builder._PublicationContext(root)
    row = next(
        item
        for item in builder.prereg.expected_failed_predecessor_attempts()
        if item["identity"] == identity
    )
    stage_leaf = Path(row["residue"]["slot1_stage"]["path"]).name
    detached_leaf = f".detached-{stage_leaf}"
    try:
        builder._validate_closed_entry_phase(
            context,
            builder.P8_CLAIM_PREFLIGHT,
        )
        original = os.stat(
            stage_leaf,
            dir_fd=context.results_fd,
            follow_symlinks=False,
        )
        os.rename(
            stage_leaf,
            detached_leaf,
            src_dir_fd=context.results_fd,
            dst_dir_fd=context.results_fd,
        )
        os.rmdir(detached_leaf, dir_fd=context.results_fd)
        os.mkdir(stage_leaf, 0o700, dir_fd=context.results_fd)
        replacement = os.stat(
            stage_leaf,
            dir_fd=context.results_fd,
            follow_symlinks=False,
        )
        assert (replacement.st_dev, replacement.st_ino) != (
            original.st_dev,
            original.st_ino,
        )
        assert stat.S_IMODE(replacement.st_mode) == 0o700
        if replacement_state == "nonempty":
            replacement_fd = os.open(
                stage_leaf,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=context.results_fd,
            )
            try:
                marker_fd = os.open(
                    "replacement-marker",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                    0o600,
                    dir_fd=replacement_fd,
                )
                os.close(marker_fd)
            finally:
                os.close(replacement_fd)
        _rebaseline_restored_results_timestamp(context)
        snapshot.rebaseline_directory_timestamps(
            matching_identity=context.results_token
        )
        with pytest.raises(
            builder.TerminalG9CB8Failure,
            match="residue|directory|parent|component|identity|graph",
        ):
            builder._validate_closed_entry_phase(
                context,
                builder.P8_CLAIM_PREFLIGHT,
            )
            builder._final_parent_snapshot_recheck(
                root,
                snapshot,
                pairs,
                context,
                builder.P8_CLAIM_PREFLIGHT,
                builder.prereg.EXPECTED_BRANCH,
            )
    finally:
        snapshot.close()
        context.close()


@pytest.mark.parametrize("identity", ["G9CB-2", "G9CB-3"])
def test_builder_retained_predecessor_residue_rejects_restored_timestamp_drift(
    tmp_path: Path,
    identity: str,
) -> None:
    results = _materialize_closed_phase_state(tmp_path)
    context = builder._PublicationContext(tmp_path)
    row = next(
        item
        for item in builder.prereg.expected_failed_predecessor_attempts()
        if item["identity"] == identity
    )
    stage = tmp_path / row["residue"]["slot1_stage"]["path"]
    try:
        builder._validate_closed_entry_phase(
            context,
            builder.Q8_PREREGISTRATION_PUBLICATION,
        )
        before = os.stat(stage, follow_symlinks=False)
        time.sleep(0.01)
        marker = stage / "ephemeral-marker"
        marker.write_bytes(b"ephemeral\n")
        marker.unlink()
        after = os.stat(stage, follow_symlinks=False)
        assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
        assert after.st_mtime_ns != before.st_mtime_ns
        assert not tuple(stage.iterdir())
        assert tuple(sorted(path.name for path in results.iterdir())) == (
            context.entries
        )
        with pytest.raises(
            builder.TerminalG9CB8Failure,
            match="timestamp|residue|directory|drift",
        ):
            builder._validate_closed_entry_phase(
                context,
                builder.Q8_PREREGISTRATION_PUBLICATION,
            )
    finally:
        context.close()


def test_real_production_checkpoints_enforce_exact_namespace_deltas(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    chain = _prepare_synthetic_a5_q5_p5_chain(tmp_path)
    claim, _publication = _publish_synthetic_claim_from_p5(chain)
    _commit_synthetic_claim(chain, claim)
    root = chain["root"]
    context = builder._PublicationContext(root)
    state = builder._ProductionStateMachine()
    stage_one = (
        root
        / "results/.gross9-structural-clock-g9cb8-worker-checkpoint-one"
    )
    stage_two = (
        root
        / "results/.gross9-structural-clock-g9cb8-worker-checkpoint-two"
    )
    transitions: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    fsynced_results = 0
    original_rebaseline = context._rebaseline
    original_fsync = builder.os.fsync

    def recorded_rebaseline(expected_entries: tuple[str, ...]) -> None:
        transitions.append((context.entries, tuple(expected_entries)))
        original_rebaseline(expected_entries)

    def recorded_fsync(descriptor: int) -> None:
        nonlocal fsynced_results
        if descriptor == context.results_fd:
            fsynced_results += 1
        original_fsync(descriptor)

    monkeypatch.setattr(context, "_rebaseline", recorded_rebaseline)
    monkeypatch.setattr(builder.os, "fsync", recorded_fsync)

    def advance(checkpoint: str) -> None:
        state.advance(
            checkpoint,
            lambda: builder._validate_production_namespace(
                context,
                checkpoint,
                stage_one,
                stage_two,
            ),
        )

    def stage_outputs(stage: Path) -> None:
        stage_fd = context.stage_descriptors[stage.name]
        for name in (
            builder._STAGED_CSV_NAME,
            builder._STAGED_CORE_NAME,
            builder._STAGED_RECEIPT_NAME,
        ):
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                0o600,
                dir_fd=stage_fd,
            )
            try:
                os.write(descriptor, name.encode("ascii"))
                os.fchmod(descriptor, 0o400)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        os.fsync(stage_fd)

    try:
        advance("C8_PRODUCTION_PREFLIGHT")
        context.probe()
        advance("CAPABILITY_PROBE_COMPLETE")
        builder._create_stage_directory(
            stage_one,
            stage_one.parent,
            publication_context=context,
        )
        advance("SLOT1_PREPARED")
        context.publish(builder.SENTINEL_PATH, b"synthetic sentinel\n")
        advance("SENTINEL_LINKED")
        context.publish(
            builder.WORKER_LEDGER_PATHS[0],
            b"synthetic ledger one\n",
        )
        advance("PASS1_LEDGER_LINKED")
        stage_outputs(stage_one)
        advance("PASS1_OUTPUT_READY")
        builder._cleanup_successful_stage(
            stage_one,
            stage_one.parent,
            publication_context=context,
        )
        builder._create_stage_directory(
            stage_two,
            stage_two.parent,
            publication_context=context,
        )
        advance("SLOT_TRANSITION")
        context.publish(
            builder.WORKER_LEDGER_PATHS[1],
            b"synthetic ledger two\n",
        )
        advance("PASS2_LEDGER_LINKED")
        stage_outputs(stage_two)
        advance("PASS2_OUTPUT_READY")
        context.publish(builder.CSV_PATH, b"synthetic csv gzip\n")
        advance("CANONICAL_CSV_LINKED")
        context.publish(builder.MANIFEST_PATH, b"synthetic manifest\n")
        advance("MANIFEST_LINKED_LAST")
        builder._cleanup_successful_stage(
            stage_two,
            stage_two.parent,
            publication_context=context,
        )
        advance("FINAL_CLEANUP")
        state.require_complete()
        assert state.current == "FINAL_CLEANUP"
        one_leaf_deltas = [
            (before, after)
            for before, after in transitions
            if before != after
        ]
        assert len(one_leaf_deltas) == 11
        assert all(
            len(set(before) ^ set(after)) == 1
            for before, after in one_leaf_deltas
        )
        assert fsynced_results >= len(one_leaf_deltas)
    finally:
        context.close()


@pytest.mark.parametrize(
    "checkpoint",
    [
        "SLOT1_PREPARED",
        "SENTINEL_LINKED",
        "PASS1_LEDGER_LINKED",
        "PASS1_OUTPUT_READY",
        "SLOT_TRANSITION",
        "PASS2_LEDGER_LINKED",
        "PASS2_OUTPUT_READY",
        "CANONICAL_CSV_LINKED",
        "MANIFEST_LINKED_LAST",
    ],
)
def test_active_stage_retained_descriptor_must_match_canonical_edge_at_checkpoint(
    tmp_path: Path,
    checkpoint: str,
) -> None:
    publication_counts = {
        "SLOT1_PREPARED": 0,
        "SENTINEL_LINKED": 1,
        "PASS1_LEDGER_LINKED": 2,
        "PASS1_OUTPUT_READY": 2,
        "SLOT_TRANSITION": 2,
        "PASS2_LEDGER_LINKED": 3,
        "PASS2_OUTPUT_READY": 3,
        "CANONICAL_CSV_LINKED": 4,
        "MANIFEST_LINKED_LAST": 5,
    }
    output_ready = checkpoint in {
        "PASS1_OUTPUT_READY",
        "PASS2_OUTPUT_READY",
        "CANONICAL_CSV_LINKED",
        "MANIFEST_LINKED_LAST",
    }
    results = _materialize_closed_phase_state(
        tmp_path,
        (builder.PREREGISTRATION_PATH, builder.CLAIM_PATH),
    )
    publication_order = (
        builder.SENTINEL_PATH,
        *builder.WORKER_LEDGER_PATHS,
        builder.CSV_PATH,
        builder.MANIFEST_PATH,
    )
    for relative in publication_order[: publication_counts[checkpoint]]:
        target = tmp_path / relative
        target.write_bytes(f"synthetic checkpoint: {relative}\n".encode())
        target.chmod(0o444)
    stage_one = (
        results / ".gross9-structural-clock-g9cb8-worker-edge-one"
    )
    stage_two = (
        results / ".gross9-structural-clock-g9cb8-worker-edge-two"
    )
    active_stage = (
        stage_two
        if checkpoint
        in {
            "SLOT_TRANSITION",
            "PASS2_LEDGER_LINKED",
            "PASS2_OUTPUT_READY",
            "CANONICAL_CSV_LINKED",
            "MANIFEST_LINKED_LAST",
        }
        else stage_one
    )
    active_stage.mkdir(mode=0o700)
    if output_ready:
        for name in (
            builder._STAGED_CSV_NAME,
            builder._STAGED_CORE_NAME,
            builder._STAGED_RECEIPT_NAME,
        ):
            target = active_stage / name
            target.write_bytes(f"synthetic stage output: {name}\n".encode())
            target.chmod(0o400)
    context = builder._PublicationContext(tmp_path)
    stage_fd = _register_test_stage(context, active_stage)
    try:
        builder._validate_production_namespace(
            context,
            checkpoint,
            stage_one,
            stage_two,
        )
        original = os.fstat(stage_fd)
        detached = tmp_path / f"detached-{active_stage.name}"
        active_stage.rename(detached)
        if not output_ready:
            detached.rmdir()
        active_stage.mkdir(mode=0o700)
        if output_ready:
            for name in (
                builder._STAGED_CSV_NAME,
                builder._STAGED_CORE_NAME,
                builder._STAGED_RECEIPT_NAME,
            ):
                target = active_stage / name
                target.write_bytes(
                    f"replacement stage output: {name}\n".encode()
                )
                target.chmod(0o400)
        replacement = os.stat(active_stage, follow_symlinks=False)
        assert (replacement.st_dev, replacement.st_ino) != (
            original.st_dev,
            original.st_ino,
        )
        _rebaseline_restored_results_timestamp(context)
        with pytest.raises(
            builder.TerminalG9CB8Failure,
            match="stage|identity|edge|component|retained",
        ):
            builder._validate_production_namespace(
                context,
                checkpoint,
                stage_one,
                stage_two,
            )
    finally:
        context.close()


def test_active_stage_retained_descriptor_rejects_restored_timestamp_drift(
    tmp_path: Path,
) -> None:
    results = _materialize_closed_phase_state(
        tmp_path,
        (builder.PREREGISTRATION_PATH, builder.CLAIM_PATH),
    )
    stage_one = (
        results / ".gross9-structural-clock-g9cb8-worker-timestamp-one"
    )
    stage_two = (
        results / ".gross9-structural-clock-g9cb8-worker-timestamp-two"
    )
    stage_one.mkdir(mode=0o700)
    context = builder._PublicationContext(tmp_path)
    stage_fd = _register_test_stage(context, stage_one)
    try:
        builder._validate_production_namespace(
            context,
            "SLOT1_PREPARED",
            stage_one,
            stage_two,
        )
        before = os.fstat(stage_fd)
        time.sleep(0.01)
        marker_fd = os.open(
            "ephemeral-marker",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
            dir_fd=stage_fd,
        )
        os.close(marker_fd)
        os.unlink("ephemeral-marker", dir_fd=stage_fd)
        after = os.fstat(stage_fd)
        assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
        assert after.st_mtime_ns != before.st_mtime_ns
        assert builder._directory_entries(stage_fd) == ()
        with pytest.raises(
            builder.TerminalG9CB8Failure,
            match="stage|timestamp|drift",
        ):
            builder._validate_production_namespace(
                context,
                "SLOT1_PREPARED",
                stage_one,
                stage_two,
            )
    finally:
        context.close()


@pytest.mark.parametrize(
    ("checkpoint", "stage_name", "ledger_paths"),
    [
        (
            "PASS1_LEDGER_LINKED",
            ".gross9-structural-clock-g9cb8-worker-ledger-one",
            (builder.SENTINEL_PATH, builder.WORKER_LEDGER_PATHS[0]),
        ),
        (
            "PASS2_LEDGER_LINKED",
            ".gross9-structural-clock-g9cb8-worker-ledger-two",
            (
                builder.SENTINEL_PATH,
                builder.WORKER_LEDGER_PATHS[0],
                builder.WORKER_LEDGER_PATHS[1],
            ),
        ),
    ],
)
def test_ledger_linked_checkpoint_rejects_any_stage_output_ready_collapse(
    tmp_path: Path,
    checkpoint: str,
    stage_name: str,
    ledger_paths: tuple[Path, ...],
) -> None:
    results = _materialize_closed_phase_state(
        tmp_path,
        (builder.PREREGISTRATION_PATH, builder.CLAIM_PATH),
    )
    for relative in ledger_paths:
        target = tmp_path / relative
        target.write_bytes(f"synthetic linked state: {relative}\n".encode())
        target.chmod(0o444)
    active_stage = results / stage_name
    active_stage.mkdir(mode=0o700)
    for name in (
        builder._STAGED_CSV_NAME,
        builder._STAGED_CORE_NAME,
        builder._STAGED_RECEIPT_NAME,
    ):
        target = active_stage / name
        target.write_bytes(f"premature output: {name}\n".encode())
        target.chmod(0o400)
    stage_one = (
        active_stage
        if checkpoint == "PASS1_LEDGER_LINKED"
        else results / ".gross9-structural-clock-g9cb8-worker-unused-one"
    )
    stage_two = (
        active_stage
        if checkpoint == "PASS2_LEDGER_LINKED"
        else results / ".gross9-structural-clock-g9cb8-worker-unused-two"
    )
    context = builder._PublicationContext(tmp_path)
    _register_test_stage(context, active_stage)
    ledger_leaf = (
        builder.WORKER_LEDGER_PATHS[0].name
        if checkpoint == "PASS1_LEDGER_LINKED"
        else builder.WORKER_LEDGER_PATHS[1].name
    )
    setattr(context, "worker_ledger_checkpoint_evidence", {ledger_leaf})
    assert getattr(context, "worker_ledger_checkpoint_evidence") == {
        ledger_leaf
    }
    try:
        with pytest.raises(
            builder.TerminalG9CB8Failure,
            match="stage|inventory|output|checkpoint",
        ):
            builder._validate_production_namespace(
                context,
                checkpoint,
                stage_one,
                stage_two,
            )
    finally:
        context.close()


def test_stage_file_and_cleanup_transients_rebaseline_exact_inventories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    context = builder._PublicationContext(tmp_path)
    stage = (
        results
        / ".gross9-structural-clock-g9cb8-worker-stage-transients"
    )
    other = "results/.gross9-structural-clock-g9cb8-worker-stage-other"
    builder._create_stage_directory(
        stage,
        results,
        publication_context=context,
    )
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    repository_fd = os.open(tmp_path, flags)
    results_fd = os.open("results", flags, dir_fd=repository_fd)
    filesystem_root_fd = os.open("/", flags)
    ledger_fd = os.open(
        ".",
        os.O_RDWR | os.O_TMPFILE | os.O_CLOEXEC,
        0o600,
        dir_fd=results_fd,
    )
    guard = builder._WorkerIsolationGuard(
        root=tmp_path.resolve(),
        own_stage=stage.relative_to(tmp_path).as_posix(),
        other_stage=other,
        ledger_paths=builder.WORKER_LEDGER_PATHS,
        repository_fd=repository_fd,
        results_fd=results_fd,
        filesystem_root_fd=filesystem_root_fd,
        ledger_fd=ledger_fd,
    )
    guard.authorize_directory_sync(stage)
    stage_fd = guard.stage_fd
    assert stage_fd is not None
    prior_guard = builder._ACTIVE_WORKER_GUARD
    inventories: list[tuple[str, ...]] = []
    creation_tokens: list[tuple[int, ...]] = []
    try:
        builder._ACTIVE_WORKER_GUARD = guard
        for name in (
            builder._STAGED_CSV_NAME,
            builder._STAGED_CORE_NAME,
            builder._STAGED_RECEIPT_NAME,
        ):
            builder._write_exclusive_guarded_file(
                stage / name,
                f"synthetic {name}\n".encode(),
                mode=0o400,
                sync_directory=stage,
            )
            inventories.append(tuple(sorted(os.listdir(stage_fd))))
            stage_token = guard.stage_token
            assert stage_token is not None
            creation_tokens.append(stage_token)
        assert inventories == [
            (builder._STAGED_CSV_NAME,),
            tuple(
                sorted(
                    (builder._STAGED_CSV_NAME, builder._STAGED_CORE_NAME)
                )
            ),
            tuple(
                sorted(
                    (
                        builder._STAGED_CSV_NAME,
                        builder._STAGED_CORE_NAME,
                        builder._STAGED_RECEIPT_NAME,
                    )
                )
            ),
        ]
        assert all(
            stat.S_IMODE(
                os.stat(
                    name,
                    dir_fd=stage_fd,
                    follow_symlinks=False,
                ).st_mode
            )
            == 0o400
            for name in inventories[-1]
        )
        assert len(creation_tokens) == 3
        assert all(token is not None for token in creation_tokens)
        assert creation_tokens[-1] == builder._descriptor_token(
            os.fstat(stage_fd)
        )
    finally:
        builder._ACTIVE_WORKER_GUARD = prior_guard
    cleanup_rebaselines: list[tuple[tuple[str, ...], tuple[int, ...]]] = []
    original_require_stage = context.require_stage

    def recorded_require_stage(
        active_stage: Path,
        expected_entries: tuple[str, ...],
        *,
        authorized_delta: bool = False,
    ) -> None:
        original_require_stage(
            active_stage,
            expected_entries,
            authorized_delta=authorized_delta,
        )
        if authorized_delta:
            token = context.stage_tokens[active_stage.name]
            assert token == builder._descriptor_token(
                os.fstat(context.stage_descriptors[active_stage.name])
            )
            cleanup_rebaselines.append((expected_entries, token))

    monkeypatch.setattr(context, "require_stage", recorded_require_stage)
    try:
        builder._cleanup_successful_stage(
            stage,
            results,
            publication_context=context,
        )
        assert stage.name not in os.listdir(context.results_fd)
        assert [entries for entries, _token in cleanup_rebaselines] == [
            tuple(
                sorted(
                    (
                        builder._STAGED_CSV_NAME,
                        builder._STAGED_CORE_NAME,
                        builder._STAGED_RECEIPT_NAME,
                    )
                )
            ),
            tuple(
                sorted((builder._STAGED_CORE_NAME, builder._STAGED_RECEIPT_NAME))
            ),
            (builder._STAGED_RECEIPT_NAME,),
            (),
        ]
        assert len(cleanup_rebaselines) == 4
        assert stage.name not in context.stage_descriptors
        assert stage.name not in context.stage_tokens
        assert stage.name not in context.stage_entries
    finally:
        context.close()
        if guard.stage_fd is not None:
            os.close(guard.stage_fd)
        for descriptor in (
            ledger_fd,
            filesystem_root_fd,
            results_fd,
            repository_fd,
        ):
            os.close(descriptor)


def test_stage_cleanup_transient_rejects_canonical_edge_substitution(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    context = builder._PublicationContext(tmp_path)
    stage = (
        results
        / ".gross9-structural-clock-g9cb8-worker-cleanup-substitution"
    )
    builder._create_stage_directory(
        stage,
        results,
        publication_context=context,
    )
    stage_fd = context.stage_descriptors[stage.name]
    expected_entries = tuple(
        sorted(
            (
                builder._STAGED_CSV_NAME,
                builder._STAGED_CORE_NAME,
                builder._STAGED_RECEIPT_NAME,
            )
        )
    )
    for name in expected_entries:
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
            dir_fd=stage_fd,
        )
        try:
            os.write(descriptor, f"synthetic cleanup: {name}\n".encode())
            os.fchmod(descriptor, 0o400)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    os.fsync(stage_fd)
    context.require_stage(stage, expected_entries, authorized_delta=True)
    original = os.fstat(stage_fd)
    detached = tmp_path / "detached-cleanup-stage"
    stage.rename(detached)
    stage.mkdir(mode=0o700)
    replacement = os.stat(stage, follow_symlinks=False)
    assert (replacement.st_dev, replacement.st_ino) != (
        original.st_dev,
        original.st_ino,
    )
    _rebaseline_restored_results_timestamp(context)
    try:
        with pytest.raises(
            builder.TerminalG9CB8Failure,
            match="stage|identity|edge|retained",
        ):
            builder._cleanup_successful_stage(
                stage,
                results,
                publication_context=context,
            )
    finally:
        context.close()


def test_preregistration_only_snapshot_succeeds_without_opening_absent_claim(
    tmp_path: Path,
) -> None:
    chain = _prepare_synthetic_a5_q5_p5_chain(tmp_path)
    snapshot, pairs = builder._preauthenticate_parent_snapshot(
        chain["root"],
        chain["preregistration"],
        content_mode=builder._PREREGISTRATION_ONLY,
    )
    try:
        assert builder.PREREGISTRATION_PATH.as_posix() in snapshot
        assert builder.CLAIM_PATH.as_posix() not in snapshot
        assert set(pairs) == set(snapshot)
        context = builder._PublicationContext(chain["root"])
        try:
            builder._validate_closed_entry_phase(
                context,
                builder.P8_CLAIM_PREFLIGHT,
            )
        finally:
            context.close()
    finally:
        snapshot.close()


def test_preregistration_plus_claim_snapshot_succeeds_at_c5(
    tmp_path: Path,
) -> None:
    chain = _prepare_synthetic_a5_q5_p5_chain(tmp_path)
    claim, _publication = _publish_synthetic_claim_from_p5(chain)
    _commit_synthetic_claim(chain, claim)
    snapshot, pairs = builder._preauthenticate_parent_snapshot(
        chain["root"],
        chain["preregistration"],
        content_mode=builder._PREREGISTRATION_PLUS_CLAIM,
    )
    context = builder._PublicationContext(chain["root"])
    try:
        assert builder.PREREGISTRATION_PATH.as_posix() in snapshot
        assert builder.CLAIM_PATH.as_posix() in snapshot
        builder._validate_closed_entry_phase(
            context,
            builder.C8_PRODUCTION_PREFLIGHT,
        )
        builder._final_parent_snapshot_recheck(
            chain["root"],
            snapshot,
            pairs,
            context,
            builder.C8_PRODUCTION_PREFLIGHT,
            builder.prereg.EXPECTED_BRANCH,
        )
    finally:
        snapshot.close()
        context.close()


def test_committed_verifier_rejects_synthetic_default_and_accepts_profile(
    tmp_path: Path,
) -> None:
    chain = _prepare_synthetic_a5_q5_p5_chain(tmp_path)
    security_profile = _synthetic_security_profile(chain)
    binding = builder.create_claim_only(
        chain["root"],
        expected_security_profile=security_profile,
    )
    claim = _read_synthetic_claim(chain)
    assert binding == _synthetic_claim_binding(chain, claim)
    c5 = _commit_synthetic_claim(chain, claim)
    security_profile["claim_commit"] = c5
    production = builder.produce_one_shot(
        chain["root"],
        synthetic_input=_write_synthetic_chain_input(chain),
        expected_security_profile=security_profile,
    )
    d5 = _commit_synthetic_d5(chain)
    security_profile["publication_commit"] = d5
    with pytest.raises(
        builder.TerminalG9CB8Failure,
        match="official A8/Q8/P8 security topology differs",
    ):
        builder.validate_committed_publication(chain["root"])
    verified = builder.validate_committed_publication(
        chain["root"],
        expected_security_profile=security_profile,
    )
    for commit_key in ("claim_commit", "publication_commit"):
        mutated_profile = copy.deepcopy(security_profile)
        mutated_profile[commit_key] = "f" * 40
        with pytest.raises(
            builder.TerminalG9CB8Failure,
            match=f"committed topology differs.*{commit_key}",
        ):
            builder.validate_committed_publication(
                chain["root"],
                expected_security_profile=mutated_profile,
            )
    assert production["identity"] == builder.IDENTITY
    assert verified["identity"] == builder.IDENTITY
    assert verified["protocol_implementation_commit"] == chain["q5"]
    assert verified["preregistration_seal_commit"] == chain["p5"]
    assert verified["claim_commit"] == c5
    assert verified["publication_commit"] == d5
    assert security_profile["protocol_implementation_commit"] == chain["q5"]
    assert security_profile["preregistration_seal_commit"] == chain["p5"]
    assert security_profile["claim_commit"] == c5
    assert security_profile["publication_commit"] == d5
    assert verified["csv_gzip_sha256"] == production["csv_gzip_sha256"]
    preregistration_at_d5 = subprocess.run(
        [
            "git",
            "show",
            f"HEAD:{builder.PREREGISTRATION_PATH.as_posix()}",
        ],
        cwd=chain["root"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    assert preregistration_at_d5 == chain["preregistration_raw"]


def test_temporary_git_lifecycle_uses_canonical_security_path_with_explicit_expectations(
    tmp_path: Path,
) -> None:
    chain = _prepare_synthetic_a5_q5_p5_chain(tmp_path)
    root = chain["root"]
    expected_security_profile = _synthetic_security_profile(chain)
    security_calls: list[str] = []
    watched_codes = {
        builder.validate_preregistration.__code__: "validate_preregistration",
        builder._validated_expected_security_profile.__code__: (
            "_validated_expected_security_profile"
        ),
        builder._preauthenticate_parent_snapshot.__code__: (
            "_preauthenticate_parent_snapshot"
        ),
        builder._validate_committed_publication_topology.__code__: (
            "_validate_committed_publication_topology"
        ),
    }
    for forbidden_name in (
        "_is_synthetic_security_preregistration",
        "_validate_synthetic_preregistration_snapshot",
    ):
        forbidden = getattr(builder, forbidden_name, None)
        if forbidden is not None:
            watched_codes[forbidden.__code__] = forbidden_name

    def record_security_call(
        frame: Any,
        event: str,
        _argument: Any,
    ) -> None:
        if event == "call" and frame.f_code in watched_codes:
            security_calls.append(watched_codes[frame.f_code])

    prior_profile = sys.getprofile()
    sys.setprofile(record_security_call)
    try:
        binding = builder.create_claim_only(
            root,
            expected_security_profile=expected_security_profile,
        )
        claim = _read_synthetic_claim(chain)
        assert binding == _synthetic_claim_binding(chain, claim)
        c5 = _commit_synthetic_claim(chain, claim)
        expected_security_profile["claim_commit"] = c5
        production = builder.produce_one_shot(
            root,
            synthetic_input=_write_synthetic_chain_input(chain),
            expected_security_profile=expected_security_profile,
        )
        d5 = _commit_synthetic_d5(chain)
        expected_security_profile["publication_commit"] = d5
        verified = builder.validate_committed_publication(
            root,
            expected_security_profile=expected_security_profile,
        )
    finally:
        sys.setprofile(prior_profile)

    assert production["identity"] == builder.IDENTITY
    assert verified["protocol_implementation_commit"] == chain["q5"]
    assert verified["preregistration_seal_commit"] == chain["p5"]
    assert verified["claim_commit"] == c5
    assert verified["publication_commit"] == d5
    assert "_is_synthetic_security_preregistration" not in security_calls
    assert "_validate_synthetic_preregistration_snapshot" not in security_calls
    assert security_calls.count("_validated_expected_security_profile") >= 3
    assert security_calls.count("validate_preregistration") >= 3
    assert security_calls.count("_preauthenticate_parent_snapshot") >= 3
    assert security_calls.count("_validate_committed_publication_topology") == 1


def test_results_inventory_drift_cannot_be_rebaselined_as_a_closed_phase(
    tmp_path: Path,
) -> None:
    chain = _prepare_synthetic_a5_q5_p5_chain(tmp_path)
    context = builder._PublicationContext(chain["root"])
    try:
        builder._validate_closed_entry_phase(
            context,
            builder.P8_CLAIM_PREFLIGHT,
        )
        unexpected = (
            chain["root"] / "results/.g9cb8-otmpfile-probe-stale"
        )
        unexpected.write_bytes(b"unexpected\n")
        with pytest.raises(
            builder.TerminalG9CB8Failure,
            match="inventory|entry|path-state",
        ):
            builder._validate_closed_entry_phase(
                context,
                builder.P8_CLAIM_PREFLIGHT,
            )
    finally:
        context.close()


def test_builder_tracked_results_projection_uses_first_component() -> None:
    assert builder._tracked_results_top_level_entries(
        "results/direct.json\n"
        "results/nested/child/direct.json\n"
        "results/collision/child/nested\n"
    ) == {"direct.json", "nested", "collision"}


@pytest.mark.parametrize(
    "path_text",
    ["other/file", "results", "results/", "results/../leaf", "/results/leaf"],
)
def test_builder_tracked_results_projection_rejects_malformed_paths(
    path_text: str,
) -> None:
    with pytest.raises(
        builder.TerminalG9CB8Failure,
        match="malformed tracked results path",
    ):
        builder._tracked_results_top_level_entries(path_text)


@pytest.mark.parametrize(
    "phase",
    [
        builder.Q8_PREREGISTRATION_PUBLICATION,
        builder.P8_CLAIM_PREFLIGHT,
        builder.C8_PRODUCTION_PREFLIGHT,
        builder.D8_COMMITTED_VERIFICATION,
    ],
)
def test_every_builder_phase_rejects_missing_tracked_top_level_entry(
    tmp_path: Path,
    phase: str,
) -> None:
    phase_presence = {
        builder.Q8_PREREGISTRATION_PUBLICATION: (),
        builder.P8_CLAIM_PREFLIGHT: (builder.PREREGISTRATION_PATH,),
        builder.C8_PRODUCTION_PREFLIGHT: (
            builder.PREREGISTRATION_PATH,
            builder.CLAIM_PATH,
        ),
        builder.D8_COMMITTED_VERIFICATION: (
            builder.PREREGISTRATION_PATH,
            builder.CLAIM_PATH,
            builder.SENTINEL_PATH,
            *builder.WORKER_LEDGER_PATHS,
            builder.CSV_PATH,
            builder.MANIFEST_PATH,
        ),
    }
    results = _materialize_closed_phase_state(tmp_path, phase_presence[phase])
    tracked = results / "tracked-top-level" / "child.json"
    tracked.parent.mkdir()
    tracked.write_bytes(b"tracked\n")
    _git_for_q5_test(tmp_path, "init", "-q")
    _git_for_q5_test(tmp_path, "config", "user.email", "missing@example.invalid")
    _git_for_q5_test(tmp_path, "config", "user.name", "Missing Inventory")
    _git_for_q5_test(tmp_path, "add", "results/tracked-top-level")
    _git_for_q5_test(tmp_path, "commit", "-qm", "track nested result")
    tracked.unlink()
    tracked.parent.rmdir()
    context = builder._PublicationContext(tmp_path)
    try:
        with pytest.raises(builder.TerminalG9CB8Failure, match="inventory"):
            builder._validate_closed_entry_phase(context, phase)
    finally:
        context.close()


@pytest.mark.parametrize(
    ("phase", "present"),
    [
        (builder.Q8_PREREGISTRATION_PUBLICATION, ()),
        (builder.P8_CLAIM_PREFLIGHT, (builder.PREREGISTRATION_PATH,)),
        (
            builder.C8_PRODUCTION_PREFLIGHT,
            (builder.PREREGISTRATION_PATH, builder.CLAIM_PATH),
        ),
        (
            builder.D8_COMMITTED_VERIFICATION,
            (
                builder.PREREGISTRATION_PATH,
                builder.CLAIM_PATH,
                builder.SENTINEL_PATH,
                *builder.WORKER_LEDGER_PATHS,
                builder.CSV_PATH,
                builder.MANIFEST_PATH,
            ),
        ),
    ],
)
def test_each_closed_entry_phase_accepts_only_its_exact_active_topology(
    tmp_path: Path,
    phase: str,
    present: tuple[Path, ...],
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    tracked_direct = results / "tracked-direct.json"
    tracked_direct.write_bytes(b"tracked direct\n")
    tracked_nested = results / "tracked-nested" / builder.PREREGISTRATION_PATH.name
    tracked_nested.parent.mkdir()
    tracked_nested.write_bytes(b"tracked nested\n")
    _git_for_q5_test(tmp_path, "init", "-q")
    _git_for_q5_test(tmp_path, "config", "user.email", "phase@example.invalid")
    _git_for_q5_test(tmp_path, "config", "user.name", "Phase Inventory")
    _git_for_q5_test(tmp_path, "add", "results")
    _git_for_q5_test(tmp_path, "commit", "-qm", "tracked results inventory")
    for row in builder.prereg.expected_failed_predecessor_attempts():
        slot_one = tmp_path / row["residue"]["slot1_stage"]["path"]
        slot_one.mkdir(mode=0o700)
    for relative in present:
        target = tmp_path / relative
        target.write_bytes(b"synthetic active metadata\n")
        target.chmod(0o444)
    context = builder._PublicationContext(tmp_path)
    try:
        builder._validate_closed_entry_phase(context, phase)
        if phase == builder.D8_COMMITTED_VERIFICATION:
            target = tmp_path / builder.MANIFEST_PATH
            target.chmod(0o644)
        else:
            candidates = (
                builder.PREREGISTRATION_PATH,
                builder.CLAIM_PATH,
                builder.SENTINEL_PATH,
            )
            target = tmp_path / next(
                candidate for candidate in candidates if candidate not in present
            )
            target.write_bytes(b"forbidden phase state\n")
        with pytest.raises(
            builder.TerminalG9CB8Failure,
            match="path-state|mode/type|inventory",
        ):
            builder._validate_closed_entry_phase(context, phase)
    finally:
        context.close()


@pytest.mark.parametrize(
    ("identity", "mutation"),
    [
        ("G9CB-2", "permanent-output"),
        ("G9CB-3", "permanent-output"),
        ("G9CB-3", "bytecode"),
        ("G9CB-4", "permanent-output"),
        ("G9CB-4", "publication-stage"),
        ("G9CB-4", "worker-stage"),
    ],
)
def test_complete_predecessor_absence_and_residue_inventory_is_behavioral(
    tmp_path: Path,
    identity: str,
    mutation: str,
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    attempts = builder.prereg.expected_failed_predecessor_attempts()
    closures = builder.prereg.expected_failed_predecessor_closures()
    for row in attempts:
        slot_one = tmp_path / row["residue"]["slot1_stage"]["path"]
        slot_one.mkdir(mode=0o700)
    if identity in {"G9CB-2", "G9CB-3"}:
        row = next(row for row in attempts if row["identity"] == identity)
    else:
        row = closures[0]
    if mutation == "permanent-output":
        target = tmp_path / row["permanently_absent_outputs"][0]
    elif mutation == "bytecode":
        target = tmp_path / row["residue"]["bytecode_cache"]["path"]
    elif mutation == "publication-stage":
        target = results / ".gross9_structural_clock_bundle_g9cb4_x.stage-y"
    else:
        target = results / ".gross9-structural-clock-g9cb4-worker-x"
    if mutation in {"publication-stage", "worker-stage", "bytecode"}:
        target.mkdir(parents=True)
    else:
        target.write_bytes(b"synthetic forbidden predecessor residue\n")
    results_fd = os.open(
        results,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    try:
        with pytest.raises(
            builder.TerminalG9CB8Failure,
            match=identity,
        ):
            builder._validate_failed_predecessor_permanent_state(
                results_fd,
                attempts,
                closures,
            )
    finally:
        os.close(results_fd)


@pytest.mark.parametrize(
    ("identity", "mutation", "output_index"),
    [
        *[
            (identity, "permanent-output", output_index)
            for identity in ("G9CB-5", "G9CB-6")
            for output_index in range(7)
        ],
        ("G9CB-5", "bytecode", None),
        ("G9CB-5", "publication-stage", None),
        ("G9CB-5", "worker-stage", None),
        ("G9CB-6", "bytecode", None),
        ("G9CB-6", "capability-probe", None),
        ("G9CB-6", "publication-stage", None),
        ("G9CB-6", "worker-stage", None),
    ],
)
def test_each_prepublication_closure_absence_and_residue_is_behavioral(
    tmp_path: Path,
    identity: str,
    mutation: str,
    output_index: int | None,
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    attempts = builder.prereg.expected_failed_predecessor_attempts()
    closures = builder.prereg.expected_failed_predecessor_closures()
    prepublication = (
        builder.prereg.expected_failed_predecessor_prepublication_closures()
    )
    for row in attempts:
        slot_one = tmp_path / row["residue"]["slot1_stage"]["path"]
        slot_one.mkdir(mode=0o700)
    row = next(row for row in prepublication if row["identity"] == identity)
    suffix = identity.lower().replace("-", "")
    if mutation == "permanent-output":
        assert output_index is not None
        target = tmp_path / row["permanently_absent_outputs"][output_index]
        target.write_bytes(b"synthetic forbidden predecessor output\n")
    elif mutation == "bytecode":
        target = tmp_path / row["residue"]["bytecode_cache"]["path"]
        target.mkdir(parents=True)
    elif mutation == "publication-stage":
        target = results / (
            f".gross9_structural_clock_bundle_{suffix}_mutation.stage-test"
        )
        target.mkdir()
    elif mutation == "worker-stage":
        target = results / f".gross9-structural-clock-{suffix}-worker-test"
        target.mkdir()
    else:
        assert mutation == "capability-probe" and identity == "G9CB-6"
        target = results / f".{suffix}-otmpfile-probe-test"
        target.write_bytes(b"synthetic forbidden capability probe\n")

    results_fd = os.open(
        results,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    try:
        with pytest.raises(
            builder.TerminalG9CB8Failure,
            match=identity,
        ):
            builder._validate_failed_predecessor_permanent_state(
                results_fd,
                attempts,
                closures,
                prepublication,
            )
    finally:
        os.close(results_fd)


def test_unexpected_initial_results_entry_is_not_a_valid_rebaseline(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    for row in builder.prereg.expected_failed_predecessor_attempts():
        (tmp_path / row["residue"]["slot1_stage"]["path"]).mkdir(mode=0o700)
    (results / ".g9cb8-unauthorized-helper-residue").write_bytes(
        b"unexpected\n"
    )
    context = builder._PublicationContext(tmp_path)
    try:
        with pytest.raises(
            builder.TerminalG9CB8Failure,
            match="inventory|entry|path-state",
        ):
            builder._validate_closed_entry_phase(
                context,
                builder.Q8_PREREGISTRATION_PUBLICATION,
            )
    finally:
        context.close()


@pytest.mark.parametrize(
    "phase",
    [
        builder.Q8_PREREGISTRATION_PUBLICATION,
        builder.P8_CLAIM_PREFLIGHT,
        builder.C8_PRODUCTION_PREFLIGHT,
        builder.D8_COMMITTED_VERIFICATION,
    ],
)
@pytest.mark.parametrize(
    "unauthorized_leaf",
    [
        ".g9cb8-otmpfile-probe-stale",
        f".{builder.SENTINEL_PATH.name}.stage-stale",
        f".{builder.WORKER_LEDGER_PATHS[0].name}.stage-stale",
        f".{builder.CSV_PATH.name}.stage-stale",
        f".{builder.MANIFEST_PATH.name}.stage-stale",
        ".gross9-structural-clock-g9cb8-helper-stale",
        "gross9_structural_clock_bundle_g9cb8_named-staging-residue",
    ],
)
def test_every_closed_phase_rejects_unauthorized_g9cb8_helper_residue(
    tmp_path: Path,
    phase: str,
    unauthorized_leaf: str,
) -> None:
    phase_presence = {
        builder.Q8_PREREGISTRATION_PUBLICATION: (),
        builder.P8_CLAIM_PREFLIGHT: (builder.PREREGISTRATION_PATH,),
        builder.C8_PRODUCTION_PREFLIGHT: (
            builder.PREREGISTRATION_PATH,
            builder.CLAIM_PATH,
        ),
        builder.D8_COMMITTED_VERIFICATION: (
            builder.PREREGISTRATION_PATH,
            builder.CLAIM_PATH,
            builder.SENTINEL_PATH,
            *builder.WORKER_LEDGER_PATHS,
            builder.CSV_PATH,
            builder.MANIFEST_PATH,
        ),
    }
    results = _materialize_closed_phase_state(
        tmp_path,
        phase_presence[phase],
    )
    (results / unauthorized_leaf).write_bytes(b"unauthorized residue\n")
    context = builder._PublicationContext(tmp_path)
    try:
        with pytest.raises(
            builder.TerminalG9CB8Failure,
            match="inventory|entry|path-state|residue|staging|probe",
        ):
            builder._validate_closed_entry_phase(context, phase)
    finally:
        context.close()


def test_actual_synthetic_claim_creation_uses_prepare_recheck_link_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    chain = _prepare_synthetic_a5_q5_p5_chain(tmp_path)
    events: list[str] = []
    verify_calls = 0
    in_claim_publication = False
    original_verify = builder._SecureBoundSnapshot.verify_final
    original_atomic = builder._atomic_link_write_once
    original_prepare = builder._open_unnamed_completed
    original_link = builder._link_unnamed_procfd

    def counted_verify(snapshot: builder._SecureBoundSnapshot) -> None:
        nonlocal verify_calls
        verify_calls += 1
        events.append("recheck")
        original_verify(snapshot)

    def recorded_atomic(path: Path, raw: bytes, **kwargs: Any) -> dict[str, Any]:
        nonlocal in_claim_publication
        assert path == chain["root"] / builder.CLAIM_PATH
        in_claim_publication = True
        try:
            return original_atomic(path, raw, **kwargs)
        finally:
            in_claim_publication = False

    def recorded_prepare(
        results_fd: int,
        raw: bytes,
        *,
        mode: int,
    ) -> tuple[int, os.stat_result]:
        prepared = original_prepare(results_fd, raw, mode=mode)
        if in_claim_publication:
            events.append("prepare")
        return prepared

    def recorded_link(
        unnamed_fd: int,
        results_fd: int,
        leaf: str,
    ) -> None:
        if leaf == builder.CLAIM_PATH.name:
            events.append("linked")
        original_link(unnamed_fd, results_fd, leaf)

    monkeypatch.setattr(
        builder._SecureBoundSnapshot,
        "verify_final",
        counted_verify,
    )
    monkeypatch.setattr(builder, "_atomic_link_write_once", recorded_atomic)
    monkeypatch.setattr(builder, "_open_unnamed_completed", recorded_prepare)
    monkeypatch.setattr(builder, "_link_unnamed_procfd", recorded_link)
    binding = builder.create_claim_only(
        chain["root"],
        expected_security_profile=_synthetic_security_profile(chain),
    )
    target = chain["root"] / builder.CLAIM_PATH
    payload = _read_synthetic_claim(chain)
    assert binding == _synthetic_claim_binding(chain, payload)
    assert events == ["prepare", "recheck", "linked"]
    assert verify_calls == 1
    assert stat.S_IMODE(target.stat().st_mode) == 0o444
    assert payload["preregistration"] == chain["preregistration_binding"]
    assert not any(
        (chain["root"] / path).exists()
        for path in (
            builder.SENTINEL_PATH,
            *builder.WORKER_LEDGER_PATHS,
            builder.CSV_PATH,
            builder.MANIFEST_PATH,
        )
    )


def test_parent_component_substitution_during_traversal_fails_closed(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "leaf").write_bytes(b"synthetic\n")
    snapshot = builder._SecureBoundSnapshot(tmp_path)
    try:
        snapshot._parent_fd("parent/leaf", True)
        parent.rename(tmp_path / "moved")
        parent.mkdir()
        (parent / "leaf").write_bytes(b"replacement\n")
        snapshot.open_initial("parent/leaf", True)
        with pytest.raises(
            builder.TerminalG9CB8Failure,
            match="directory graph|parent",
        ):
            snapshot.verify_final()
    finally:
        snapshot.close()


def test_bound_mode_is_authenticated_from_open_descriptor(
    tmp_path: Path,
) -> None:
    leaf = tmp_path / "leaf"
    leaf.write_bytes(b"synthetic\n")
    leaf.chmod(0o444)
    snapshot = builder._SecureBoundSnapshot(tmp_path)
    try:
        snapshot.open_initial("leaf", True)
        leaf.chmod(0o644)
        with pytest.raises(
            builder.TerminalG9CB8Failure,
            match="final|snapshot",
        ):
            snapshot.verify_final()
    finally:
        snapshot.close()


def test_at_empty_path_enoent_preserves_required_procfd_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "results").mkdir()
    calls: list[tuple[bytes | None, int]] = []
    original_linkat = builder._LIBC.linkat

    def linkat_with_empty_path_enoent(
        olddirfd: Any,
        oldpath: Any,
        newdirfd: Any,
        newpath: Any,
        flags: Any,
    ) -> int:
        source = getattr(oldpath, "value", oldpath)
        flag_value = int(getattr(flags, "value", flags))
        calls.append((source, flag_value))
        if source in (b"", None):
            builder.ctypes.set_errno(errno.ENOENT)
            return -1
        return int(
            original_linkat(
                olddirfd,
                oldpath,
                newdirfd,
                newpath,
                flags,
            )
        )

    monkeypatch.setattr(builder._LIBC, "linkat", linkat_with_empty_path_enoent)
    context = builder._PublicationContext(tmp_path)
    try:
        context.probe()
        empty_attempt = next(
            index
            for index, (source, _flags) in enumerate(calls)
            if source in (b"", None)
        )
        procfd_fallback = next(
            index
            for index, (source, _flags) in enumerate(calls)
            if index > empty_attempt
            and isinstance(source, bytes)
            and source.startswith(b"/proc/self/fd/")
        )
        assert empty_attempt < procfd_fallback
        assert context.probed is True
        assert tuple((tmp_path / "results").iterdir()) == ()
    finally:
        context.close()


def test_rank7_parity_counter_increments_at_each_comparison() -> None:
    dates = pd.date_range(builder.DOMAIN_START, periods=4, freq="5min")
    context = {
        "dates": dates,
        "matrix": np.arange(4, dtype=float).reshape(4, 1),
        "anchors": np.asarray([True, False, True, False]),
        "funding_leg": np.asarray([True, False, False, False]),
        "premium_leg": np.asarray([False, False, True, False]),
    }
    bundle = SimpleNamespace(
        valid_from=builder.DOMAIN_START,
        valid_until=builder.DOMAIN_END,
        models=(object(),) * 5,
    )

    class Runtime:
        @staticmethod
        def score_rank7_row(
            _bundle: object,
            row: np.ndarray,
            **_kwargs: Any,
        ) -> SimpleNamespace:
            signal = int(row[0])
            return SimpleNamespace(
                active=signal == 0,
                source="funding" if signal == 0 else "premium",
            )

    historical = np.asarray([True, False, False, False])
    counters = builder._empty_counters()
    observed = builder._rank7_bundle_activation_with_parity(
        Runtime,
        bundle,
        context,
        historical,
        counters,
        np,
        pd,
    )
    assert observed.tolist() == historical.tolist()
    assert counters["rows_used"][
        "rank7_bundle_activation_rows_scored"
    ] == 2
    assert counters["rows_used"][
        "rank7_bundle_parity_rows_compared"
    ] == 4

    drifted = historical.copy()
    drifted[2] = True
    failing_counters = builder._empty_counters()
    with pytest.raises(
        builder.TerminalG9CB8Failure,
        match="activation differ",
    ):
        builder._rank7_bundle_activation_with_parity(
            Runtime,
            bundle,
            context,
            drifted,
            failing_counters,
            np,
            pd,
        )
    assert failing_counters["rows_used"][
        "rank7_bundle_parity_rows_compared"
    ] == 3
