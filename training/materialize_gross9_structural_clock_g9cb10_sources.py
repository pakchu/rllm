"""One-shot G9CB-10 source-support materializer.

The production entry point deliberately has no command-line options.  Tests use
``materialize`` with a synthetic :class:`MaterializationConfig`; production
uses the frozen configuration returned by :func:`official_config`.
"""

from __future__ import annotations

import csv
import ctypes
import errno
import gzip
import hashlib
import io
import json
import math
import os
import stat
import subprocess
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Sequence

import numpy as np
import pandas as pd

from training.gross9_structural_clock_primitives import (
    _merge_aux,
    normalise_funding_history_frame,
    normalise_premium_index_frame,
)


class SourceSupportFailure(RuntimeError):
    """Terminal failure for the consumed G9CB-10 source-support identity."""


IDENTITY = "G9CB-10-SOURCE-SUPPORT"
VERSION = "gross9_structural_clock_bundle_g9cb10_source_support_v1"
BRANCH = "codex/gross9-structural-clock-bundle-20260731"
A9 = "98fe1e95708ad095cf0727363c32a89e7d03ead6"
T8 = "4188f35caa2c491f7b12f400d0815ea3a1a6144b"
S9 = "fe7dbb94e474d0d6f7ec3514ef79402e46c47c1e"
A10 = "6f9dd21554bc7b3282d0b2cbf7badee126e75c1a"
T9 = "a3ce195b02598b139068294089695b5d5dcd5044"

MARKET_SCHEMA = (
    "date", "open", "high", "low", "close", "volume",
    "quote_asset_volume", "number_of_trades", "taker_buy_base",
    "taker_buy_quote", "tic", "day", "dxy", "kimchi_premium",
    "usdkrw", "btckrw", "dxy_available", "kimchi_available",
    "usdkrw_available", "external_any_available", "dxy_zscore",
    "dxy_momentum", "kimchi_premium_zscore", "kimchi_premium_change",
    "usdkrw_zscore", "usdkrw_momentum",
)
OI_SCHEMA = ("date", "open_interest")
METRICS_SCHEMA = (
    "create_time", "symbol", "sum_open_interest",
    "sum_open_interest_value", "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio", "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)
RANK7_PROJECTION = (
    "date", "spot_close", "spot_rows", "premium_index_1m_close",
    "premium_rows",
)


@dataclass(frozen=True)
class InputBinding:
    name: str
    path: str
    sha256: str
    size_bytes: int
    compressed: bool
    mode: int = 0o644


@dataclass(frozen=True)
class MaterializationConfig:
    root: Path
    inputs: tuple[InputBinding, ...]
    attempt_path: str
    market_output_path: str
    oi_output_path: str
    manifest_output_path: str
    support_output_path: str
    inherited_manifest_path: str
    old_last: pd.Timestamp
    domain_end: pd.Timestamp
    expected_old_rows: int
    expected_complete_rows: int
    expected_append_rows: int
    splice_rows: int = 13
    rank7_tail_rows: int = 3000
    repository_head: str = "synthetic-s10"
    repository_parent: str = T9
    authority_commit: str = A10
    branch: str = BRANCH
    enforce_repository_gates: bool = False
    inherited_manifest: Mapping[str, Any] | None = None

    @property
    def output_paths(self) -> tuple[str, ...]:
        return (
            self.attempt_path,
            self.market_output_path,
            self.oi_output_path,
            self.manifest_output_path,
            self.support_output_path,
        )


OFFICIAL_INPUTS = (
    InputBinding("old_market", "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz", "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c", 66696659, True),
    InputBinding("replacement_market", "/home/pakchu/rllm/data/cache_market_ext_5m_wavefull_2020-01-01_2026-07-05_dbappend.csv.gz", "0447a2c89926a1deebdfd495edde069a697d9481bc5936bc360c8c1488de2ebe", 65420089, True),
    InputBinding("funding", "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz", "4d381be086e275bacaf31df431dc31307a71a26b3947b7082efffc10bb129dd7", 89326, True),
    InputBinding("premium", "data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz", "b45fcc5a3cf75c8e594effe61a698c4652f841b1d304107e9669524e0fc9d0d7", 1196481, True),
    InputBinding("old_open_interest", "/tmp/btcusdt_open_interest_5m_2020_2026.csv", "e08f93033e56959e8e7a9c1e21f27c5f01efc8d06fa6b4fbbfe7354697122b31", 19657777, False),
    InputBinding("binance_metrics_open_interest", "/home/pakchu/rllm/data/binance_um_metrics_BTCUSDT_5m_2020-09-01_2026-06-01.csv.gz", "d391022352d5b14dea7ffd207a9d1f84f603d06ddae42da55dd792f722fc0106", 21440132, True),
    InputBinding("rank7_spot_premium_5m", "/home/pakchu/rllm/data/cache_spot_premium_5m_2020-01-01_2026-06-01.csv.gz", "c21ed3f52804a3f879ef5167b6b81ee0fd7dd262f2e9d87ee5c1c25dbad73617", 15772146, True),
)


def official_config(root: Path | None = None) -> MaterializationConfig:
    repository = Path(__file__).resolve().parents[1] if root is None else root
    return MaterializationConfig(
        root=repository,
        inputs=OFFICIAL_INPUTS,
        attempt_path="results/gross9_structural_clock_bundle_g9cb10_source_support_attempt_consumed_2026-07-31.json",
        market_output_path="data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb10_complete.csv.gz",
        oi_output_path="data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb10_complete.csv.gz",
        manifest_output_path="configs/shadow/gross9_structural_clock_bundle_g9cb10_sources_2026-07-31.json",
        support_output_path="results/gross9_structural_clock_bundle_g9cb10_source_support_2026-07-31.json",
        inherited_manifest_path="configs/shadow/portfolio_added_alpha_signal_parity_sources_2026-07-16.json",
        old_last=pd.Timestamp("2026-05-31 15:00:00"),
        domain_end=pd.Timestamp("2026-06-01 00:00:00"),
        expected_old_rows=674785,
        expected_complete_rows=674892,
        expected_append_rows=107,
        enforce_repository_gates=True,
    )


def _fail(message: str) -> NoReturn:
    raise SourceSupportFailure(message)


def canonical_json_bytes(payload: Any, *, trailing_lf: bool = True) -> bytes:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return raw + (b"\n" if trailing_lf else b"")


def object_hash(payload: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(canonical_json_bytes({k: v for k, v in payload.items() if k != field}, trailing_lf=False)).hexdigest()


def _with_hash(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = object_hash(result, field)
    return result


def _path(config: MaterializationConfig, text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else config.root / path


def _timestamp_z(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_git(root: Path, *args: str) -> str:
    completed = subprocess.run(["git", "-C", str(root), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
    if completed.returncode:
        _fail(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _validate_command_and_root(config: MaterializationConfig) -> None:
    if sys.argv[1:]:
        _fail("the official entry point accepts no arguments")
    canonical = Path(__file__).resolve().parents[1]
    if config.root.resolve(strict=True) != canonical or Path.cwd().resolve(strict=True) != canonical:
        _fail("canonical repository root differs")
    if os.environ.get("PYTHONPATH") != str(canonical):
        _fail("official PYTHONPATH command shape differs")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or not sys.dont_write_bytecode:
        _fail("bytecode-disabled command shape differs")


def _validate_bytecode_first_gate(root: Path) -> None:
    # This is the shared Q8 preflight, imported lazily so command-shape checking
    # precedes its repository traversal.
    from training.preregister_gross9_structural_clock_bundle import validate_repository_bytecode_preflight

    try:
        validate_repository_bytecode_preflight(root)
    except (OSError, ValueError) as exc:
        raise SourceSupportFailure(str(exc)) from exc


def _validate_git_gate(config: MaterializationConfig) -> str:
    root = config.root
    head = _run_git(root, "rev-parse", "HEAD")
    if _run_git(root, "branch", "--show-current") != config.branch:
        _fail("official branch differs")
    if _run_git(root, "rev-parse", "@{upstream}") != head:
        _fail("HEAD and upstream differ")
    parents = _run_git(root, "rev-list", "--parents", "-n", "1", head).split()
    if len(parents) != 2 or parents[1] != T9:
        _fail("S10 direct parent differs")
    if _run_git(root, "rev-parse", f"{T9}^") != A10:
        _fail("T9/A10 topology differs")
    if _run_git(root, "rev-parse", f"{A10}^") != S9:
        _fail("A10/S9 topology differs")
    if _run_git(root, "rev-parse", f"{S9}^") != T8 or _run_git(root, "rev-parse", f"{T8}^") != A9:
        _fail("S9/T8/A9 topology differs")
    if _run_git(root, "diff", "--name-status", S9, A10).splitlines() != [
        "A\tdocs/gross9-structural-clock-bundle-g9cb10-successor-authority-decision-2026-07-31.md"
    ]:
        _fail("exact A10 one-file diff differs")
    if sorted(_run_git(root, "diff", "--name-status", A10, T9).splitlines()) != [
        "A\tresults/gross9_structural_clock_bundle_g9cb9_source_support_attempt_consumed_2026-07-31.json",
        "A\tresults/gross9_structural_clock_bundle_g9cb9_source_support_terminal_failure_2026-07-31.json",
    ]:
        _fail("exact T9 two-file diff differs")
    expected = [
        "A\ttests/test_materialize_gross9_structural_clock_g9cb10_sources.py",
        "A\ttraining/materialize_gross9_structural_clock_g9cb10_sources.py",
    ]
    actual = sorted(_run_git(root, "diff", "--name-status", T9, head).splitlines())
    if actual != expected:
        _fail("exact S10 two-file diff differs")
    for entry in expected:
        relative = entry.split("\t", 1)[1]
        indexed = _run_git(root, "ls-files", "-s", "--", relative).split()
        if len(indexed) < 4 or indexed[0] != "100644" or indexed[2] != "0" or indexed[3] != relative:
            _fail(f"S10 Git mode or index binding differs: {relative}")
    if _run_git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        _fail("index or worktree is not clean")
    return head


_A10_AUTHORITY = (
    "docs/gross9-structural-clock-bundle-g9cb10-successor-authority-decision-2026-07-31.md",
    61207,
    "dc23f38b724790004600ce001d73bea627a9111823a533c41ac659872369fa22",
    "0dfb881fb7853b5c76b9757cafed4a3e0f95702e",
)
_T9_EVIDENCE = (
    (
        "results/gross9_structural_clock_bundle_g9cb9_source_support_attempt_consumed_2026-07-31.json",
        3049,
        "aabfc7ec1fc5e7ec7f06803a48e6a7d4c024f73531b134f9f7af8051f913421c",
        "e6f7485675d974d1ce8d20194ff3b715238464f2",
    ),
    (
        "results/gross9_structural_clock_bundle_g9cb9_source_support_terminal_failure_2026-07-31.json",
        2835,
        "e2379760507306f9e810e8d504af37e3fd3aa2f58c545c72467a76904991289c",
        "ee6a5ddea961c8fe605c5f9ec9104bbdc8bcfb7e",
    ),
)
_G9CB9_PERMANENT_ABSENCES = (
    "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb9_complete.csv.gz",
    "data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb9_complete.csv.gz",
    "configs/shadow/gross9_structural_clock_bundle_g9cb9_sources_2026-07-31.json",
    "results/gross9_structural_clock_bundle_g9cb9_source_support_2026-07-31.json",
)


def _read_bound_json(
    config: MaterializationConfig,
    relative: str,
    size: int,
    digest: str,
    blob: str,
    *,
    mode: int,
) -> dict[str, Any]:
    fd = _open_nofollow_components(config.root / relative, os.O_RDONLY)
    try:
        info = os.fstat(fd)
        raw = _pread_complete(fd, info.st_size)
    finally:
        os.close(fd)
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != mode
        or info.st_nlink != 1
        or info.st_size != size
        or hashlib.sha256(raw).hexdigest() != digest
    ):
        _fail(f"bound evidence differs: {relative}")
    indexed = _run_git(config.root, "ls-files", "-s", "--", relative).split()
    if len(indexed) < 4 or indexed[:2] != ["100644", blob] or indexed[2:] != ["0", relative]:
        _fail(f"bound evidence Git identity differs: {relative}")
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=lambda pairs: _unique_json(pairs))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceSupportFailure(f"bound evidence JSON differs: {relative}") from exc
    if canonical_json_bytes(payload) != raw:
        _fail(f"bound evidence canonical bytes differ: {relative}")
    return payload


def _validate_t9_gate(config: MaterializationConfig) -> None:
    authority_path, authority_size, authority_digest, authority_blob = _A10_AUTHORITY
    authority_fd = _open_nofollow_components(config.root / authority_path, os.O_RDONLY)
    try:
        authority_info = os.fstat(authority_fd)
        authority_raw = _pread_complete(authority_fd, authority_info.st_size)
    finally:
        os.close(authority_fd)
    authority_index = _run_git(config.root, "ls-files", "-s", "--", authority_path).split()
    if (
        not stat.S_ISREG(authority_info.st_mode)
        or stat.S_IMODE(authority_info.st_mode) != 0o644
        or authority_info.st_nlink != 1
        or authority_info.st_size != authority_size
        or hashlib.sha256(authority_raw).hexdigest() != authority_digest
        or authority_index != ["100644", authority_blob, "0", authority_path]
    ):
        _fail("A10 authority binding differs")

    sentinel = _read_bound_json(config, *_T9_EVIDENCE[0], mode=0o444)
    if set(sentinel) != {
        "attempt_hash", "branch", "expected_outputs", "identity", "one_shot",
        "raw_inputs", "repository_head", "repository_parent", "resume_allowed",
        "retry_allowed", "source_access_at_publication", "status", "topology",
        "version",
    }:
        _fail("S9 attempt sentinel schema differs")
    prior_outputs = [
        "results/gross9_structural_clock_bundle_g9cb9_source_support_attempt_consumed_2026-07-31.json",
        *_G9CB9_PERMANENT_ABSENCES,
    ]
    expected_sentinel_core = {
        "branch": BRANCH,
        "expected_outputs": prior_outputs,
        "identity": "G9CB-9-SOURCE-SUPPORT",
        "one_shot": True,
        "raw_inputs": [
            {
                "mode_octal": f"{row.mode:04o}",
                "name": row.name,
                "path": row.path,
                "path_type": "regular_file",
                "sha256": row.sha256,
                "size_bytes": row.size_bytes,
            }
            for row in OFFICIAL_INPUTS
        ],
        "repository_head": S9,
        "repository_parent": T8,
        "resume_allowed": False,
        "retry_allowed": False,
        "source_access_at_publication": {
            "opaque_bytes_hashed": 190272610,
            "preexisting_sources_decoded": 0,
            "source_rows_decoded": 0,
        },
        "status": "source_support_attempt_consumed_before_source_decode",
        "topology": {
            "authority_commit": A9,
            "implementation_commit": S9,
            "terminal_evidence_commit": T8,
        },
        "version": "gross9_structural_clock_bundle_g9cb9_source_support_v1",
    }
    expected_sentinel = _with_hash(expected_sentinel_core, "attempt_hash")
    if canonical_json_bytes(sentinel) != canonical_json_bytes(expected_sentinel):
        _fail("S9 attempt sentinel constants differ")

    terminal = _read_bound_json(config, *_T9_EVIDENCE[1], mode=0o444)
    if set(terminal) != {
        "schema_version", "ledger_kind", "identity", "status", "authority",
        "seal_authority", "implementation", "attempt_sentinel", "execution",
        "failure", "access", "output_state", "terminal_failure_hash",
    }:
        _fail("T9 terminal ledger schema differs")
    terminal_core = {
        "schema_version": 1,
        "ledger_kind": "gross9_structural_clock_bundle_g9cb9_source_support_terminal_failure_v1",
        "identity": "G9CB-9-SOURCE-SUPPORT",
        "status": "terminal_market_prefix_mismatch",
        "authority": {
            "commit": A9,
            "path": "docs/gross9-structural-clock-bundle-g9cb9-successor-authority-decision-2026-07-31.md",
        },
        "seal_authority": {"commit": A10, "path": authority_path},
        "implementation": {
            "commit": S9,
            "parent_commit": T8,
            "files": [
                "training/materialize_gross9_structural_clock_g9cb9_sources.py",
                "tests/test_materialize_gross9_structural_clock_g9cb9_sources.py",
            ],
        },
        "attempt_sentinel": {
            "path": _T9_EVIDENCE[0][0],
            "path_type": "regular_file",
            "filesystem_mode_octal": "0444",
            "link_count": 1,
            "size_bytes": _T9_EVIDENCE[0][1],
            "sha256": _T9_EVIDENCE[0][2],
            "attempt_hash": expected_sentinel["attempt_hash"],
        },
        "execution": {
            "official_command": "PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.materialize_gross9_structural_clock_g9cb9_sources",
            "official_invocation_count": 1,
            "exit_status": 1,
            "retry_allowed": False,
            "resume_allowed": False,
        },
        "failure": {
            "class": "SourceSupportFailure",
            "message": "market logical prefix differs",
            "disclosed_mismatch_columns": ["kimchi_premium"],
            "mismatch_fraction_percent": "0.00148",
            "traceback_value_excerpt_emitted": True,
            "traceback_value_excerpt_restated_in_ledger": False,
        },
        "access": {
            "raw_file_count": 7,
            "decode_pass_count": 7,
            "decoded_preexisting_sources": [row.name for row in OFFICIAL_INPUTS],
            "market_prefix_comparison_count": 1,
            "generated_readback_decode_count": 0,
            "candidate_rows_opened": 0,
            "comparator_clock_rows_opened": 0,
            "model_history_or_rex_values_opened": 0,
            "pre2025_anchor_value_rows_opened": 0,
            "feature_signal_schedule_or_interval_values_computed": 0,
            "economic_or_overlap_values_computed": 0,
        },
        "output_state": {
            "published_terminal_evidence": [_T9_EVIDENCE[0][0]],
            "permanently_absent_outputs": list(_G9CB9_PERMANENT_ABSENCES),
            "generated_source_readback_count": 0,
            "m9_permitted": False,
            "q9_permitted": False,
        },
    }
    expected_terminal = _with_hash(terminal_core, "terminal_failure_hash")
    if canonical_json_bytes(terminal) != canonical_json_bytes(expected_terminal):
        _fail("T9 terminal ledger constants differ")
    for relative in _G9CB9_PERMANENT_ABSENCES:
        path = config.root / relative
        if path.exists() or path.is_symlink():
            _fail(f"permanent G9CB-9 absence differs: {relative}")


def _stat_token(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode), stat.S_IMODE(info.st_mode), info.st_size, info.st_nlink)


def _descriptor_hash(fd: int) -> tuple[str, int]:
    info = os.fstat(fd)
    digest = hashlib.sha256()
    offset = 0
    while offset < info.st_size:
        chunk = os.pread(fd, min(1024 * 1024, info.st_size - offset), offset)
        if not chunk:
            _fail("retained source descriptor short read")
        digest.update(chunk)
        offset += len(chunk)
    if offset != info.st_size or os.pread(fd, 1, offset):
        _fail("retained source descriptor size differs")
    return digest.hexdigest(), offset


def _pread_complete(fd: int, size: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        chunk = os.pread(fd, min(1024 * 1024, size - offset), offset)
        if not chunk:
            _fail("descriptor readback was short")
        chunks.append(chunk)
        offset += len(chunk)
    if os.pread(fd, 1, size):
        _fail("descriptor readback exceeded bound size")
    return b"".join(chunks)


@dataclass
class _Retained:
    binding: InputBinding
    fd: int
    token: tuple[int, int, int, int, int, int]


def _open_nofollow_components(path: Path, final_flags: int) -> int:
    """Open *path* without allowing a symlink in any traversed component."""
    absolute = path.absolute()
    parts = absolute.parts
    if not absolute.is_absolute() or len(parts) < 2:
        _fail(f"invalid bound path: {path}")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    directory_fd = os.open(parts[0], directory_flags)
    try:
        for component in parts[1:-1]:
            if component in ("", ".", ".."):
                _fail(f"invalid bound path component: {path}")
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        if parts[-1] in ("", ".", ".."):
            _fail(f"invalid bound path leaf: {path}")
        return os.open(parts[-1], final_flags | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0), dir_fd=directory_fd)
    finally:
        os.close(directory_fd)


def _stat_nofollow_components(path: Path) -> os.stat_result:
    """Stat *path* without reopening its leaf or following any symlink."""
    absolute = path.absolute()
    parts = absolute.parts
    if not absolute.is_absolute() or len(parts) < 2:
        _fail(f"invalid bound path: {path}")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    directory_fd = os.open(parts[0], directory_flags)
    try:
        for component in parts[1:-1]:
            if component in ("", ".", ".."):
                _fail(f"invalid bound path component: {path}")
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        if parts[-1] in ("", ".", ".."):
            _fail(f"invalid bound path leaf: {path}")
        return os.stat(parts[-1], dir_fd=directory_fd, follow_symlinks=False)
    finally:
        os.close(directory_fd)


def _path_edge_check(retained: _Retained) -> None:
    try:
        edge = _stat_nofollow_components(Path(retained.binding.path))
    except OSError as exc:
        raise SourceSupportFailure(f"raw input path-edge drift: {retained.binding.name}") from exc
    if _stat_token(edge) != retained.token:
        _fail(f"raw input pathname replacement: {retained.binding.name}")


def _identity_check(retained: _Retained, phase: str, failpoint: Callable[[str], None]) -> None:
    info = os.fstat(retained.fd)
    if _stat_token(info) != retained.token or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        _fail(f"retained descriptor identity drift: {retained.binding.name}:{phase}")
    _path_edge_check(retained)
    failpoint(f"identity:{retained.binding.name}:{phase}")


def _open_inputs(config: MaterializationConfig, failpoint: Callable[[str], None]) -> dict[str, _Retained]:
    if len(config.inputs) != 7 or len({row.name for row in config.inputs}) != 7:
        _fail("exactly seven unique raw inputs are required")
    retained: dict[str, _Retained] = {}
    identities: set[tuple[int, int]] = set()
    try:
        for binding in config.inputs:
            path = _path(config, binding.path)
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            try:
                fd = _open_nofollow_components(path, flags)
            except OSError as exc:
                raise SourceSupportFailure(f"cannot open bound raw input: {binding.name}") from exc
            info = os.fstat(fd)
            token = _stat_token(info)
            item = _Retained(InputBinding(binding.name, str(path), binding.sha256, binding.size_bytes, binding.compressed, binding.mode), fd, token)
            try:
                if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != binding.mode or info.st_nlink != 1 or info.st_size != binding.size_bytes:
                    _fail(f"raw input metadata drift: {binding.name}")
                identity = (info.st_dev, info.st_ino)
                if identity in identities:
                    _fail("raw input inode aliasing")
                identities.add(identity)
                failpoint(f"identity:{binding.name}:before_hash")
                digest, size = _descriptor_hash(fd)
                if digest != binding.sha256 or size != binding.size_bytes:
                    _fail(f"raw input byte binding drift: {binding.name}")
                _identity_check(item, "after_open_hash", failpoint)
                retained[binding.name] = item
            except BaseException:
                os.close(fd)
                raise
        return retained
    except BaseException:
        for item in retained.values():
            os.close(item.fd)
        raise


def _validate_output_absence(config: MaterializationConfig) -> None:
    input_ids: set[tuple[int, int]] = set()
    for binding in config.inputs:
        info = os.lstat(_path(config, binding.path))
        input_ids.add((info.st_dev, info.st_ino))
    for text in config.output_paths:
        path = _path(config, text)
        current = Path(path.anchor)
        for component in path.parts[1:-1]:
            current /= component
            info = os.lstat(current)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                _fail(f"canonical output parent is not a real directory: {text}")
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            _fail(f"canonical output is a symlink: {text}")
        if (info.st_dev, info.st_ino) in input_ids:
            _fail(f"canonical output aliases an input: {text}")
        _fail(f"canonical output already exists: {text}")


def _snapshot_output_directories(config: MaterializationConfig) -> dict[Path, frozenset[str]]:
    return {
        _path(config, text).parent: frozenset(os.listdir(_path(config, text).parent))
        for text in config.output_paths
    }


def _validate_output_directory_delta(
    config: MaterializationConfig,
    baseline: Mapping[Path, frozenset[str]],
) -> None:
    expected_by_parent: dict[Path, set[str]] = {parent: set() for parent in baseline}
    for text in config.output_paths:
        path = _path(config, text)
        expected_by_parent[path.parent].add(path.name)
    for parent, original in baseline.items():
        current = frozenset(os.listdir(parent))
        if current != original.union(expected_by_parent[parent]):
            _fail(f"unauthorized output or temporary residue: {parent}")


def _decode_csv(item: _Retained, failpoint: Callable[[str], None]) -> tuple[pd.DataFrame, tuple[str, ...]]:
    _identity_check(item, "before_decode", failpoint)
    duplicate = os.dup(item.fd)
    try:
        os.lseek(duplicate, 0, os.SEEK_SET)
        with ExitStack() as stack:
            raw: io.BufferedIOBase = stack.enter_context(os.fdopen(duplicate, "rb", closefd=True))
            duplicate = -1
            stream: Any = (
                stack.enter_context(gzip.GzipFile(fileobj=raw, mode="rb"))
                if item.binding.compressed
                else raw
            )
            text = stack.enter_context(io.TextIOWrapper(stream, encoding="utf-8", newline=""))
            header_line = text.readline()
            try:
                header = tuple(next(csv.reader([header_line], strict=True)))
            except (csv.Error, StopIteration) as exc:
                raise SourceSupportFailure(f"invalid CSV header: {item.binding.name}") from exc
            if not header or len(set(header)) != len(header):
                _fail(f"duplicate or empty CSV columns: {item.binding.name}")
            text.seek(0)
            frame = pd.read_csv(text, low_memory=False)
            if tuple(frame.columns) != header:
                _fail(f"CSV schema decode drift: {item.binding.name}")
    finally:
        if duplicate >= 0:
            os.close(duplicate)
    _identity_check(item, "after_decode", failpoint)
    digest, size = _descriptor_hash(item.fd)
    if digest != item.binding.sha256 or size != item.binding.size_bytes:
        _fail(f"postdecode source rehash differs: {item.binding.name}")
    _identity_check(item, "after_rehash", failpoint)
    return frame, header


@dataclass(frozen=True)
class ReplacementDateScan:
    schema: tuple[str, ...]
    row_count: int
    tail_positions: tuple[int, ...]
    tail_dates: pd.Series


def _scan_replacement_dates(
    item: _Retained,
    config: MaterializationConfig,
    failpoint: Callable[[str], None],
) -> ReplacementDateScan:
    """Scan only replacement timestamps and select physical tail positions."""
    _identity_check(item, "before_date_scan", failpoint)
    duplicate = os.dup(item.fd)
    dates: list[str] = []
    try:
        os.lseek(duplicate, 0, os.SEEK_SET)
        with ExitStack() as stack:
            raw: io.BufferedIOBase = stack.enter_context(os.fdopen(duplicate, "rb", closefd=True))
            duplicate = -1
            stream: Any = (
                stack.enter_context(gzip.GzipFile(fileobj=raw, mode="rb"))
                if item.binding.compressed
                else raw
            )
            text = stack.enter_context(io.TextIOWrapper(stream, encoding="utf-8", newline=""))
            reader = csv.reader(text, strict=True)
            try:
                header = tuple(next(reader))
                if header != MARKET_SCHEMA:
                    _fail("replacement market schema drift")
                for row in reader:
                    if not row:
                        _fail("replacement market row lacks date")
                    dates.append(row[0])
            except (csv.Error, StopIteration) as exc:
                raise SourceSupportFailure("replacement market date scan failed") from exc
    finally:
        if duplicate >= 0:
            os.close(duplicate)
    parsed = _dates(pd.Series(dates, dtype="object"), "replacement_market_date_scan")
    if parsed.duplicated().any():
        _fail("duplicate timestamps: replacement_market_date_scan")
    if not parsed.is_monotonic_increasing:
        _fail("out-of-order timestamps: replacement_market_date_scan")
    _require_five_minute_alignment(parsed, label="replacement_market_date_scan")
    mask = (parsed > config.old_last) & (parsed < config.domain_end)
    positions = tuple(int(index) for index in np.flatnonzero(mask.to_numpy()))
    tail_dates = parsed.iloc[list(positions)].reset_index(drop=True)
    required_first = config.old_last + pd.Timedelta(minutes=5)
    required_last = config.domain_end - pd.Timedelta(minutes=5)
    if len(positions) != config.expected_append_rows:
        _fail("replacement market tail row count differs")
    _require_grid(
        tail_dates,
        first=required_first,
        last=required_last,
        label="replacement_market_tail_positions",
    )
    _identity_check(item, "after_date_scan", failpoint)
    digest, size = _descriptor_hash(item.fd)
    if digest != item.binding.sha256 or size != item.binding.size_bytes:
        _fail("postscan source rehash differs: replacement_market")
    _identity_check(item, "after_date_scan_rehash", failpoint)
    return ReplacementDateScan(header, len(parsed), positions, tail_dates)


def _decode_replacement_tail(
    item: _Retained,
    scan: ReplacementDateScan,
    failpoint: Callable[[str], None],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Decode full values only for the timestamp-selected replacement tail."""
    _identity_check(item, "before_tail_decode", failpoint)
    duplicate = os.dup(item.fd)
    selected: list[list[str]] = []
    row_count = 0
    wanted = iter(scan.tail_positions)
    next_position = next(wanted, None)
    try:
        os.lseek(duplicate, 0, os.SEEK_SET)
        with ExitStack() as stack:
            raw: io.BufferedIOBase = stack.enter_context(os.fdopen(duplicate, "rb", closefd=True))
            duplicate = -1
            stream: Any = (
                stack.enter_context(gzip.GzipFile(fileobj=raw, mode="rb"))
                if item.binding.compressed
                else raw
            )
            text = stack.enter_context(io.TextIOWrapper(stream, encoding="utf-8", newline=""))
            reader = csv.reader(text, strict=True)
            try:
                header = tuple(next(reader))
                if header != scan.schema or header != MARKET_SCHEMA:
                    _fail("replacement market tail schema drift")
                for row_count, row in enumerate(reader, start=1):
                    physical_position = row_count - 1
                    if physical_position != next_position:
                        continue
                    if len(row) != len(header):
                        _fail("replacement market selected tail row width differs")
                    selected.append(row)
                    next_position = next(wanted, None)
            except (csv.Error, StopIteration) as exc:
                raise SourceSupportFailure("replacement market tail decode failed") from exc
    finally:
        if duplicate >= 0:
            os.close(duplicate)
    if row_count != scan.row_count or next_position is not None or len(selected) != len(scan.tail_positions):
        _fail("replacement market physical tail selection drift")
    rebuilt = io.StringIO(newline="")
    writer = csv.writer(rebuilt, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(scan.schema)
    writer.writerows(selected)
    rebuilt.seek(0)
    frame = pd.read_csv(rebuilt, low_memory=False)
    if tuple(frame.columns) != scan.schema or len(frame) != len(selected):
        _fail("replacement market selected tail decode drift")
    _identity_check(item, "after_tail_decode", failpoint)
    digest, size = _descriptor_hash(item.fd)
    if digest != item.binding.sha256 or size != item.binding.size_bytes:
        _fail("postdecode source rehash differs: replacement_market_tail")
    _identity_check(item, "after_tail_rehash", failpoint)
    return frame, scan.schema


def _dates(series: pd.Series, label: str) -> pd.Series:
    try:
        result = pd.to_datetime(series, errors="raise", utc=True).dt.tz_convert(None)
    except (TypeError, ValueError) as exc:
        raise SourceSupportFailure(f"invalid timestamps: {label}") from exc
    if result.isna().any():
        _fail(f"null timestamps: {label}")
    return result


def _normalise_ordered(frame: pd.DataFrame, date_column: str, label: str) -> pd.DataFrame:
    out = frame.copy()
    out[date_column] = _dates(out[date_column], label)
    if out[date_column].duplicated().any():
        _fail(f"duplicate timestamps: {label}")
    if not out[date_column].is_monotonic_increasing:
        _fail(f"out-of-order timestamps: {label}")
    sorted_out = out.sort_values(date_column, kind="mergesort").drop_duplicates(date_column, keep="last").reset_index(drop=True)
    if len(sorted_out) != len(out):
        _fail(f"timestamp normalization changed rows: {label}")
    return sorted_out


def _require_grid(dates: pd.Series, *, first: pd.Timestamp | None = None, last: pd.Timestamp | None = None, label: str) -> None:
    if dates.empty:
        _fail(f"empty timestamp grid: {label}")
    if first is not None and dates.iloc[0] != first:
        _fail(f"early/late first timestamp: {label}")
    if last is not None and dates.iloc[-1] != last:
        _fail(f"early/late terminal coverage: {label}")
    values = dates.astype("int64").to_numpy()
    if len(values) > 1 and not np.all(np.diff(values) == 300_000_000_000):
        _fail(f"internal five-minute grid gap: {label}")
    if np.any((values // 1_000_000_000) % 300 != 0):
        _fail(f"off-grid timestamps: {label}")


def _require_five_minute_alignment(dates: pd.Series, *, label: str) -> None:
    if dates.empty:
        _fail(f"empty timestamp sequence: {label}")
    values = dates.astype("int64").to_numpy()
    if np.any((values // 1_000_000_000) % 300 != 0):
        _fail(f"off-grid timestamps: {label}")


def _exact_frame(left: pd.DataFrame, right: pd.DataFrame, label: str) -> None:
    try:
        pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True), check_dtype=True, check_exact=True)
    except AssertionError as exc:
        raise SourceSupportFailure(f"{label} differs") from exc


def _numeric_float64_preserving_missing(series: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_numeric(series, errors="coerce")
    malformed = parsed.isna() & ~series.isna()
    if malformed.any():
        _fail(f"malformed numeric value: {label}")
    return parsed.astype("float64")


def _transform(
    config: MaterializationConfig,
    frames: Mapping[str, pd.DataFrame],
    schemas: Mapping[str, tuple[str, ...]],
    replacement_scan: ReplacementDateScan,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, int]]:
    old_market_raw = frames["old_market"]
    replacement_tail_raw = frames["replacement_market"]
    if schemas["old_market"] != MARKET_SCHEMA or schemas["replacement_market"] != MARKET_SCHEMA:
        _fail("market schema drift")
    old_market = _normalise_ordered(old_market_raw, "date", "old_market")
    replacement_tail = _normalise_ordered(
        replacement_tail_raw, "date", "replacement_market_tail"
    )
    if len(old_market) != config.expected_old_rows or old_market.date.iloc[-1] != config.old_last:
        _fail("old market row count or terminal timestamp differs")
    required_last = config.domain_end - pd.Timedelta(minutes=5)
    if len(replacement_tail) != config.expected_append_rows:
        _fail("replacement market tail row count differs")
    _require_grid(
        replacement_tail.date,
        first=config.old_last + pd.Timedelta(minutes=5),
        last=required_last,
        label="replacement_market_tail",
    )
    if not replacement_tail.date.equals(replacement_scan.tail_dates):
        _fail("replacement market tail timestamps differ from date scan")
    old_timestamps = set(old_market.date.astype("int64").tolist())
    tail_timestamps = set(replacement_tail.date.astype("int64").tolist())
    if old_timestamps.intersection(tail_timestamps):
        _fail("market partition timestamp intersection differs")
    market = pd.concat([old_market, replacement_tail], ignore_index=True)
    output_timestamps = set(market.date.astype("int64").tolist())
    if output_timestamps != old_timestamps.union(tail_timestamps):
        _fail("market partition union differs")
    if len(market) != len(old_market) + len(replacement_tail) or len(market) != config.expected_complete_rows:
        _fail("materialized market row counts differ")
    _exact_frame(market.iloc[: len(old_market)], old_market, "materialized old market partition")
    _exact_frame(market.iloc[len(old_market) :], replacement_tail, "materialized replacement tail partition")
    _require_grid(market.date, first=old_market.date.iloc[0], last=required_last, label="materialized_market")

    old_oi_raw = frames["old_open_interest"]
    metrics_raw = frames["binance_metrics_open_interest"]
    if schemas["old_open_interest"] != OI_SCHEMA or schemas["binance_metrics_open_interest"] != METRICS_SCHEMA:
        _fail("open-interest schema drift")
    old_oi = _normalise_ordered(old_oi_raw, "date", "old_open_interest")
    metrics = _normalise_ordered(metrics_raw, "create_time", "binance_metrics_open_interest")
    if len(old_oi) != config.expected_old_rows or old_oi.date.iloc[-1] != config.old_last:
        _fail("old OI row count or terminal timestamp differs")
    _require_grid(old_oi.date, last=config.old_last, label="old_open_interest")
    if not metrics["symbol"].eq("BTCUSDT").all():
        _fail("metrics symbol differs")
    metrics_oi = _numeric_float64_preserving_missing(
        metrics["sum_open_interest"], "binance_metrics_open_interest.sum_open_interest"
    )
    mapped = pd.DataFrame({"date": metrics["create_time"], "open_interest": metrics_oi})
    _require_five_minute_alignment(mapped.date, label="binance_metrics_open_interest")
    old_values = _numeric_float64_preserving_missing(
        old_oi.open_interest, "old_open_interest.open_interest"
    )
    old_numeric = pd.DataFrame({"date": old_oi.date, "open_interest": old_values})
    common = old_numeric.merge(mapped, on="date", how="inner", suffixes=("_old", "_metrics"), validate="one_to_one")
    left = common.open_interest_old.to_numpy(dtype="float64")
    right = common.open_interest_metrics.to_numpy(dtype="float64")
    if not np.array_equal(left, right, equal_nan=True):
        _fail("OI overlap conflict")
    splice_start = config.old_last - pd.Timedelta(minutes=5 * (config.splice_rows - 1))
    splice = common.loc[(common.date >= splice_start) & (common.date <= config.old_last)]
    if len(splice) != config.splice_rows:
        _fail("missing OI splice anchor window")
    _require_grid(splice.date, first=splice_start, last=config.old_last, label="OI splice anchor window")
    tail_dates = pd.date_range(config.old_last + pd.Timedelta(minutes=5), required_last, freq="5min")
    tail = mapped.set_index("date").reindex(tail_dates)
    if len(tail) != config.expected_append_rows or tail.open_interest.isna().any():
        _fail("missing tail OI")
    tail_values = tail.open_interest.to_numpy(dtype="float64")
    if not np.isfinite(tail_values).all() or not (tail_values > 0).all():
        _fail("non-positive or non-finite tail OI")
    tail_frame = pd.DataFrame({"date": tail.index, "open_interest": tail_values})
    materialized_oi = pd.concat([old_numeric, tail_frame], ignore_index=True)
    if len(materialized_oi) != config.expected_complete_rows:
        _fail("materialized OI row count differs")
    _require_grid(materialized_oi.date, first=old_oi.date.iloc[0], last=required_last, label="materialized_open_interest")

    rank_raw = frames["rank7_spot_premium_5m"]
    rank_schema = schemas["rank7_spot_premium_5m"]
    if any(column not in rank_schema for column in RANK7_PROJECTION):
        _fail("Rank7 projection column missing")
    rank = _normalise_ordered(rank_raw.loc[:, list(RANK7_PROJECTION)], "date", "rank7_spot_premium_5m")
    _require_five_minute_alignment(rank.date, label="rank7_spot_premium_5m")
    joined_rank = market.loc[:, ["date"]].merge(rank, on="date", how="left", validate="one_to_one", indicator=True)
    if not joined_rank._merge.eq("both").all():
        _fail("incomplete Rank7 projection coverage")
    rank_tail = joined_rank.tail(config.rank7_tail_rows)
    if len(rank_tail) != min(config.rank7_tail_rows, len(market)):
        _fail("Rank7 tail count differs")
    for column in ("spot_rows", "premium_rows"):
        values = pd.to_numeric(rank_tail[column], errors="coerce").to_numpy(dtype="float64")
        if not np.isfinite(values).all() or not np.equal(values, 5.0).all():
            _fail("invalid recent Rank7 row counts")
    latest_spot = float(pd.to_numeric(joined_rank.spot_close, errors="coerce").iloc[-1])
    latest_premium = float(pd.to_numeric(joined_rank.premium_index_1m_close, errors="coerce").iloc[-1])
    if not math.isfinite(latest_spot) or latest_spot <= 0 or not math.isfinite(latest_premium):
        _fail("invalid latest Rank7 values")

    funding_raw = frames["funding"]
    premium_raw = frames["premium"]
    funding_dates = "funding_time" if "funding_time" in funding_raw.columns and "date" not in funding_raw.columns else "date"
    if funding_dates not in funding_raw or "funding_rate" not in funding_raw:
        _fail("funding columns differ")
    funding_check = _dates(funding_raw[funding_dates], "funding")
    if funding_check.duplicated().any() or not funding_check.is_monotonic_increasing:
        _fail("funding timestamp ordering/uniqueness differs")
    funding_values_raw = pd.to_numeric(funding_raw["funding_rate"], errors="coerce")
    if funding_values_raw.isna().any() or not np.isfinite(funding_values_raw.to_numpy(dtype="float64")).all():
        _fail("funding contains null or non-finite values")
    if "premium_index" in premium_raw.columns and "date" in premium_raw.columns:
        premium_dates = _dates(premium_raw.date, "premium")
    elif "close" in premium_raw.columns and "close_time" in premium_raw.columns:
        numeric = pd.to_numeric(premium_raw.close_time, errors="coerce")
        premium_dates = pd.to_datetime(numeric, unit="ms", errors="raise", utc=True).dt.tz_convert(None) if numeric.notna().any() else _dates(premium_raw.close_time, "premium")
    elif "close" in premium_raw.columns and "date" in premium_raw.columns:
        premium_dates = _dates(premium_raw.date, "premium")
    else:
        _fail("premium columns differ")
    if premium_dates.isna().any() or premium_dates.duplicated().any() or not premium_dates.is_monotonic_increasing:
        _fail("premium timestamp ordering/uniqueness differs")
    premium_value_column = "premium_index" if "premium_index" in premium_raw.columns else "close"
    premium_values_raw = pd.to_numeric(premium_raw[premium_value_column], errors="coerce")
    if premium_values_raw.isna().any() or not np.isfinite(premium_values_raw.to_numpy(dtype="float64")).all():
        _fail("premium contains null or non-finite values")
    funding = normalise_funding_history_frame(funding_raw)
    premium = normalise_premium_index_frame(premium_raw)
    attached, funding_available = _merge_aux(market, funding, value_cols=["funding_rate"], tolerance="12h")
    attached, premium_available = _merge_aux(attached, premium, value_cols=["premium_index"], tolerance="2h")
    if len(attached) != len(market) or not attached.date.equals(market.date):
        _fail("auxiliary attachment changed market timestamps")
    tail_slice = slice(len(old_market), len(market))
    funding_tail = funding_available.iloc[tail_slice]
    premium_tail = premium_available.iloc[tail_slice]
    funding_values = pd.to_numeric(attached.funding_rate.iloc[tail_slice], errors="coerce").to_numpy(dtype="float64")
    premium_values = pd.to_numeric(attached.premium_index.iloc[tail_slice], errors="coerce").to_numpy(dtype="float64")
    if len(funding_tail) != config.expected_append_rows or not funding_tail.eq(1.0).all() or not np.isfinite(funding_values).all():
        _fail("funding tail is unavailable or stale")
    if len(premium_tail) != config.expected_append_rows or not premium_tail.eq(1.0).all() or not np.isfinite(premium_values).all():
        _fail("premium tail is unavailable or stale")

    validation = {
        "appended_market_rows": config.expected_append_rows,
        "appended_open_interest_rows": config.expected_append_rows,
        "domain_end_exclusive": _timestamp_z(config.domain_end),
        "funding_attachment_tolerance": "12h",
        "funding_tail_available_rows": int(funding_tail.eq(1.0).sum()),
        "latest_funding_available_after_causal_attachment": bool(funding_tail.iloc[-1] == 1.0),
        "latest_open_interest_positive_after_exact_join": bool(tail_values[-1] > 0),
        "latest_premium_available_after_causal_attachment": bool(premium_tail.iloc[-1] == 1.0),
        "market_grid_seconds": 300,
        "market_partition": {
            "boundary_inclusive_old": _timestamp_z(config.old_last),
            "disjoint_union": True,
            "domain_end_exclusive": _timestamp_z(config.domain_end),
            "old_prefix_rows": len(old_market),
            "output_rows": len(market),
            "overlap_value_comparison_count": 0,
            "replacement_prefix_non_date_values_semantically_evaluated": 0,
            "replacement_prefix_rows_selected": 0,
            "replacement_tail_rows": len(replacement_tail),
            "timestamp_intersection_rows": 0,
        },
        "metrics_common_timestamp_values_exact": True,
        "oi_grid_seconds": 300,
        "oi_splice_window_exact_rows": len(splice),
        "oi_tail_exact_rows": len(tail),
        "old_last_timestamp": _timestamp_z(config.old_last),
        "premium_attachment_tolerance": "2h",
        "premium_tail_available_rows": int(premium_tail.eq(1.0).sum()),
        "rank7_spot_premium_exact_join_rows": len(joined_rank),
        "rank7_spot_premium_first_timestamp": _timestamp_z(rank.date.iloc[0]),
        "rank7_spot_premium_last_timestamp": _timestamp_z(rank.date.iloc[-1]),
        "rank7_spot_premium_latest_values_valid": True,
        "rank7_spot_premium_projection_schema": list(RANK7_PROJECTION),
        "rank7_spot_premium_raw_column_count": len(rank_schema),
        "rank7_spot_premium_raw_schema_sha256": hashlib.sha256(canonical_json_bytes(list(rank_schema), trailing_lf=False)).hexdigest(),
        "rank7_spot_premium_tail_complete_rows": len(rank_tail),
        "required_last_timestamp": _timestamp_z(required_last),
    }
    normalized = {
        "old_market": len(old_market), "replacement_market": len(replacement_tail),
        "funding": len(funding), "premium": len(premium),
        "old_open_interest": len(old_oi), "binance_metrics_open_interest": len(mapped),
        "rank7_spot_premium_5m": len(rank),
    }
    return market, materialized_oi, validation, normalized


def serialize_csv_bytes(frame: pd.DataFrame) -> tuple[bytes, str]:
    text = frame.to_csv(index=False, lineterminator="\n", date_format="%Y-%m-%d %H:%M:%S", float_format="%.17g", na_rep="", quoting=csv.QUOTE_MINIMAL)
    csv_bytes = text.encode("utf-8")
    return csv_bytes, hashlib.sha256(csv_bytes).hexdigest()


def serialize_csv_gzip(frame: pd.DataFrame) -> tuple[bytes, bytes, str]:
    csv_bytes, frame_hash = serialize_csv_bytes(frame)
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0) as handle:
        handle.write(csv_bytes)
    return buffer.getvalue(), csv_bytes, frame_hash


_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.linkat.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
_LIBC.linkat.restype = ctypes.c_int
_AT_FDCWD = -100
_AT_SYMLINK_FOLLOW = 0x400


def _write_all(fd: int, raw: bytes) -> None:
    view = memoryview(raw)
    offset = 0
    while offset < len(view):
        written = os.write(fd, view[offset:])
        if written <= 0:
            _fail("unnamed publication write stalled")
        offset += written


def _linkat(fd: int, directory_fd: int, leaf: str) -> None:
    result = _LIBC.linkat(_AT_FDCWD, f"/proc/self/fd/{fd}".encode("ascii"), directory_fd, os.fsencode(leaf), _AT_SYMLINK_FOLLOW)
    if result:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(f"write-once path exists: {leaf}")
        raise OSError(error, os.strerror(error), leaf)


def _publish(config: MaterializationConfig, relative: str, raw: bytes, label: str, failpoint: Callable[[str], None]) -> dict[str, Any]:
    path = _path(config, relative)
    parent = path.parent
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    parent_fd = _open_nofollow_components(parent, flags)
    unnamed = -1
    canonical = -1
    try:
        if not getattr(os, "O_TMPFILE", 0):
            _fail("O_TMPFILE is unavailable")
        unnamed = os.open(".", os.O_RDWR | os.O_TMPFILE | getattr(os, "O_CLOEXEC", 0), 0o600, dir_fd=parent_fd)
        initial = os.fstat(unnamed)
        if not stat.S_ISREG(initial.st_mode) or initial.st_size != 0 or initial.st_nlink != 0:
            _fail("unnamed publication inode is invalid")
        failpoint(f"publication:{label}:unnamed_initial")
        _write_all(unnamed, raw)
        os.fchmod(unnamed, 0o444)
        os.fsync(unnamed)
        complete = os.fstat(unnamed)
        readback = _pread_complete(unnamed, complete.st_size)
        if not stat.S_ISREG(initial.st_mode) or stat.S_IMODE(initial.st_mode) != 0o600 or not stat.S_ISREG(complete.st_mode) or stat.S_IMODE(complete.st_mode) != 0o444 or complete.st_nlink != 0 or complete.st_size != len(raw) or readback != raw:
            _fail("completed unnamed inode verification differs")
        failpoint(f"publication:{label}:unnamed_complete")
        failpoint(f"before_link:{label}")
        _linkat(unnamed, parent_fd, path.name)
        os.fsync(parent_fd)
        canonical = os.open(path.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0), dir_fd=parent_fd)
        info = os.fstat(canonical)
        canonical_raw = _pread_complete(canonical, info.st_size)
        if (info.st_dev, info.st_ino) != (complete.st_dev, complete.st_ino) or not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444 or info.st_nlink != 1 or canonical_raw != raw:
            _fail("linked canonical inode verification differs")
        failpoint(f"publication:{label}:linked_canonical")
        failpoint(f"after_link:{label}")
        return {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw), "inode": (info.st_dev, info.st_ino)}
    finally:
        if canonical >= 0:
            os.close(canonical)
        if unnamed >= 0:
            os.close(unnamed)
        os.close(parent_fd)


def _readback_generated(
    config: MaterializationConfig,
    relative: str,
    compressed: bool = True,
    failpoint: Callable[[str], None] | None = None,
) -> tuple[str, int, pd.DataFrame]:
    inject = failpoint or (lambda _name: None)
    path = _path(config, relative)
    fd = _open_nofollow_components(path, os.O_RDONLY)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444 or info.st_nlink != 1:
            _fail("generated readback metadata differs")
        digest, size = _descriptor_hash(fd)
        inject(f"readback:{relative}:authenticated")
        duplicate = os.dup(fd)
        with ExitStack() as stack:
            source = stack.enter_context(os.fdopen(duplicate, "rb"))
            stream: Any = (
                stack.enter_context(gzip.GzipFile(fileobj=source, mode="rb"))
                if compressed
                else source
            )
            frame = pd.read_csv(stream, float_precision="round_trip")
        return digest, size, frame
    finally:
        os.close(fd)


def _load_inherited_manifest(config: MaterializationConfig) -> dict[str, Any]:
    if config.inherited_manifest is not None:
        return json.loads(json.dumps(config.inherited_manifest))
    raw = _path(config, config.inherited_manifest_path).read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=lambda pairs: _unique_json(pairs))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceSupportFailure("inherited source manifest is invalid") from exc
    return payload


def _unique_json(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _build_manifest(config: MaterializationConfig, market_digest: str, oi_digest: str) -> dict[str, Any]:
    inherited = _load_inherited_manifest(config)
    if set(inherited) != {"schema_version", "as_of", "sources"} or inherited.get("schema_version") != 1 or not isinstance(inherited.get("sources"), list) or len(inherited["sources"]) != 8:
        _fail("inherited source manifest schema differs")
    rows: list[dict[str, Any]] = []
    for original in inherited["sources"]:
        row = dict(original)
        if row.get("name") == "market_5m":
            row["path"] = config.market_output_path
            row["sha256"] = market_digest
        elif row.get("name") == "open_interest":
            row["path"] = config.oi_output_path
            row["sha256"] = oi_digest
        rows.append(row)
        if row.get("name") == "open_interest":
            rank = next(binding for binding in config.inputs if binding.name == "rank7_spot_premium_5m")
            rows.append({"name": rank.name, "path": rank.path, "sha256": rank.sha256})
    names = [row.get("name") for row in rows]
    if len(rows) != 9 or len(set(names)) != 9 or names[4] != "rank7_spot_premium_5m":
        _fail("nine-row source manifest ordering differs")
    return {"schema_version": 1, "as_of": "2026-07-31", "sources": rows}


def _attempt(config: MaterializationConfig, head: str) -> dict[str, Any]:
    raw_inputs = [{"mode_octal": f"{row.mode:04o}", "name": row.name, "path": row.path, "path_type": "regular_file", "sha256": row.sha256, "size_bytes": row.size_bytes} for row in config.inputs]
    core = {
        "branch": config.branch,
        "expected_outputs": list(config.output_paths),
        "identity": IDENTITY,
        "one_shot": True,
        "raw_inputs": raw_inputs,
        "repository_head": head,
        "repository_parent": config.repository_parent,
        "resume_allowed": False,
        "retry_allowed": False,
        "source_access_at_publication": {"opaque_bytes_hashed": sum(row.size_bytes for row in config.inputs), "preexisting_sources_decoded": 0, "source_rows_decoded": 0},
        "status": "source_support_attempt_consumed_before_source_decode",
        "topology": {"authority_commit": config.authority_commit, "implementation_commit": head, "terminal_evidence_commit": config.repository_parent},
        "version": VERSION,
    }
    return _with_hash(core, "attempt_hash")


def _expected_access_ledger(
    config: MaterializationConfig,
    head: str,
    attempt: Mapping[str, Any],
    attempt_binding: Mapping[str, Any],
    replacement_scan: ReplacementDateScan,
    decoded_rows: Mapping[str, int],
) -> dict[str, Any]:
    historical_s9 = {
        "identity": "G9CB-9-SOURCE-SUPPORT",
        "attempt_sentinel": {
            "path": _T9_EVIDENCE[0][0],
            "sha256": _T9_EVIDENCE[0][2],
            "size_bytes": _T9_EVIDENCE[0][1],
            "attempt_hash": "fa7c7df3d7ab8b7622b9c954741330dfbbe5182599e441f63459e2249d4a2887",
            "git_blob": _T9_EVIDENCE[0][3],
            "git_mode": "100644",
            "seal_commit": T9,
        },
        "terminal_failure_ledger": {
            "path": _T9_EVIDENCE[1][0],
            "sha256": _T9_EVIDENCE[1][2],
            "size_bytes": _T9_EVIDENCE[1][1],
            "terminal_failure_hash": "29ca03ab230e644499fd28704dffa75c38091ad597a72cd6c7e90ef7dfd3ef78",
            "git_blob": _T9_EVIDENCE[1][3],
            "git_mode": "100644",
            "seal_commit": T9,
        },
        "official_invocation_count": 1,
        "exit_status": 1,
        "raw_file_count": 7,
        "decode_pass_count": 7,
        "market_prefix_comparison_count": 1,
        "mismatch_fraction_percent": "0.00148",
        "traceback_value_excerpt_emitted": True,
        "traceback_value_excerpt_restated_in_ledger": False,
        "generated_readback_decode_count": 0,
        "candidate_rows_opened": 0,
        "comparator_clock_rows_opened": 0,
        "feature_signal_schedule_or_interval_values_computed": 0,
        "economic_or_overlap_values_computed": 0,
    }
    current_s10 = {
        "identity": IDENTITY,
        "attempt_sentinel": {
            "path": config.attempt_path,
            "sha256": attempt_binding["sha256"],
            "size_bytes": attempt_binding["size_bytes"],
            "attempt_hash": attempt["attempt_hash"],
        },
        "official_invocation_count": 1,
        "raw_file_count": 7,
        "decode_pass_count": 8,
        "decode_passes": [
            "old_market",
            "replacement_market_date_scan",
            "replacement_market_tail",
            "funding",
            "premium",
            "old_open_interest",
            "binance_metrics_open_interest",
            "rank7_spot_premium_5m",
        ],
        "replacement_date_scan_count": 1,
        "replacement_tail_decode_count": 1,
        "generated_readback_decode_count": 2,
        "replacement_prefix_rows_selected": 0,
        "replacement_prefix_non_date_values_semantically_evaluated": 0,
        "replacement_tail_rows_selected": len(replacement_scan.tail_positions),
        "overlap_value_comparison_count": 0,
        "candidate_rows_opened": 0,
        "comparator_clock_rows_opened": 0,
        "feature_signal_schedule_or_interval_values_computed": 0,
        "economic_or_overlap_values_computed": 0,
    }
    source_value_rows_opened = sum(decoded_rows.values()) + len(replacement_scan.tail_positions)
    process_local = {
        "stage": "S10",
        "slot": 0,
        "invocation_count": 1,
        "source_files_opened": 7,
        "source_value_rows_opened": source_value_rows_opened,
        "candidate_rows_opened": 0,
        "comparator_clock_rows_opened": 0,
        "model_files_opened": 0,
        "runtime_modules_imported": 0,
        "pre2025_anchor_value_rows_opened": 0,
        "feature_signal_schedule_or_interval_values_computed": 0,
        "portfolio_economic_values_computed": 0,
        "economic_or_overlap_values_computed": 0,
    }
    prior_outputs = [
        "results/gross9_structural_clock_bundle_g9cb9_source_support_attempt_consumed_2026-07-31.json",
        "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb9_complete.csv.gz",
        "data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb9_complete.csv.gz",
        "configs/shadow/gross9_structural_clock_bundle_g9cb9_sources_2026-07-31.json",
        "results/gross9_structural_clock_bundle_g9cb9_source_support_2026-07-31.json",
    ]
    current_outputs = list(config.output_paths)
    if (
        set(prior_outputs).intersection(current_outputs)
        or attempt["attempt_hash"] == historical_s9["attempt_sentinel"]["attempt_hash"]
        or IDENTITY == historical_s9["identity"]
    ):
        _fail("G9CB-10 replay guard inputs differ")
    replay_guard = _with_hash(
        {
            "prior_identity": historical_s9["identity"],
            "current_identity": IDENTITY,
            "prior_attempt_hash": historical_s9["attempt_sentinel"]["attempt_hash"],
            "current_attempt_hash": attempt["attempt_hash"],
            "prior_terminal_failure_hash": historical_s9["terminal_failure_ledger"]["terminal_failure_hash"],
            "authority_commit": config.authority_commit,
            "authority_path": _A10_AUTHORITY[0],
            "terminal_evidence_commit": config.repository_parent,
            "terminal_failure_ledger_path": _T9_EVIDENCE[1][0],
            "implementation_commit": head,
            "implementation_paths": [
                "training/materialize_gross9_structural_clock_g9cb10_sources.py",
                "tests/test_materialize_gross9_structural_clock_g9cb10_sources.py",
            ],
            "prior_expected_output_paths": prior_outputs,
            "current_expected_output_paths": current_outputs,
        },
        "replay_guard_hash",
    )
    return _with_hash(
        {
            "schema_version": 1,
            "ledger_kind": "gross9_structural_clock_bundle_g9cb10_access_v1",
            "historical_s9": historical_s9,
            "current_s10": current_s10,
            "process_local": process_local,
            "replay_guard": replay_guard,
        },
        "access_ledger_hash",
    )


def _build_access_ledger(
    config: MaterializationConfig,
    head: str,
    attempt: Mapping[str, Any],
    attempt_binding: Mapping[str, Any],
    replacement_scan: ReplacementDateScan,
    decoded_rows: Mapping[str, int],
) -> dict[str, Any]:
    return _expected_access_ledger(
        config,
        head,
        attempt,
        attempt_binding,
        replacement_scan,
        decoded_rows,
    )


def _validate_access_ledger(
    ledger: Mapping[str, Any],
    config: MaterializationConfig,
    *,
    head: str,
    attempt: Mapping[str, Any],
    attempt_binding: Mapping[str, Any],
    replacement_scan: ReplacementDateScan,
    decoded_rows: Mapping[str, int],
) -> None:
    expected = _expected_access_ledger(
        config,
        head,
        attempt,
        attempt_binding,
        replacement_scan,
        decoded_rows,
    )
    try:
        actual_raw = canonical_json_bytes(ledger)
    except (TypeError, ValueError) as exc:
        raise SourceSupportFailure(
            "access ledger exact schema, type, order, or binding differs"
        ) from exc
    if actual_raw != canonical_json_bytes(expected):
        _fail("access ledger exact schema, type, order, or binding differs")


def materialize(config: MaterializationConfig, *, failpoint: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Run the source-support pipeline against official or synthetic bindings."""
    inject = failpoint or (lambda _name: None)
    head = config.repository_head
    if config.enforce_repository_gates:
        _validate_command_and_root(config)
        _validate_bytecode_first_gate(config.root)
        head = _validate_git_gate(config)
        _validate_t9_gate(config)
    _validate_output_absence(config)
    output_directory_baseline = _snapshot_output_directories(config)
    retained = _open_inputs(config, inject)
    try:
        inject("checkpoint:1")
        attempt = _attempt(config, head)
        attempt_raw = canonical_json_bytes(attempt)
        attempt_binding = _publish(config, config.attempt_path, attempt_raw, "attempt_sentinel", inject)
        inject("checkpoint:2")

        frames: dict[str, pd.DataFrame] = {}
        schemas: dict[str, tuple[str, ...]] = {}
        decoded_rows: dict[str, int] = {}
        replacement_scan: ReplacementDateScan | None = None
        for binding in config.inputs:
            if binding.name == "replacement_market":
                replacement_scan = _scan_replacement_dates(retained[binding.name], config, inject)
                frame, schema = _decode_replacement_tail(
                    retained[binding.name], replacement_scan, inject
                )
                decoded_rows[binding.name] = replacement_scan.row_count
            else:
                frame, schema = _decode_csv(retained[binding.name], inject)
                decoded_rows[binding.name] = len(frame)
            frames[binding.name] = frame
            schemas[binding.name] = schema
        if replacement_scan is None:
            _fail("replacement market decode pass is absent")
        market, oi, validation, normalized_rows = _transform(
            config, frames, schemas, replacement_scan
        )
        frames.clear()
        schemas.clear()
        del frame, schema
        inject("checkpoint:3")

        market_rows = len(market)
        market_first = market.date.iloc[0]
        market_last = market.date.iloc[-1]
        oi_rows = len(oi)
        oi_first = oi.date.iloc[0]
        oi_last = oi.date.iloc[-1]

        market_gzip, market_csv, market_frame_hash = serialize_csv_gzip(market)
        market_binding = _publish(config, config.market_output_path, market_gzip, "materialized_market", inject)
        del market_gzip
        inject("checkpoint:4")
        oi_gzip, oi_csv, oi_frame_hash = serialize_csv_gzip(oi)
        oi_binding = _publish(config, config.oi_output_path, oi_gzip, "materialized_open_interest", inject)
        del oi_gzip
        inject("checkpoint:5")

        market_digest, market_size, market_read = _readback_generated(
            config, config.market_output_path, failpoint=inject
        )
        if market_digest != market_binding["sha256"] or market_size != market_binding["size_bytes"]:
            _fail("generated market gzip readback binding differs")
        market_read["date"] = _dates(market_read.date, "materialized_market_readback")
        market_recsv, _ = serialize_csv_bytes(market_read)
        if market_recsv != market_csv or len(market_read) != market_rows:
            _fail("generated market logical-frame readback differs")
        del market, market_csv, market_read, market_recsv

        oi_digest, oi_size, oi_read = _readback_generated(
            config, config.oi_output_path, failpoint=inject
        )
        if oi_digest != oi_binding["sha256"] or oi_size != oi_binding["size_bytes"]:
            _fail("generated OI gzip readback binding differs")
        oi_read["date"] = _dates(oi_read.date, "materialized_oi_readback")
        oi_recsv, _ = serialize_csv_bytes(oi_read)
        if oi_recsv != oi_csv or len(oi_read) != oi_rows:
            _fail("generated OI logical-frame readback differs")
        del oi, oi_csv, oi_read, oi_recsv
        inject("checkpoint:6")

        manifest = _build_manifest(config, market_binding["sha256"], oi_binding["sha256"])
        manifest_raw = canonical_json_bytes(manifest)
        manifest_binding = _publish(config, config.manifest_output_path, manifest_raw, "source_manifest", inject)
        inject("checkpoint:7")

        raw_source_rows = []
        for binding in config.inputs:
            raw_source_rows.append({
                "decoded_rows": decoded_rows[binding.name], "mode_octal": f"{binding.mode:04o}",
                "name": binding.name, "normalized_rows": normalized_rows[binding.name],
                "path": binding.path, "path_type": "regular_file", "sha256": binding.sha256,
                "size_bytes": binding.size_bytes,
            })
        access_ledger = _build_access_ledger(
            config, head, attempt, attempt_binding, replacement_scan, decoded_rows
        )
        _validate_access_ledger(
            access_ledger,
            config,
            head=head,
            attempt=attempt,
            attempt_binding=attempt_binding,
            replacement_scan=replacement_scan,
            decoded_rows=decoded_rows,
        )
        support_core = {
            "access_ledger": access_ledger,
            "attempt_sentinel": {"attempt_hash": attempt["attempt_hash"], "path": config.attempt_path, "sha256": attempt_binding["sha256"], "size_bytes": attempt_binding["size_bytes"]},
            "identity": IDENTITY,
            "materialized_sources": {
                "market_5m": {"filesystem_mode_octal": "0444", "first_timestamp": _timestamp_z(market_first), "frame_hash": market_frame_hash, "gzip": {"compresslevel": 9, "embedded_filename": "", "mtime": 0}, "last_timestamp": _timestamp_z(market_last), "path": config.market_output_path, "path_type": "regular_file", "rows": market_rows, "schema": list(MARKET_SCHEMA), "sha256": market_binding["sha256"], "size_bytes": market_binding["size_bytes"]},
                "open_interest": {"filesystem_mode_octal": "0444", "first_timestamp": _timestamp_z(oi_first), "frame_hash": oi_frame_hash, "gzip": {"compresslevel": 9, "embedded_filename": "", "mtime": 0}, "last_timestamp": _timestamp_z(oi_last), "path": config.oi_output_path, "path_type": "regular_file", "rows": oi_rows, "schema": list(OI_SCHEMA), "sha256": oi_binding["sha256"], "size_bytes": oi_binding["size_bytes"]},
            },
            "raw_sources": raw_source_rows,
            "source_manifest": {"path": config.manifest_output_path, "sha256": manifest_binding["sha256"], "size_bytes": manifest_binding["size_bytes"]},
            "source_support_commit": head,
            "validation": validation,
            "version": VERSION,
        }
        support = _with_hash(support_core, "support_hash")
        support_raw = canonical_json_bytes(support)
        support_binding = _publish(config, config.support_output_path, support_raw, "source_support", inject)
        inject("checkpoint:8")

        for item in retained.values():
            _identity_check(item, "final", inject)
        expected_output_bindings = {
            config.attempt_path: attempt_binding,
            config.market_output_path: market_binding,
            config.oi_output_path: oi_binding,
            config.manifest_output_path: manifest_binding,
            config.support_output_path: support_binding,
        }
        for text in config.output_paths:
            fd = _open_nofollow_components(_path(config, text), os.O_RDONLY)
            try:
                info = os.fstat(fd)
                digest, size = _descriptor_hash(fd)
            finally:
                os.close(fd)
            expected = expected_output_bindings[text]
            if (
                not stat.S_ISREG(info.st_mode)
                or stat.S_IMODE(info.st_mode) != 0o444
                or info.st_nlink != 1
                or (info.st_dev, info.st_ino) != expected["inode"]
                or size != expected["size_bytes"]
                or digest != expected["sha256"]
            ):
                _fail(f"final output binding differs: {text}")
            inject(f"output:{text}:final")
        _validate_output_directory_delta(config, output_directory_baseline)
        if config.enforce_repository_gates:
            _validate_bytecode_first_gate(config.root)
        inject("checkpoint:9")
        return support
    finally:
        for item in retained.values():
            os.close(item.fd)


def main() -> int:
    materialize(official_config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
