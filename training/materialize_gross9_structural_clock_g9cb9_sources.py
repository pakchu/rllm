"""One-shot G9CB-9 source-support materializer.

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
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from training.gross9_structural_clock_primitives import (
    _merge_aux,
    normalise_funding_history_frame,
    normalise_premium_index_frame,
)


class SourceSupportFailure(RuntimeError):
    """Terminal failure for the consumed G9CB-9 source-support identity."""


IDENTITY = "G9CB-9-SOURCE-SUPPORT"
VERSION = "gross9_structural_clock_bundle_g9cb9_source_support_v1"
BRANCH = "codex/gross9-structural-clock-bundle-20260731"
A9 = "98fe1e95708ad095cf0727363c32a89e7d03ead6"
T8 = "4188f35caa2c491f7b12f400d0815ea3a1a6144b"

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
    repository_head: str = "synthetic-s9"
    repository_parent: str = T8
    authority_commit: str = A9
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
        attempt_path="results/gross9_structural_clock_bundle_g9cb9_source_support_attempt_consumed_2026-07-31.json",
        market_output_path="data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb9_complete.csv.gz",
        oi_output_path="data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb9_complete.csv.gz",
        manifest_output_path="configs/shadow/gross9_structural_clock_bundle_g9cb9_sources_2026-07-31.json",
        support_output_path="results/gross9_structural_clock_bundle_g9cb9_source_support_2026-07-31.json",
        inherited_manifest_path="configs/shadow/portfolio_added_alpha_signal_parity_sources_2026-07-16.json",
        old_last=pd.Timestamp("2026-05-31 15:00:00"),
        domain_end=pd.Timestamp("2026-06-01 00:00:00"),
        expected_old_rows=674785,
        expected_complete_rows=674892,
        expected_append_rows=107,
        enforce_repository_gates=True,
    )


def _fail(message: str) -> None:
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
    if len(parents) != 2 or parents[1] != T8:
        _fail("S9 direct parent differs")
    if _run_git(root, "rev-parse", f"{T8}^") != A9:
        _fail("T8/A9 topology differs")
    expected = [
        "A\ttests/test_materialize_gross9_structural_clock_g9cb9_sources.py",
        "A\ttraining/materialize_gross9_structural_clock_g9cb9_sources.py",
    ]
    actual = sorted(_run_git(root, "diff", "--name-status", T8, head).splitlines())
    if actual != expected:
        _fail("exact S9 two-file diff differs")
    for entry in expected:
        relative = entry.split("\t", 1)[1]
        indexed = _run_git(root, "ls-files", "-s", "--", relative).split()
        if len(indexed) < 4 or indexed[0] != "100644" or indexed[2] != "0" or indexed[3] != relative:
            _fail(f"S9 Git mode or index binding differs: {relative}")
    if _run_git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        _fail("index or worktree is not clean")
    return head


_T8_EVIDENCE = (
    ("results/gross9_structural_clock_bundle_g9cb8_attempt_consumed_2026-07-31.json", 3654, "024c89a4ec6590f656f0b0e092e49997e1661dede37d23932ee2cf3822f09ffe", "7459f27563c36a8a2cf2141e4add6b7c6b8cbb4e"),
    ("results/gross9_structural_clock_bundle_g9cb8_worker_capability_consumed_pass1_2026-07-31.json", 1766, "070baca2b4f04f61216e08c60a2a1176fef6b0d6fa9c9a87e6a5bf6058d0cf4d", "98ed78849c31dc26dc2f420aa43807a7ba75e5ad"),
)


def _is_forbidden_g9cb8_residue(name: str, retained_stage_name: str) -> bool:
    if name == retained_stage_name:
        return False
    return (
        name.startswith(".g9cb8-otmpfile-probe-")
        or name.startswith(".gross9-structural-clock-g9cb8-worker-")
        or (
            name.startswith(".gross9_structural_clock_bundle_g9cb8_")
            and ".stage-" in name
        )
    )


def _validate_t8_gate(config: MaterializationConfig) -> None:
    for relative, size, digest, blob in _T8_EVIDENCE:
        path = config.root / relative
        fd = _open_nofollow_components(path, os.O_RDONLY)
        try:
            info = os.fstat(fd)
            raw = _pread_complete(fd, info.st_size)
        finally:
            os.close(fd)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444 or info.st_nlink != 1 or info.st_size != size or hashlib.sha256(raw).hexdigest() != digest:
            _fail(f"T8 evidence binding differs: {relative}")
        row = _run_git(config.root, "ls-files", "-s", "--", relative).split()
        if row[:2] != ["100644", blob]:
            _fail(f"T8 Git binding differs: {relative}")
    stage = config.root / "results/.gross9-structural-clock-g9cb8-worker-b04b561d045e074567a96761"
    info = stage.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700 or any(stage.iterdir()):
        _fail("retained G9CB-8 pass-1 stage differs")
    absent = (
        "results/gross9_structural_clock_bundle_g9cb8_worker_capability_consumed_pass2_2026-07-31.json",
        "results/gross9_structural_clock_bundle_g9cb8_2026-07-31.csv.gz",
        "results/gross9_structural_clock_bundle_g9cb8_manifest_2026-07-31.json",
        "results/.g9cb8-bytecode-cache-disabled",
        "results/.gross9-structural-clock-g9cb8-worker-dcb23c75d25376df58352acb",
    )
    for relative in absent:
        path = config.root / relative
        if path.exists() or path.is_symlink():
            _fail(f"permanent G9CB-8 absence differs: {relative}")
    names = tuple(os.listdir(config.root / "results"))
    if any(_is_forbidden_g9cb8_residue(name, stage.name) for name in names):
        _fail("forbidden G9CB-8 helper or stage residue differs")


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


def _transform(config: MaterializationConfig, frames: Mapping[str, pd.DataFrame], schemas: Mapping[str, tuple[str, ...]]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, int]]:
    old_market_raw = frames["old_market"]
    replacement_raw = frames["replacement_market"]
    if schemas["old_market"] != MARKET_SCHEMA or schemas["replacement_market"] != MARKET_SCHEMA:
        _fail("market schema drift")
    old_market = _normalise_ordered(old_market_raw, "date", "old_market")
    replacement = _normalise_ordered(replacement_raw, "date", "replacement_market")
    if len(old_market) != config.expected_old_rows or old_market.date.iloc[-1] != config.old_last:
        _fail("old market row count or terminal timestamp differs")
    prefix = replacement.loc[replacement.date <= config.old_last]
    _exact_frame(old_market, prefix, "market logical prefix")
    market = replacement.loc[replacement.date < config.domain_end].reset_index(drop=True)
    required_last = config.domain_end - pd.Timedelta(minutes=5)
    if len(market) != config.expected_complete_rows or len(market) - len(old_market) != config.expected_append_rows:
        _fail("materialized market row counts differ")
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
        "market_prefix_exact": True,
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
        "old_market": len(old_market), "replacement_market": len(replacement),
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


def materialize(config: MaterializationConfig, *, failpoint: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Run the source-support pipeline against official or synthetic bindings."""
    inject = failpoint or (lambda _name: None)
    head = config.repository_head
    if config.enforce_repository_gates:
        _validate_command_and_root(config)
        _validate_bytecode_first_gate(config.root)
        head = _validate_git_gate(config)
        _validate_t8_gate(config)
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
        for binding in config.inputs:
            frame, schema = _decode_csv(retained[binding.name], inject)
            frames[binding.name] = frame
            schemas[binding.name] = schema
            decoded_rows[binding.name] = len(frame)
        market, oi, validation, normalized_rows = _transform(config, frames, schemas)
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
        support_core = {
            "access": {
                "candidate_rows_opened": 0, "comparator_clock_rows_opened": 0,
                "decoded_generated_readbacks": ["materialized_market", "materialized_open_interest"],
                "decoded_preexisting_sources": [row.name for row in config.inputs],
                "economic_or_overlap_values_computed": 0,
                "feature_signal_schedule_or_interval_values_computed": 0,
                "model_history_or_rex_values_opened": 0, "pre2025_anchor_value_rows_opened": 0,
                "raw_source_decode_count": 7, "readback_decode_count": 2,
            },
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
