"""Evaluate the frozen, outcome-blind TUSI-168 novelty gates.

The evaluator opens comparator clocks only after authenticating the promoted
TUSI source-support report and primary clock.  It never opens candidate market,
funding, return, PnL, or other outcome data.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from fractions import Fraction
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import secrets
import stat
import subprocess
import tempfile
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Iterator,
    Mapping,
    Sequence,
    cast,
    overload,
)

from training import build_tron_usdt_supply_events as source_builder
from training import (
    evaluate_ethereum_settlement_demand_impulse_novelty as esdi_novelty,
)
from training import (
    evaluate_tron_usdt_supply_impulse_source_support as source_support_evaluator,
)
from training import preregister_tron_usdt_supply_impulse as prereg


PROTOCOL_VERSION = "tron_usdt_supply_impulse_novelty_v1"
POLICY_ID = "TUSI-168"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION_PATH = prereg.DEFAULT_OUTPUT
PREREGISTRATION_SHA256 = (
    "54817044b8df76dc347ed64b6fe5f6f2dfdddcdb211bded4ba2b1af133d49067"
)
PREREGISTRATION_MANIFEST_HASH = (
    "d67cd1b67632ae92e9458395e729627a6f4c3b4b75ce97187653eac3a09e40c1"
)
ESDI_PREREGISTRATION_PATH = prereg.ESDI_PREREGISTRATION_PATH
ESDI_PREREGISTRATION_SHA256 = prereg.ESDI_PREREGISTRATION_SHA256
ESDI_PREREGISTRATION_MANIFEST_HASH = prereg.ESDI_MANIFEST_HASH
COMPARATOR_REGISTRY_SHA256 = prereg.ESDI_COMPARATOR_SUBTREE_SHA256
GROSS9_SUBTREE_SHA256 = prereg.ESDI_GROSS9_SUBTREE_SHA256
GROSS9_AUTHORITY_SHA256 = prereg.ESDI_GROSS9_AUTHORITY_SHA256
GROSS9_RUNTIME_CLOSURE_SHA256 = prereg.ESDI_RUNTIME_CLOSURE_SHA256

DEFAULT_SOURCE_SUPPORT_PATH = source_support_evaluator.DEFAULT_REPORT_OUTPUT
DEFAULT_PRIMARY_CLOCK_PATH = source_support_evaluator.DEFAULT_PRIMARY_OUTPUT
DEFAULT_CONTROL_CLOCK_PATH = source_support_evaluator.DEFAULT_CONTROLS_OUTPUT
DEFAULT_SOURCE_MANIFEST_PATH = source_support_evaluator.DEFAULT_SOURCE_MANIFEST
DEFAULT_SOURCE_CSV_PATH = source_support_evaluator.DEFAULT_SOURCE_CSV
DEFAULT_REPLAY_CLAIM_PATH = source_builder.REPLAY_CLAIM_PATH
DEFAULT_GROSS9_CLOCKS_PATH = (
    esdi_novelty.DEFAULT_GROSS9_CLOCKS_PATH
)
GROSS9_CLOCKS_PROTOCOL_VERSION = (
    esdi_novelty.GROSS9_CLOCKS_PROTOCOL_VERSION
)
DEFAULT_OUTPUT_PATH = Path(
    "results/tron_usdt_supply_impulse_novelty_2026-07-30.json"
)
DEFAULT_ATTEMPT_CLAIM_PATH = Path(
    "results/tron_usdt_supply_impulse_novelty_attempt_claim_2026-07-30.json"
)
ATTEMPT_CLAIM_PROTOCOL_VERSION = (
    "tron_usdt_supply_impulse_novelty_attempt_claim_v1"
)
GROSS9_DOMAIN = esdi_novelty.GROSS9_DOMAIN
GROSS9_SLEEVES = esdi_novelty.GROSS9_SLEEVES
MINIMUM_GATING_ENTRIES = 10
REPORT_KEYS = {
    "protocol_version",
    "policy_id",
    "status",
    "terminal",
    "decision",
    "preregistration",
    "attempt_claim",
    "source_support",
    "candidate_clock",
    "gross9_clock_artifact",
    "registry",
    "novelty",
    "evidence_boundary",
    "manifest_hash",
}

NOVELTY_EVIDENCE_BOUNDARY = {
    "source_support_report_bytes_opened": True,
    "source_support_primary_clock_bytes_opened": True,
    "comparator_clock_artifact_bytes_opened": True,
    "gross9_clock_artifact_bytes_opened": True,
    "candidate_market_rows_opened": False,
    "candidate_funding_rows_opened": False,
    "candidate_outcome_rows_opened": False,
    "candidate_returns_or_pnl_computed": False,
    "portfolio_return_or_pnl_metrics_computed": False,
    "network_calls": 0,
}
SOURCE_MANIFEST_KEYS = {
    "protocol_version",
    "source_only",
    "protocol_parent_commit",
    "replay_claim_commit",
    "replay_claim_sha256",
    "generation_commit",
    "chain",
    "source_range",
    "transports",
    "source_replay_schedule",
    "transport_exact_set_equal",
    "category_counts",
    "category_canonical_sha256",
    "global_log_count",
    "global_canonical_sha256",
    "event_counts",
    "event_count",
    "event_canonical_sha256",
    "year_counts",
    "source_csv_sha256",
    "receipt_count",
    "receipt_canonical_sha256",
    "header_count",
    "header_canonical_sha256",
    "common_finalized_head",
    "boundary_evidence",
    "protocol_guards",
    "outcome_access",
    "source_integrity",
    "manifest_hash",
}

NoveltyTerminalError = esdi_novelty.NoveltyTerminalError
SignedInterval = esdi_novelty.SignedInterval
ComparatorClock = esdi_novelty.ComparatorClock
Gross9SleeveClocks = esdi_novelty.Gross9SleeveClocks


@dataclass(frozen=True)
class VerifiedSourceSupport:
    path: Path
    raw_bytes: bytes
    sha256: str
    manifest_hash: str
    payload: Mapping[str, Any]
    production_authenticated: bool = False
    provenance: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class InjectedVerifiedGross9Fixture:
    """Synthetic-only authentication evidence for injected Gross9 bytes."""

    source_support_binding: Mapping[str, str]
    frozen_contract_validation: Mapping[str, Any]
    authority_hash: str


@dataclass(frozen=True)
class VerifiedGross9Clocks:
    path: Path
    raw_bytes: bytes
    sha256: str
    manifest_hash: str
    authority_hash: str
    clocks: Mapping[str, tuple[SignedInterval, ...]]
    payload: Mapping[str, Any]
    authentication_mode: str


@dataclass(frozen=True)
class VerifiedCandidateClock(Sequence[SignedInterval]):
    path: Path
    raw_bytes: bytes
    sha256: str
    source_support_sha256: str
    intervals: tuple[SignedInterval, ...]
    authentication_mode: str

    @overload
    def __getitem__(self, index: int) -> SignedInterval: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[SignedInterval, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> SignedInterval | tuple[SignedInterval, ...]:
        return self.intervals[index]

    def __len__(self) -> int:
        return len(self.intervals)

    def __iter__(self) -> Iterator[SignedInterval]:
        return iter(self.intervals)


@dataclass(frozen=True)
class VerifiedComparatorClocks(Mapping[str, ComparatorClock]):
    clocks: Mapping[str, ComparatorClock]
    artifact_bytes: Mapping[str, bytes]
    artifact_sha256: Mapping[str, str]
    registry_sha256: str
    authentication_mode: str

    def __getitem__(self, key: str) -> ComparatorClock:
        return self.clocks[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.clocks)

    def __len__(self) -> int:
        return len(self.clocks)


@dataclass(frozen=True)
class AuthenticatedNoveltyInputs:
    registration: Mapping[str, Any]
    source_support: VerifiedSourceSupport
    candidate: VerifiedCandidateClock
    comparators: VerifiedComparatorClocks
    gross9_artifact: VerifiedGross9Clocks
    attempt_claim: Mapping[str, Any]
    protocol_paths: tuple[Path, ...]
    production: bool


ComparatorLoader = Callable[
    [Mapping[str, Mapping[str, Any]]], Mapping[str, ComparatorClock]
]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    return prereg.canonical_hash(payload)


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _decode_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise NoveltyTerminalError(f"TUSI-168 invalid {label} JSON") from error
    if not isinstance(payload, dict):
        raise NoveltyTerminalError(f"TUSI-168 {label} is not an object")
    return payload


def _expected_preregistration_binding() -> dict[str, str]:
    return {
        "path": str(PREREGISTRATION_PATH),
        "sha256": PREREGISTRATION_SHA256,
        "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
    }


def _git(
    *arguments: str,
    repository_root: str | Path = REPOSITORY_ROOT,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        input=input_bytes,
        capture_output=True,
        check=False,
    )


def _canonical_relative_path(path: str | Path, expected: Path, label: str) -> Path:
    supplied = os.fspath(path)
    canonical = expected.as_posix()
    if (
        supplied != canonical
        or expected.is_absolute()
        or ".." in expected.parts
        or expected.name in {"", ".", ".."}
    ):
        raise NoveltyTerminalError(
            f"TUSI-168 {label} must use exact canonical relative path {canonical}"
        )
    return expected


def _open_regular_nofollow(
    relative: Path,
    *,
    repository_root: str | Path,
    label: str,
) -> int:
    root = Path(repository_root)
    if not root.is_absolute():
        root = root.resolve()
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    if not getattr(os, "O_DIRECTORY", 0) or not getattr(os, "O_NOFOLLOW", 0):
        raise NoveltyTerminalError(
            "TUSI-168 secure no-follow path traversal is unavailable"
        )
    descriptor = os.open(root, directory_flags)
    try:
        for part in relative.parent.parts:
            next_descriptor = os.open(
                part,
                directory_flags,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        file_descriptor = os.open(
            relative.name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=descriptor,
        )
        if not stat.S_ISREG(os.fstat(file_descriptor).st_mode):
            os.close(file_descriptor)
            raise NoveltyTerminalError(
                f"TUSI-168 {label} is not a regular file"
            )
        return file_descriptor
    except OSError as error:
        raise NoveltyTerminalError(
            f"TUSI-168 {label} is missing or unsafe"
        ) from error
    finally:
        os.close(descriptor)


def _read_canonical_regular(
    path: str | Path,
    expected: Path,
    label: str,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> bytes:
    relative = _canonical_relative_path(path, expected, label)
    descriptor = _open_regular_nofollow(
        relative,
        repository_root=repository_root,
        label=label,
    )
    try:
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _digest_canonical_regular(
    path: str | Path,
    expected: Path,
    label: str,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> tuple[str, int]:
    """Hash opaque bytes from one no-follow descriptor without parsing rows."""

    relative = _canonical_relative_path(path, expected, label)
    descriptor = _open_regular_nofollow(
        relative,
        repository_root=repository_root,
        label=label,
    )
    digest = hashlib.sha256()
    byte_count = 0
    try:
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest(), byte_count


def _read_regular_path_nofollow(
    path: str | Path,
    label: str,
    *,
    repository_root: str | Path,
) -> bytes:
    supplied = os.fspath(path)
    candidate = Path(supplied)
    if (
        supplied != candidate.as_posix()
        or ".." in candidate.parts
        or candidate.name in {"", ".", ".."}
    ):
        raise NoveltyTerminalError(f"TUSI-168 {label} path is not canonical")
    if candidate.is_absolute():
        root = Path(candidate.anchor)
        relative = Path(*candidate.parts[1:])
    else:
        root = Path(repository_root)
        relative = candidate
    descriptor = _open_regular_nofollow(
        relative,
        repository_root=root,
        label=label,
    )
    try:
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _require_canonical_committed_clean(
    path: str | Path,
    expected: Path,
    label: str,
    raw: bytes,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> Path:
    _canonical_relative_path(path, expected, label)
    tracked = _git(
        "ls-files",
        "--error-unmatch",
        "--",
        expected.as_posix(),
        repository_root=repository_root,
    )
    status = _git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        expected.as_posix(),
        repository_root=repository_root,
    )
    head_blob = _git(
        "rev-parse",
        f"HEAD:{expected.as_posix()}",
        repository_root=repository_root,
    )
    supplied_blob = _git(
        "hash-object",
        "--stdin",
        repository_root=repository_root,
        input_bytes=raw,
    )
    if (
        tracked.returncode != 0
        or status.returncode != 0
        or status.stdout
        or head_blob.returncode != 0
        or supplied_blob.returncode != 0
        or head_blob.stdout.strip() != supplied_blob.stdout.strip()
    ):
        raise NoveltyTerminalError(
            f"TUSI-168 {label} must be committed and clean"
        )
    descriptor = _open_regular_nofollow(
        expected,
        repository_root=repository_root,
        label=label,
    )
    os.close(descriptor)
    return expected


def verify_preregistration(
    path: str | Path = PREREGISTRATION_PATH,
    *,
    production: bool = True,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    try:
        if production:
            raw = _read_canonical_regular(
                path,
                PREREGISTRATION_PATH,
                "preregistration",
                repository_root=repository_root,
            )
        else:
            raw = Path(path).read_bytes()
    except OSError as error:
        raise NoveltyTerminalError("TUSI-168 preregistration is unreadable") from error
    if production:
        _require_canonical_committed_clean(
            path,
            PREREGISTRATION_PATH,
            "preregistration",
            raw,
            repository_root=repository_root,
        )
    if sha256_bytes(raw) != PREREGISTRATION_SHA256:
        raise NoveltyTerminalError("TUSI-168 preregistration file hash drift")
    payload = _decode_json_bytes(raw, "preregistration")
    if raw != prereg.canonical_manifest_bytes(payload):
        raise NoveltyTerminalError("TUSI-168 preregistration serialization drift")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if (
        payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH
        or canonical_hash(core) != PREREGISTRATION_MANIFEST_HASH
        or payload.get("policy_id") != POLICY_ID
    ):
        raise NoveltyTerminalError("TUSI-168 preregistration manifest drift")
    prereg.validate_manifest(payload)
    _validate_registration_authorities(payload)
    return payload


def _validate_registration_authorities(
    registration: Mapping[str, Any],
) -> None:
    try:
        registry = registration["novelty"]["frozen_comparator_artifacts"]
        registry_authority = registration["novelty"]["registry_authority"]
        gross9 = registration["gross9"]
        authority = gross9["authority"]
        closure = authority["runtime_code_closure"]
        esdi_binding = gross9["esdi_artifact_binding"]
    except (KeyError, TypeError) as error:
        raise NoveltyTerminalError(
            "TUSI-168 preregistered novelty authority is missing"
        ) from error
    if (
        registration.get("policy_id") != POLICY_ID
        or registration.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH
        or canonical_hash(
            {
                key: value
                for key, value in registration.items()
                if key != "manifest_hash"
            }
        )
        != PREREGISTRATION_MANIFEST_HASH
        or not isinstance(registry, Mapping)
        or len(registry) != 18
        or canonical_hash(registry) != COMPARATOR_REGISTRY_SHA256
        or registry_authority.get("canonical_compact_sorted_sha256")
        != COMPARATOR_REGISTRY_SHA256
        or canonical_hash(authority) != GROSS9_AUTHORITY_SHA256
        or canonical_hash(closure) != GROSS9_RUNTIME_CLOSURE_SHA256
        or esdi_binding.get("path") != str(ESDI_PREREGISTRATION_PATH)
        or esdi_binding.get("file_sha256") != ESDI_PREREGISTRATION_SHA256
        or esdi_binding.get("manifest_hash")
        != ESDI_PREREGISTRATION_MANIFEST_HASH
        or esdi_binding.get("gross9_subtree_sha256")
        != GROSS9_SUBTREE_SHA256
        or esdi_binding.get("authority_subtree_sha256")
        != GROSS9_AUTHORITY_SHA256
        or esdi_binding.get("runtime_closure_subtree_sha256")
        != GROSS9_RUNTIME_CLOSURE_SHA256
    ):
        raise NoveltyTerminalError(
            "TUSI-168 comparator or Gross9 authority drift"
        )


def require_passed_source_support(payload: Mapping[str, Any]) -> None:
    """Require the exact promoted source-support PASS contract."""

    if not isinstance(payload, Mapping):
        raise NoveltyTerminalError("TUSI-168 source-support artifact is invalid")
    core = {
        key: _thaw_json(value)
        for key, value in payload.items()
        if key != "manifest_hash"
    }
    if payload.get("manifest_hash") != canonical_hash(core):
        raise NoveltyTerminalError(
            "TUSI-168 source-support internal manifest drift"
        )
    try:
        raw_counts = payload["raw_candidate_counts"]
        accepted_counts = payload["accepted_clock_counts"]
        source_support_evaluator._validate_report_schema(
            _thaw_json(payload),
            primary_counts={"primary": accepted_counts["primary"]},
            control_counts={
                name: accepted_counts[name]
                for name in source_support_evaluator.CONTROL_ORDER[1:]
            },
            artifact_eligible_authorized=True,
        )
    except (KeyError, TypeError, RuntimeError) as error:
        raise NoveltyTerminalError(
            "TUSI-168 source-support exact schema drift"
        ) from error
    if (
        set(raw_counts) != set(source_support_evaluator.CONTROL_ORDER)
        or payload.get("protocol_version")
        != source_support_evaluator.PROTOCOL_VERSION
        or payload.get("policy_id") != POLICY_ID
        or payload.get("status") != "source_support_passed"
        or payload.get("terminal") is not True
        or payload.get("artifact_eligible") is not True
        or payload.get("support_passed") is not True
        or payload.get("decision") != "SOURCE_SUPPORT_PASS"
        or payload.get("registration")
        != {
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
            "mode": "artifact",
        }
        or payload.get("source_support_precedes_novelty") is not True
        or payload.get(
            "novelty_comparator_market_or_outcome_artifacts_opened"
        )
        is not False
        or any(value is not True for value in payload["support_checks"].values())
    ):
        raise NoveltyTerminalError(
            "TUSI-168 source support did not pass before comparator access"
        )


def _git_output(
    *arguments: str,
    repository_root: str | Path,
    label: str,
) -> bytes:
    completed = _git(*arguments, repository_root=repository_root)
    if completed.returncode != 0:
        raise NoveltyTerminalError(f"TUSI-168 Git validation failed: {label}")
    return completed.stdout


def _require_metadata_only_committed_clean(
    relative: Path,
    *,
    repository_root: str | Path,
    label: str,
    open_file: bool = True,
) -> str:
    _canonical_relative_path(relative.as_posix(), relative, label)
    if _git(
        "ls-files",
        "--error-unmatch",
        "--",
        relative.as_posix(),
        repository_root=repository_root,
    ).returncode:
        raise NoveltyTerminalError(f"TUSI-168 {label} is not tracked")
    status = _git_output(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        relative.as_posix(),
        repository_root=repository_root,
        label=f"{label} status",
    )
    if status:
        raise NoveltyTerminalError(f"TUSI-168 {label} is not HEAD-clean")
    if open_file:
        descriptor = _open_regular_nofollow(
            relative,
            repository_root=repository_root,
            label=label,
        )
        os.close(descriptor)
    else:
        directory_descriptor = _open_parent_directory(
            relative,
            repository_root=repository_root,
            label=label,
        )
        try:
            metadata = os.stat(
                relative.name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise NoveltyTerminalError(
                f"TUSI-168 {label} is missing or unsafe"
            ) from error
        finally:
            os.close(directory_descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise NoveltyTerminalError(
                f"TUSI-168 {label} is not a regular file"
            )
    return (
        _git_output(
            "rev-parse",
            f"HEAD:{relative.as_posix()}",
            repository_root=repository_root,
            label=f"{label} HEAD blob",
        )
        .decode("ascii")
        .strip()
    )


def _immutable_add_commit(
    relative: Path,
    *,
    claim_commit: str,
    repository_root: str | Path,
    allow_claim_commit: bool = False,
) -> str:
    additions = [
        line
        for line in _git_output(
            "log",
            "--format=%H",
            "--diff-filter=A",
            "HEAD",
            "--",
            relative.as_posix(),
            repository_root=repository_root,
            label=f"{relative} addition history",
        )
        .decode("ascii")
        .splitlines()
        if line
    ]
    if len(additions) != 1:
        raise NoveltyTerminalError(
            f"TUSI-168 {relative} must have exactly one immutable addition"
        )
    addition = additions[0]
    ancestor = _git(
        "merge-base",
        "--is-ancestor",
        claim_commit,
        addition,
        repository_root=repository_root,
    )
    if ancestor.returncode != 0 or (
        addition == claim_commit and not allow_claim_commit
    ):
        raise NoveltyTerminalError(
            f"TUSI-168 {relative} was not added after the replay claim"
        )
    later_changes = _git_output(
        "log",
        "--format=%H",
        "--diff-filter=MDRCT",
        f"{addition}..HEAD",
        "--",
        relative.as_posix(),
        repository_root=repository_root,
        label=f"{relative} immutable history",
    )
    if later_changes.strip():
        raise NoveltyTerminalError(
            f"TUSI-168 {relative} changed after its immutable addition"
        )
    return addition


def _validate_source_manifest_metadata(
    raw: bytes,
) -> dict[str, Any]:
    try:
        manifest = source_builder._decode_canonical_manifest(raw)
        source_support_evaluator._validate_source_generation_commit(
            manifest, production=True
        )
        source_support_evaluator._validate_boundary_evidence(
            manifest.get("boundary_evidence"), production=True
        )
    except (RuntimeError, source_builder.TerminalSourceFailure) as error:
        raise NoveltyTerminalError(
            "TUSI-168 source manifest metadata authentication failed"
        ) from error
    expected_source_range = source_builder._source_range_manifest(
        source_builder.frozen_chunks()
    )
    category_counts = manifest.get("category_counts")
    category_hashes = manifest.get("category_canonical_sha256")
    if (
        set(manifest) != SOURCE_MANIFEST_KEYS
        or manifest.get("protocol_version") != source_builder.PROTOCOL_VERSION
        or manifest.get("source_only") is not True
        or manifest.get("generation_commit")
        != source_builder.PRODUCTION_GENERATION_COMMIT
        or manifest.get("source_range") != expected_source_range
        or manifest.get("chain")
        != {
            "name": "TRON mainnet",
            "chain_id": source_builder.CHAIN_ID_HEX,
            "usdt_contract_base58": source_builder.USDT_CONTRACT_BASE58,
            "usdt_contract_evm": source_builder.USDT_CONTRACT,
        }
        or manifest.get("transports")
        != [dict(item) for item in source_builder.SANITIZED_TRANSPORTS]
        or manifest.get("transport_exact_set_equal") is not True
        or manifest.get("source_replay_schedule")
        != {
            "inter_batch_throttle_seconds": (
                source_builder.PRODUCTION_THROTTLE_SECONDS
            ),
            "maximum_batch_by_role": dict(source_builder.TRANSPORT_MAX_BATCH),
            "rpc_methods": sorted(source_builder.RPC_METHODS),
        }
        or not isinstance(category_counts, Mapping)
        or set(category_counts) != set(source_builder.CATEGORIES)
        or any(
            type(value) is not int or value < 0
            for value in category_counts.values()
        )
        or not isinstance(category_hashes, Mapping)
        or set(category_hashes) != set(source_builder.CATEGORIES)
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in category_hashes.values()
        )
        or type(manifest.get("event_count")) is not int
        or manifest["event_count"] <= 0
        or manifest.get("source_integrity")
        != source_builder.ZERO_SOURCE_INTEGRITY
        or manifest.get("protocol_guards")
        != {
            "retry_backoff_fallback_resume": False,
            "response_dependent_sleep": False,
            "deprecate_terminal": True,
            "market_policy_performance_opened": False,
        }
        or manifest.get("outcome_access")
        != {
            "btc_market_rows_opened": 0,
            "funding_rows_opened": 0,
            "returns_opened": 0,
            "pnl_opened": 0,
            "cagr_opened": 0,
            "strict_mdd_opened": 0,
            "outcomes_opened": 0,
        }
    ):
        raise NoveltyTerminalError(
            "TUSI-168 source manifest metadata contract drift"
        )
    return manifest


def authenticate_production_source_support(
    report: Mapping[str, Any],
    report_raw: bytes,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
    protocol_paths: Sequence[Path] = source_builder.PROTOCOL_PATHS,
) -> dict[str, Any]:
    """Authenticate provenance while treating source CSV bytes as opaque."""

    source_contract = cast(Mapping[str, Any], report["source_contract"])
    if (
        source_contract.get("source_csv_path") != DEFAULT_SOURCE_CSV_PATH.as_posix()
        or source_contract.get("source_manifest_path")
        != DEFAULT_SOURCE_MANIFEST_PATH.as_posix()
    ):
        raise NoveltyTerminalError(
            "TUSI-168 source-support direct artifact paths drift"
        )
    _require_canonical_committed_clean(
        DEFAULT_SOURCE_SUPPORT_PATH.as_posix(),
        DEFAULT_SOURCE_SUPPORT_PATH,
        "source-support report",
        report_raw,
        repository_root=repository_root,
    )
    manifest_raw = _read_canonical_regular(
        DEFAULT_SOURCE_MANIFEST_PATH.as_posix(),
        DEFAULT_SOURCE_MANIFEST_PATH,
        "source manifest",
        repository_root=repository_root,
    )
    _require_canonical_committed_clean(
        DEFAULT_SOURCE_MANIFEST_PATH.as_posix(),
        DEFAULT_SOURCE_MANIFEST_PATH,
        "source manifest",
        manifest_raw,
        repository_root=repository_root,
    )
    manifest = _validate_source_manifest_metadata(manifest_raw)
    claim_raw = _read_canonical_regular(
        DEFAULT_REPLAY_CLAIM_PATH.as_posix(),
        DEFAULT_REPLAY_CLAIM_PATH,
        "source replay claim",
        repository_root=repository_root,
    )
    _require_canonical_committed_clean(
        DEFAULT_REPLAY_CLAIM_PATH.as_posix(),
        DEFAULT_REPLAY_CLAIM_PATH,
        "source replay claim",
        claim_raw,
        repository_root=repository_root,
    )
    claim = _decode_json_bytes(claim_raw, "source replay claim")
    seal = claim.get("protocol_seal")
    if (
        not isinstance(seal, Mapping)
        or claim
        != source_builder._claim_payload(
            seal, source_builder.SANITIZED_TRANSPORTS
        )
        or claim_raw
        != source_builder._canonical_json_bytes(claim, trailing_lf=True)
    ):
        raise NoveltyTerminalError("TUSI-168 source replay claim drift")
    claim_commit = manifest.get("replay_claim_commit")
    protocol_parent = manifest.get("protocol_parent_commit")
    if (
        not isinstance(claim_commit, str)
        or not isinstance(protocol_parent, str)
        or manifest.get("replay_claim_sha256") != sha256_bytes(claim_raw)
        or claim.get("protocol_parent_commit") != protocol_parent
        or seal.get("git_head") != protocol_parent
    ):
        raise NoveltyTerminalError(
            "TUSI-168 source manifest/replay-claim direct binding drift"
        )
    parent_line = (
        _git_output(
            "rev-list",
            "--parents",
            "-n",
            "1",
            claim_commit,
            repository_root=repository_root,
            label="replay claim parent",
        )
        .decode("ascii")
        .strip()
        .split()
    )
    changed = [
        line
        for line in _git_output(
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "-r",
            claim_commit,
            repository_root=repository_root,
            label="replay claim commit contents",
        )
        .decode("utf-8")
        .splitlines()
        if line
    ]
    committed_claim = _git_output(
        "show",
        f"{claim_commit}:{DEFAULT_REPLAY_CLAIM_PATH.as_posix()}",
        repository_root=repository_root,
        label="replay claim committed bytes",
    )
    if (
        parent_line != [claim_commit, protocol_parent]
        or changed != [f"A\t{DEFAULT_REPLAY_CLAIM_PATH.as_posix()}"]
        or committed_claim != claim_raw
    ):
        raise NoveltyTerminalError(
            "TUSI-168 replay claim is not an immutable claim-only commit"
        )
    try:
        source_builder.validate_protocol_seal(
            seal,
            repository_root=repository_root,
            protocol_paths=protocol_paths,
        )
    except source_builder.TerminalSourceFailure as error:
        raise NoveltyTerminalError(
            "TUSI-168 replay-claim protocol seal no longer matches HEAD"
        ) from error
    if (
        source_contract.get("source_manifest_sha256")
        != sha256_bytes(manifest_raw)
        or source_contract.get("source_manifest_hash")
        != manifest.get("manifest_hash")
        or source_contract.get("source_csv_sha256")
        != manifest.get("source_csv_sha256")
        or source_contract.get("rows") != manifest.get("event_count")
        or source_contract.get("source_integrity")
        != manifest.get("source_integrity")
    ):
        raise NoveltyTerminalError(
            "TUSI-168 source-support report direct source binding drift"
        )
    source_csv_sha256, source_csv_bytes = _digest_canonical_regular(
        DEFAULT_SOURCE_CSV_PATH.as_posix(),
        DEFAULT_SOURCE_CSV_PATH,
        "source CSV blob",
        repository_root=repository_root,
    )
    if (
        manifest.get("source_csv_sha256") != source_csv_sha256
        or source_contract.get("source_csv_sha256") != source_csv_sha256
        or source_contract.get("source_csv_bytes") != source_csv_bytes
    ):
        raise NoveltyTerminalError(
            "TUSI-168 opaque source CSV blob binding drift"
        )
    tracked_paths = (
        DEFAULT_REPLAY_CLAIM_PATH,
        DEFAULT_SOURCE_CSV_PATH,
        DEFAULT_SOURCE_MANIFEST_PATH,
        DEFAULT_PRIMARY_CLOCK_PATH,
        DEFAULT_CONTROL_CLOCK_PATH,
        DEFAULT_SOURCE_SUPPORT_PATH,
    )
    for relative in tracked_paths:
        _require_metadata_only_committed_clean(
            relative,
            repository_root=repository_root,
            label=relative.as_posix(),
            open_file=relative != DEFAULT_SOURCE_CSV_PATH,
        )
    additions = {
        relative: _immutable_add_commit(
            relative,
            claim_commit=claim_commit,
            repository_root=repository_root,
            allow_claim_commit=relative == DEFAULT_REPLAY_CLAIM_PATH,
        )
        for relative in tracked_paths
    }
    if (
        additions[DEFAULT_REPLAY_CLAIM_PATH] != claim_commit
        or additions[DEFAULT_SOURCE_CSV_PATH]
        != additions[DEFAULT_SOURCE_MANIFEST_PATH]
        or additions[DEFAULT_PRIMARY_CLOCK_PATH]
        != additions[DEFAULT_CONTROL_CLOCK_PATH]
        or additions[DEFAULT_PRIMARY_CLOCK_PATH]
        != additions[DEFAULT_SOURCE_SUPPORT_PATH]
    ):
        raise NoveltyTerminalError(
            "TUSI-168 immutable source/support artifact publication history drift"
        )
    return {
        "replay_claim_commit": claim_commit,
        "protocol_parent_commit": protocol_parent,
        "protocol_seal_hash": seal["seal_hash"],
        "source_artifact_add_commit": additions[DEFAULT_SOURCE_MANIFEST_PATH],
        "support_artifact_add_commit": additions[DEFAULT_SOURCE_SUPPORT_PATH],
        "source_csv_sha256": source_csv_sha256,
        "source_csv_bytes": source_csv_bytes,
        "source_manifest_sha256": sha256_bytes(manifest_raw),
        "source_manifest_hash": manifest["manifest_hash"],
    }


def parse_passed_source_support_bytes(
    raw: bytes,
    *,
    path: str | Path,
    production: bool,
    repository_root: str | Path = REPOSITORY_ROOT,
    protocol_paths: Sequence[Path] = source_builder.PROTOCOL_PATHS,
) -> VerifiedSourceSupport:
    if not isinstance(raw, bytes):
        raise NoveltyTerminalError("TUSI-168 source-support bytes are invalid")
    if production:
        artifact_path = _require_canonical_committed_clean(
            path,
            DEFAULT_SOURCE_SUPPORT_PATH,
            "source-support report",
            raw,
            repository_root=repository_root,
        )
    else:
        artifact_path = Path(path)
    payload = _decode_json_bytes(raw, "source-support report")
    try:
        canonical = source_support_evaluator._json_bytes(payload)
    except (TypeError, ValueError, RuntimeError) as error:
        raise NoveltyTerminalError(
            "TUSI-168 source-support serialization drift"
        ) from error
    if raw != canonical:
        raise NoveltyTerminalError(
            "TUSI-168 source-support serialization drift"
        )
    require_passed_source_support(payload)
    provenance = (
        authenticate_production_source_support(
            payload,
            raw,
            repository_root=repository_root,
            protocol_paths=protocol_paths,
        )
        if production
        else None
    )
    return VerifiedSourceSupport(
        path=artifact_path,
        raw_bytes=raw,
        sha256=sha256_bytes(raw),
        manifest_hash=str(payload["manifest_hash"]),
        payload=_freeze_json(payload),
        production_authenticated=production,
        provenance=_freeze_json(provenance) if provenance is not None else None,
    )


def load_passed_source_support(
    path: str | Path = DEFAULT_SOURCE_SUPPORT_PATH,
    *,
    production: bool = True,
    repository_root: str | Path = REPOSITORY_ROOT,
    protocol_paths: Sequence[Path] = source_builder.PROTOCOL_PATHS,
) -> VerifiedSourceSupport:
    try:
        raw = (
            _read_canonical_regular(
                path,
                DEFAULT_SOURCE_SUPPORT_PATH,
                "source-support report",
                repository_root=repository_root,
            )
            if production
            else Path(path).read_bytes()
        )
    except OSError as error:
        raise NoveltyTerminalError(
            "TUSI-168 source-support report is unreadable"
        ) from error
    return parse_passed_source_support_bytes(
        raw,
        path=path,
        production=production,
        repository_root=repository_root,
        protocol_paths=protocol_paths,
    )


def frozen_registry(
    registration: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    _validate_registration_authorities(registration)
    registry = _thaw_json(
        registration["novelty"]["frozen_comparator_artifacts"]
    )
    validate_registry(registry)
    return registry


def _expected_comparator_contracts(
    registry: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}
    for artifact_name, spec in registry.items():
        if spec.get("capability") == "directional_interval":
            capability_groups = {
                group: "directional_interval"
                for group in (
                    spec["groups"] if spec.get("group_column") else [None]
                )
            }
        else:
            capability_groups = {
                **{
                    group: "directional_interval"
                    for group in spec["directional_interval_groups"]
                },
                **{
                    group: "timestamp_only"
                    for group in spec["timestamp_only_groups"]
                },
            }
        for group, capability in capability_groups.items():
            comparator_id = (
                artifact_name if group is None else f"{artifact_name}:{group}"
            )
            contracts[comparator_id] = {
                "artifact_name": artifact_name,
                "group": group,
                "capability": capability,
                "comparison_domain": list(spec["comparison_domain"]),
            }
    return contracts


def validate_registry(
    registry: Mapping[str, Mapping[str, Any]],
    expected: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    frozen = prereg.frozen_comparator_registry() if expected is None else expected
    if canonical_hash(frozen) != COMPARATOR_REGISTRY_SHA256:
        raise NoveltyTerminalError("TUSI-168 comparator authority drift")
    esdi_novelty.validate_registry(registry, frozen)


def _read_comparator_artifact_bytes(
    artifact_name: str,
    spec: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> bytes:
    supplied = spec.get("path")
    if not isinstance(supplied, str):
        raise NoveltyTerminalError(
            f"TUSI-168 comparator path is invalid: {artifact_name}"
        )
    expected = Path(supplied)
    _canonical_relative_path(
        supplied,
        expected,
        f"comparator artifact {artifact_name}",
    )
    return _read_canonical_regular(
        supplied,
        expected,
        f"comparator artifact {artifact_name}",
        repository_root=repository_root,
    )


def _decode_comparator_csv_bytes(
    artifact_name: str,
    spec: Mapping[str, Any],
    compressed: bytes,
) -> list[Mapping[str, str]]:
    if sha256_bytes(compressed) != spec.get("sha256"):
        raise NoveltyTerminalError(
            f"TUSI-168 comparator artifact hash drift: {artifact_name}"
        )
    try:
        decompressed = gzip.decompress(compressed)
    except (OSError, EOFError) as error:
        raise NoveltyTerminalError(
            f"TUSI-168 comparator gzip invalid: {artifact_name}"
        ) from error
    header = decompressed.splitlines(keepends=True)[:1]
    if not header or sha256_bytes(header[0]) != spec.get("header_line_sha256"):
        raise NoveltyTerminalError(
            f"TUSI-168 comparator header hash drift: {artifact_name}"
        )
    try:
        text = decompressed.decode("utf-8")
    except UnicodeDecodeError as error:
        raise NoveltyTerminalError(
            f"TUSI-168 comparator is not UTF-8: {artifact_name}"
        ) from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    columns = reader.fieldnames
    required = spec.get("required_columns")
    if (
        columns is None
        or len(columns) != len(set(columns))
        or not isinstance(required, list)
        or not set(required).issubset(columns)
    ):
        raise NoveltyTerminalError(
            f"TUSI-168 comparator required columns missing: {artifact_name}"
        )
    try:
        physical_rows = list(reader)
    except csv.Error as error:
        raise NoveltyTerminalError(
            f"TUSI-168 comparator CSV invalid: {artifact_name}"
        ) from error
    if any(None in row or any(value is None for value in row.values()) for row in physical_rows):
        raise NoveltyTerminalError(
            f"TUSI-168 comparator row width drift: {artifact_name}"
        )
    return cast(list[Mapping[str, str]], physical_rows)


def _comparator_clocks_from_artifact_bytes(
    registry: Mapping[str, Mapping[str, Any]],
    artifact_bytes: Mapping[str, bytes],
) -> dict[str, ComparatorClock]:
    if set(artifact_bytes) != set(registry):
        raise NoveltyTerminalError(
            "TUSI-168 comparator artifact byte inventory drift"
        )
    clocks: dict[str, ComparatorClock] = {}
    for artifact_name, spec in registry.items():
        physical_rows = _decode_comparator_csv_bytes(
            artifact_name,
            spec,
            artifact_bytes[artifact_name],
        )
        rows = esdi_novelty._filtered_rows(physical_rows, spec["filters"])
        if not rows:
            raise NoveltyTerminalError(
                f"TUSI-168 comparator has no rows after filters: {artifact_name}"
        )
        if spec.get("capability") == "directional_interval":
            group_column = spec.get("group_column")
            raw_groups = spec.get("groups")
            if group_column is not None:
                if not isinstance(raw_groups, list):
                    raise NoveltyTerminalError(
                        f"TUSI-168 comparator groups missing: {artifact_name}"
                    )
                groups: Sequence[str | None] = cast(
                    list[str],
                    raw_groups,
                )
            else:
                groups = (None,)
            observed = (
                {row[group_column] for row in rows}
                if group_column is not None
                else {None}
            )
            if observed != set(groups):
                raise NoveltyTerminalError(
                    f"TUSI-168 comparator groups drift: {artifact_name}"
                )
            for group in groups:
                group_rows = (
                    [row for row in rows if row[group_column] == group]
                    if group_column is not None
                    else rows
                )
                comparator_id = (
                    artifact_name if group is None else f"{artifact_name}:{group}"
                )
                clocks[comparator_id] = esdi_novelty._clock_from_rows(
                    comparator_id=comparator_id,
                    artifact_name=artifact_name,
                    group=group,
                    capability="directional_interval",
                    rows=group_rows,
                    spec=spec,
                )
        else:
            group_column = spec["group_column"]
            capability_column = spec["capability_column"]
            capabilities = {
                **{
                    group: "directional_interval"
                    for group in spec["directional_interval_groups"]
                },
                **{
                    group: "timestamp_only"
                    for group in spec["timestamp_only_groups"]
                },
            }
            observed = {row[group_column] for row in rows}
            if observed != set(capabilities):
                raise NoveltyTerminalError(
                    f"TUSI-168 comparator bundle groups drift: {artifact_name}"
                )
            for group, capability in capabilities.items():
                group_rows = [
                    row for row in rows if row[group_column] == group
                ]
                if any(
                    row[capability_column] != capability for row in group_rows
                ):
                    raise NoveltyTerminalError(
                        "TUSI-168 frozen comparator capability drift: "
                        f"{artifact_name}:{group}"
                    )
                comparator_id = f"{artifact_name}:{group}"
                clocks[comparator_id] = esdi_novelty._clock_from_rows(
                    comparator_id=comparator_id,
                    artifact_name=artifact_name,
                    group=group,
                    capability=capability,
                    rows=group_rows,
                    spec=spec,
                )
    return clocks


def load_comparator_artifacts(
    registry: Mapping[str, Mapping[str, Any]],
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
    expected_registry: Mapping[str, Mapping[str, Any]] | None = None,
) -> VerifiedComparatorClocks:
    frozen = (
        prereg.frozen_comparator_registry()
        if expected_registry is None
        else expected_registry
    )
    validate_registry(registry, frozen)
    exact_bytes = {
        artifact_name: _read_comparator_artifact_bytes(
            artifact_name,
            spec,
            repository_root=repository_root,
        )
        for artifact_name, spec in registry.items()
    }
    clocks = _comparator_clocks_from_artifact_bytes(registry, exact_bytes)
    return VerifiedComparatorClocks(
        clocks=MappingProxyType(clocks),
        artifact_bytes=MappingProxyType(exact_bytes),
        artifact_sha256=MappingProxyType(
            {
                artifact_name: sha256_bytes(raw)
                for artifact_name, raw in exact_bytes.items()
            }
        ),
        registry_sha256=canonical_hash(registry),
        authentication_mode="canonical_nofollow_registry",
    )


_parse_timestamp = esdi_novelty._parse_timestamp
_parse_side = esdi_novelty._parse_side
_domain_seconds = esdi_novelty._domain_seconds
_canonical_intervals = esdi_novelty._canonical_intervals
_intervals_in_domain = esdi_novelty._intervals_in_domain
inclusive_fraction_gate = esdi_novelty.inclusive_fraction_gate
strict_fraction_gate = esdi_novelty.strict_fraction_gate
evaluate_prior_comparator = esdi_novelty.evaluate_prior_comparator
validate_gross9_sleeves = esdi_novelty.validate_gross9_sleeves
evaluate_gross9_sleeve = esdi_novelty.evaluate_gross9_sleeve


def parse_gross9_clock_artifact_bytes(
    raw: bytes,
    *,
    path: str | Path,
    registration: Mapping[str, Any],
    source_support: VerifiedSourceSupport,
    production: bool,
    synthetic_authentication: InjectedVerifiedGross9Fixture | None = None,
) -> VerifiedGross9Clocks:
    """Parse only explicitly injected synthetic Gross9 authentication fixtures."""

    if production:
        raise NoveltyTerminalError(
            "TUSI-168 production Gross9 must use the authoritative ESDI authenticator"
        )
    if not isinstance(synthetic_authentication, InjectedVerifiedGross9Fixture):
        raise NoveltyTerminalError(
            "TUSI-168 synthetic Gross9 requires injected verified authentication"
        )
    if not isinstance(source_support, VerifiedSourceSupport):
        raise NoveltyTerminalError(
            "TUSI-168 Gross9 clocks require verified source support"
        )
    verified_support = parse_passed_source_support_bytes(
        source_support.raw_bytes,
        path=source_support.path,
        production=False,
    )
    if verified_support.sha256 != source_support.sha256:
        raise NoveltyTerminalError(
            "TUSI-168 source-support immutable hash drift"
        )
    _validate_registration_authorities(registration)
    artifact_path = Path(path)
    payload = _decode_json_bytes(raw, "Gross9 clock artifact")
    required = {
        "protocol_version",
        "policy_id",
        "preregistration",
        "source_support",
        "authority_hash",
        "clocks",
        "frozen_contract_validation",
        "manifest_hash",
    }
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if (
        set(payload) != required
        or payload.get("manifest_hash") != canonical_hash(core)
        or payload.get("protocol_version") != GROSS9_CLOCKS_PROTOCOL_VERSION
        or payload.get("policy_id") != esdi_novelty.POLICY_ID
        or payload.get("preregistration")
        != {
            "path": str(ESDI_PREREGISTRATION_PATH),
            "sha256": ESDI_PREREGISTRATION_SHA256,
            "manifest_hash": ESDI_PREREGISTRATION_MANIFEST_HASH,
        }
        or payload.get("authority_hash")
        != synthetic_authentication.authority_hash
        or payload.get("frozen_contract_validation")
        != dict(synthetic_authentication.frozen_contract_validation)
        or payload.get("source_support")
        != dict(synthetic_authentication.source_support_binding)
        or synthetic_authentication.authority_hash
        != GROSS9_AUTHORITY_SHA256
    ):
        raise NoveltyTerminalError("TUSI-168 Gross9 clock binding drift")
    raw_clocks = payload.get("clocks")
    if not isinstance(raw_clocks, Mapping) or set(raw_clocks) != set(
        GROSS9_SLEEVES
    ):
        raise NoveltyTerminalError(
            "TUSI-168 Gross9 artifact requires exactly five clocks"
        )
    clocks: dict[str, tuple[SignedInterval, ...]] = {}
    for sleeve in GROSS9_SLEEVES:
        clock = raw_clocks[sleeve]
        if (
            not isinstance(clock, Mapping)
            or set(clock) != {"intervals", "sha256"}
            or not isinstance(clock["intervals"], list)
            or clock["sha256"]
            != canonical_hash({"intervals": clock["intervals"]})
        ):
            raise NoveltyTerminalError(
                f"TUSI-168 Gross9 sleeve schema drift: {sleeve}"
            )
        intervals: list[SignedInterval] = []
        for row in clock["intervals"]:
            if not isinstance(row, Mapping) or set(row) != {
                "entry",
                "exit",
                "side",
            }:
                raise NoveltyTerminalError(
                    f"TUSI-168 Gross9 interval schema drift: {sleeve}"
                )
            entry_value = row["entry"]
            exit_value = row["exit"]
            side_value = row["side"]
            if (
                not isinstance(entry_value, str)
                or not isinstance(exit_value, str)
                or not isinstance(side_value, str)
            ):
                raise NoveltyTerminalError(
                    f"TUSI-168 Gross9 interval type drift: {sleeve}"
                )
            intervals.append(
                SignedInterval(
                    _parse_timestamp(entry_value),
                    _parse_timestamp(exit_value),
                    _parse_side(side_value),
                )
            )
        clocks[sleeve] = _canonical_intervals(intervals, f"Gross9:{sleeve}")
    validate_gross9_sleeves(clocks)
    return VerifiedGross9Clocks(
        path=artifact_path,
        raw_bytes=raw,
        sha256=sha256_bytes(raw),
        manifest_hash=str(payload["manifest_hash"]),
        authority_hash=GROSS9_AUTHORITY_SHA256,
        clocks=MappingProxyType(clocks),
        payload=_freeze_json(payload),
        authentication_mode="injected_synthetic",
    )


def _authenticate_authoritative_esdi_gross9(
    raw_bytes: bytes,
) -> esdi_novelty.VerifiedGross9Clocks:
    """Authenticate the exact Gross9 bytes opened by this evaluator."""

    esdi_registration = esdi_novelty.verify_preregistration()
    esdi_support = esdi_novelty.load_passed_source_support(production=True)
    return esdi_novelty.parse_gross9_clock_artifact_bytes(
        raw_bytes,
        path=DEFAULT_GROSS9_CLOCKS_PATH,
        registration=esdi_registration,
        source_support=esdi_support,
        production=True,
    )


def _authenticate_gross9_exact_bytes(
    raw: bytes,
    *,
    authoritative_authenticator: (
        Callable[[bytes], esdi_novelty.VerifiedGross9Clocks] | None
    ),
    repository_root: str | Path,
) -> VerifiedGross9Clocks:
    if (
        authoritative_authenticator is not None
        and Path(repository_root).resolve() == REPOSITORY_ROOT.resolve()
    ):
        raise NoveltyTerminalError(
            "TUSI-168 injected authoritative authenticator is test-only"
        )
    authenticator = (
        _authenticate_authoritative_esdi_gross9
        if authoritative_authenticator is None
        else authoritative_authenticator
    )
    try:
        authenticated = authenticator(raw)
    except Exception as error:
        raise NoveltyTerminalError(
            "TUSI-168 authoritative ESDI Gross9 authentication failed"
        ) from error
    if (
        not isinstance(authenticated, esdi_novelty.VerifiedGross9Clocks)
        or authenticated.path != DEFAULT_GROSS9_CLOCKS_PATH
        or authenticated.raw_bytes != raw
        or authenticated.sha256 != sha256_bytes(raw)
        or authenticated.authority_hash != GROSS9_AUTHORITY_SHA256
        or set(authenticated.clocks) != set(GROSS9_SLEEVES)
    ):
        raise NoveltyTerminalError(
            "TUSI-168 authoritative ESDI Gross9 completion drift"
        )
    return VerifiedGross9Clocks(
        path=DEFAULT_GROSS9_CLOCKS_PATH,
        raw_bytes=raw,
        sha256=authenticated.sha256,
        manifest_hash=authenticated.manifest_hash,
        authority_hash=authenticated.authority_hash,
        clocks=authenticated.clocks,
        payload=authenticated.payload,
        authentication_mode="authoritative_esdi_production",
    )


def load_gross9_clock_artifact(
    *,
    registration: Mapping[str, Any],
    source_support: VerifiedSourceSupport,
    path: str | Path = DEFAULT_GROSS9_CLOCKS_PATH,
    production: bool = True,
    synthetic_authentication: InjectedVerifiedGross9Fixture | None = None,
    authoritative_authenticator: (
        Callable[[bytes], esdi_novelty.VerifiedGross9Clocks] | None
    ) = None,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> VerifiedGross9Clocks:
    _validate_registration_authorities(registration)
    if production:
        _canonical_relative_path(
            path, DEFAULT_GROSS9_CLOCKS_PATH, "Gross9 clock artifact"
        )
        if synthetic_authentication is not None:
            raise NoveltyTerminalError(
                "TUSI-168 injected Gross9 authentication is synthetic-only"
            )
        if source_support.production_authenticated is not True:
            raise NoveltyTerminalError(
                "TUSI-168 Gross9 authentication requires production source support"
            )
        raw = _read_canonical_regular(
            path,
            DEFAULT_GROSS9_CLOCKS_PATH,
            "Gross9 clock artifact",
            repository_root=repository_root,
        )
        return _authenticate_gross9_exact_bytes(
            raw,
            authoritative_authenticator=authoritative_authenticator,
            repository_root=repository_root,
        )
    raw = _read_regular_path_nofollow(
        path,
        "synthetic Gross9 fixture",
        repository_root=repository_root,
    )
    return parse_gross9_clock_artifact_bytes(
        raw,
        path=path,
        registration=registration,
        source_support=source_support,
        production=False,
        synthetic_authentication=synthetic_authentication,
    )


def evaluate_novelty(
    candidate: Sequence[SignedInterval | tuple[int, int, int]],
    comparators: Mapping[str, ComparatorClock],
    registry: Mapping[str, Mapping[str, Any]],
    gross9_artifact: VerifiedGross9Clocks,
) -> dict[str, Any]:
    if gross9_artifact.authentication_mode not in {
        "authoritative_esdi_production",
        "injected_synthetic",
    }:
        raise NoveltyTerminalError(
            "TUSI-168 Gross9 authentication mode is invalid"
        )
    authoritative_shape = esdi_novelty.VerifiedGross9Clocks(
        path=gross9_artifact.path,
        raw_bytes=gross9_artifact.raw_bytes,
        sha256=gross9_artifact.sha256,
        manifest_hash=gross9_artifact.manifest_hash,
        authority_hash=gross9_artifact.authority_hash,
        clocks=gross9_artifact.clocks,
        payload=gross9_artifact.payload,
    )
    return esdi_novelty.evaluate_novelty(
        candidate, comparators, registry, authoritative_shape
    )


def _parse_candidate_clock_bytes(
    compressed: bytes,
    *,
    path: str | Path,
    source_support: VerifiedSourceSupport,
    authentication_mode: str,
) -> VerifiedCandidateClock:
    if not isinstance(source_support, VerifiedSourceSupport):
        raise NoveltyTerminalError(
            "TUSI-168 primary clock requires verified source support"
        )
    verified = parse_passed_source_support_bytes(
        source_support.raw_bytes,
        path=source_support.path,
        production=False,
    )
    expected_hash = verified.payload["clock_artifacts"]["primary_sha256"]
    if sha256_bytes(compressed) != expected_hash:
        raise NoveltyTerminalError("TUSI-168 primary clock hash drift")
    try:
        rows, counts = source_support_evaluator._validate_clock_csv(
            compressed, primary=True
        )
    except RuntimeError as error:
        raise NoveltyTerminalError(
            "TUSI-168 primary clock contract drift"
        ) from error
    if counts["primary"] != verified.payload["accepted_clock_counts"]["primary"]:
        raise NoveltyTerminalError(
            "TUSI-168 primary clock/report count drift"
        )
    intervals = tuple(
        SignedInterval(
            _parse_timestamp(row["entry_time_utc"]),
            _parse_timestamp(row["exit_time_utc"]),
            _parse_side(row["side"]),
        )
        for row in rows
    )
    canonical = _canonical_intervals(intervals, "TUSI-168 primary clock")
    return VerifiedCandidateClock(
        path=Path(path),
        raw_bytes=compressed,
        sha256=expected_hash,
        source_support_sha256=source_support.sha256,
        intervals=canonical,
        authentication_mode=authentication_mode,
    )


def load_candidate_clock_csv(
    path: str | Path,
    source_support: VerifiedSourceSupport,
    *,
    production: bool = False,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> VerifiedCandidateClock:
    try:
        compressed = (
            _read_canonical_regular(
                path,
                DEFAULT_PRIMARY_CLOCK_PATH,
                "primary clock artifact",
                repository_root=repository_root,
            )
            if production
            else Path(path).read_bytes()
        )
    except OSError as error:
        raise NoveltyTerminalError("TUSI-168 primary clock is missing") from error
    if production:
        _require_canonical_committed_clean(
            path,
            DEFAULT_PRIMARY_CLOCK_PATH,
            "primary clock artifact",
            compressed,
            repository_root=repository_root,
        )
    return _parse_candidate_clock_bytes(
        compressed,
        path=DEFAULT_PRIMARY_CLOCK_PATH if production else path,
        source_support=source_support,
        authentication_mode=(
            "canonical_nofollow_committed"
            if production
            else "injected_synthetic"
        ),
    )


def _build_report_from_clocks(
    *,
    source_support: VerifiedSourceSupport,
    candidate: Sequence[SignedInterval | tuple[int, int, int]],
    gross9_artifact: VerifiedGross9Clocks,
    comparators: Mapping[str, ComparatorClock],
    registration: Mapping[str, Any],
    attempt_claim: Mapping[str, Any],
) -> dict[str, Any]:
    verified_support = parse_passed_source_support_bytes(
        source_support.raw_bytes,
        path=source_support.path,
        production=False,
    )
    if (
        verified_support.sha256 != source_support.sha256
        or verified_support.manifest_hash != source_support.manifest_hash
    ):
        raise NoveltyTerminalError(
            "TUSI-168 source-support immutable binding drift"
        )
    _validate_registration_authorities(registration)
    if (
        not isinstance(gross9_artifact, VerifiedGross9Clocks)
        or gross9_artifact.authentication_mode
        not in {"authoritative_esdi_production", "injected_synthetic"}
        or sha256_bytes(gross9_artifact.raw_bytes) != gross9_artifact.sha256
        or gross9_artifact.authority_hash != GROSS9_AUTHORITY_SHA256
        or set(gross9_artifact.clocks) != set(GROSS9_SLEEVES)
    ):
        raise NoveltyTerminalError("TUSI-168 Gross9 immutable binding drift")
    registry = frozen_registry(registration)
    expected_comparators = _expected_comparator_contracts(registry)
    novelty = evaluate_novelty(
        candidate, comparators, registry, gross9_artifact
    )
    passed = novelty["passed"]
    core: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "status": "novelty_passed" if passed else "retired_after_novelty",
        "terminal": not passed,
        "decision": (
            "NOVELTY_PASS_OPEN_STRICT_ECONOMICS"
            if passed
            else "RETIRE_TUSI_168_UNCHANGED_AFTER_NOVELTY"
        ),
        "preregistration": _expected_preregistration_binding(),
        "attempt_claim": dict(attempt_claim),
        "source_support": {
            "path": DEFAULT_SOURCE_SUPPORT_PATH.as_posix(),
            "sha256": verified_support.sha256,
            "manifest_hash": verified_support.manifest_hash,
            "passed": True,
        },
        "candidate_clock": {
            "path": DEFAULT_PRIMARY_CLOCK_PATH.as_posix(),
            "sha256": verified_support.payload["clock_artifacts"][
                "primary_sha256"
            ],
            "accepted_intervals": len(candidate),
        },
        "gross9_clock_artifact": {
            "path": DEFAULT_GROSS9_CLOCKS_PATH.as_posix(),
            "sha256": gross9_artifact.sha256,
            "manifest_hash": gross9_artifact.manifest_hash,
            "authority_hash": gross9_artifact.authority_hash,
            "authentication_mode": gross9_artifact.authentication_mode,
        },
        "registry": {
            "artifacts": len(registry),
            "comparator_groups": len(comparators),
            "comparator_ids": sorted(expected_comparators),
            "canonical_compact_sorted_sha256": COMPARATOR_REGISTRY_SHA256,
        },
        "novelty": novelty,
        "evidence_boundary": dict(NOVELTY_EVIDENCE_BOUNDARY),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def build_report_after_source_support(
    *,
    source_support: VerifiedSourceSupport,
    candidate: Sequence[SignedInterval | tuple[int, int, int]],
    gross9_artifact: VerifiedGross9Clocks,
    comparator_loader: ComparatorLoader = load_comparator_artifacts,
    registration: Mapping[str, Any] | None = None,
    attempt_claim: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a report; publication authenticity requires retained inputs."""

    active_registration = (
        verify_preregistration() if registration is None else registration
    )
    verified_support = parse_passed_source_support_bytes(
        source_support.raw_bytes,
        path=source_support.path,
        production=False,
    )
    if (
        verified_support.sha256 != source_support.sha256
        or verified_support.manifest_hash != source_support.manifest_hash
    ):
        raise NoveltyTerminalError(
            "TUSI-168 source-support immutable binding drift"
        )
    registry = frozen_registry(active_registration)
    comparators = comparator_loader(registry)
    return _build_report_from_clocks(
        source_support=source_support,
        candidate=candidate,
        gross9_artifact=gross9_artifact,
        comparators=comparators,
        registration=active_registration,
        attempt_claim=(
            attempt_claim
            if attempt_claim is not None
            else {"mode": "synthetic_only"}
        ),
    )


def _exact_fraction(value: Any, label: str) -> Fraction:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"numerator", "denominator"}
        or type(value.get("numerator")) is not int
        or type(value.get("denominator")) is not int
    ):
        raise NoveltyTerminalError(f"TUSI-168 invalid report fraction: {label}")
    numerator = cast(int, value["numerator"])
    denominator = cast(int, value["denominator"])
    if numerator < 0 or denominator <= 0 or numerator > denominator:
        raise NoveltyTerminalError(f"TUSI-168 invalid report fraction: {label}")
    fraction = Fraction(numerator, denominator)
    if (
        fraction.numerator != numerator
        or fraction.denominator != denominator
    ):
        raise NoveltyTerminalError(
            f"TUSI-168 noncanonical report fraction: {label}"
        )
    return fraction


def _validate_prior_result(
    item: Any,
    comparator_id: str,
    contract: Mapping[str, Any],
) -> bool:
    required = {
        "comparator_id",
        "artifact_name",
        "group",
        "capability",
        "comparison_domain",
        "candidate_entries",
        "comparator_entries",
        "minimum_count_after_common_domain_filter",
        "gating",
        "metrics",
        "checks",
        "passed",
        "would_pass_if_gating",
    }
    if not isinstance(item, Mapping) or set(item) != required:
        raise NoveltyTerminalError(
            f"TUSI-168 prior comparator report schema drift: {comparator_id}"
        )
    candidate_entries = item.get("candidate_entries")
    comparator_entries = item.get("comparator_entries")
    capability = contract["capability"]
    if (
        item.get("comparator_id") != comparator_id
        or item.get("artifact_name") != contract["artifact_name"]
        or item.get("group") != contract["group"]
        or item.get("capability") != capability
        or item.get("comparison_domain") != contract["comparison_domain"]
        or type(candidate_entries) is not int
        or candidate_entries < 0
        or type(comparator_entries) is not int
        or comparator_entries < 0
        or item.get("minimum_count_after_common_domain_filter") is not True
        or item.get("gating")
        is not (comparator_entries >= MINIMUM_GATING_ENTRIES)
    ):
        raise NoveltyTerminalError(
            f"TUSI-168 prior comparator identity/count drift: {comparator_id}"
        )
    metrics = item.get("metrics")
    checks = item.get("checks")
    expected_metric_keys = {
        "exact_entry_jaccard",
        "candidate_24h_containment",
        "squared_signed_exposure_pearson",
    }
    expected_check_keys = {
        "exact_entry_jaccard",
        "candidate_24h_containment",
    }
    if (
        not isinstance(metrics, Mapping)
        or set(metrics) != expected_metric_keys
        or not isinstance(checks, Mapping)
        or any(type(value) is not bool for value in checks.values())
    ):
        raise NoveltyTerminalError(
            f"TUSI-168 prior comparator metric schema drift: {comparator_id}"
        )
    jaccard = _exact_fraction(
        metrics["exact_entry_jaccard"], f"{comparator_id} entry Jaccard"
    )
    containment = _exact_fraction(
        metrics["candidate_24h_containment"],
        f"{comparator_id} containment",
    )
    expected_checks = {
        "exact_entry_jaccard": jaccard <= Fraction(1, 5),
        "candidate_24h_containment": containment <= Fraction(1, 2),
    }
    pearson_payload = metrics["squared_signed_exposure_pearson"]
    if capability == "timestamp_only":
        if pearson_payload != {
            "applicable": False,
            "reason": "frozen_timestamp_only_capability",
        }:
            raise NoveltyTerminalError(
                f"TUSI-168 timestamp-only Pearson drift: {comparator_id}"
            )
    else:
        expected_check_keys.add("squared_signed_exposure_pearson")
        if (
            not isinstance(pearson_payload, Mapping)
            or set(pearson_payload)
            != {"applicable", "squared_exact", "absolute_correlation"}
            or pearson_payload.get("applicable") is not True
        ):
            raise NoveltyTerminalError(
                f"TUSI-168 directional Pearson schema drift: {comparator_id}"
            )
        pearson = _exact_fraction(
            pearson_payload["squared_exact"], f"{comparator_id} Pearson"
        )
        absolute = pearson_payload["absolute_correlation"]
        if (
            type(absolute) is not float
            or not math.isfinite(absolute)
            or absolute != math.sqrt(pearson.numerator / pearson.denominator)
        ):
            raise NoveltyTerminalError(
                f"TUSI-168 Pearson display drift: {comparator_id}"
            )
        expected_checks["squared_signed_exposure_pearson"] = (
            pearson <= Fraction(4, 25)
        )
    would_pass = all(expected_checks.values())
    expected_pass = (
        would_pass
        if comparator_entries >= MINIMUM_GATING_ENTRIES
        else True
    )
    if (
        set(checks) != expected_check_keys
        or dict(checks) != expected_checks
        or item.get("would_pass_if_gating") is not would_pass
        or item.get("passed") is not expected_pass
    ):
        raise NoveltyTerminalError(
            f"TUSI-168 prior comparator gate drift: {comparator_id}"
        )
    return expected_pass


def _validate_gross9_result(item: Any, sleeve: str) -> bool:
    required = {
        "sleeve",
        "weight",
        "comparison_domain",
        "candidate_entries",
        "sleeve_entries",
        "metrics",
        "checks",
        "passed",
    }
    if not isinstance(item, Mapping) or set(item) != required:
        raise NoveltyTerminalError(
            f"TUSI-168 Gross9 report schema drift: {sleeve}"
        )
    candidate_entries = item.get("candidate_entries")
    sleeve_entries = item.get("sleeve_entries")
    metrics = item.get("metrics")
    checks = item.get("checks")
    if (
        item.get("sleeve") != sleeve
        or item.get("weight") != prereg.esdi.GROSS9_WEIGHTS[sleeve]
        or item.get("comparison_domain") != list(GROSS9_DOMAIN)
        or type(candidate_entries) is not int
        or candidate_entries < 0
        or type(sleeve_entries) is not int
        or sleeve_entries <= 0
        or not isinstance(metrics, Mapping)
        or set(metrics)
        != {
            "exact_entry_jaccard",
            "candidate_6h_containment",
            "occupied_bar_jaccard",
            "squared_signed_exposure_pearson",
        }
        or not isinstance(checks, Mapping)
        or any(type(value) is not bool for value in checks.values())
    ):
        raise NoveltyTerminalError(
            f"TUSI-168 Gross9 identity/metric drift: {sleeve}"
        )
    jaccard = _exact_fraction(
        metrics["exact_entry_jaccard"], f"{sleeve} entry Jaccard"
    )
    containment = _exact_fraction(
        metrics["candidate_6h_containment"], f"{sleeve} containment"
    )
    occupied = _exact_fraction(
        metrics["occupied_bar_jaccard"], f"{sleeve} occupied Jaccard"
    )
    pearson_payload = metrics["squared_signed_exposure_pearson"]
    if (
        not isinstance(pearson_payload, Mapping)
        or set(pearson_payload)
        != {"squared_exact", "absolute_correlation"}
    ):
        raise NoveltyTerminalError(
            f"TUSI-168 Gross9 Pearson schema drift: {sleeve}"
        )
    pearson = _exact_fraction(
        pearson_payload["squared_exact"], f"{sleeve} Pearson"
    )
    absolute = pearson_payload["absolute_correlation"]
    if (
        type(absolute) is not float
        or not math.isfinite(absolute)
        or absolute != math.sqrt(pearson.numerator / pearson.denominator)
    ):
        raise NoveltyTerminalError(
            f"TUSI-168 Gross9 Pearson display drift: {sleeve}"
        )
    expected_checks = {
        "exact_entry_jaccard": jaccard <= Fraction(1, 10),
        "candidate_6h_containment": containment <= Fraction(7, 20),
        "occupied_bar_jaccard": occupied <= Fraction(1, 4),
        "squared_signed_exposure_pearson": pearson <= Fraction(49, 400),
    }
    passed = all(expected_checks.values())
    if dict(checks) != expected_checks or item.get("passed") is not passed:
        raise NoveltyTerminalError(f"TUSI-168 Gross9 gate drift: {sleeve}")
    return passed


def _validate_novelty_results(
    novelty_payload: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, Any]],
) -> bool:
    contracts = _expected_comparator_contracts(registry)
    prior = novelty_payload.get("prior_source_comparators")
    gross9 = novelty_payload.get("gross9_sleeves")
    if (
        not isinstance(prior, list)
        or len(prior) != len(contracts)
        or [
            item.get("comparator_id") if isinstance(item, Mapping) else None
            for item in prior
        ]
        != sorted(contracts)
        or not isinstance(gross9, list)
        or [
            item.get("sleeve") if isinstance(item, Mapping) else None
            for item in gross9
        ]
        != list(GROSS9_SLEEVES)
    ):
        raise NoveltyTerminalError(
            "TUSI-168 exhaustive comparator result inventory drift"
        )
    prior_passes: list[bool] = []
    failed_checks: list[str] = []
    for item in prior:
        comparator_id = cast(str, item["comparator_id"])
        passed = _validate_prior_result(
            item, comparator_id, contracts[comparator_id]
        )
        prior_passes.append(passed)
        if item["gating"]:
            failed_checks.extend(
                f"prior:{comparator_id}:{name}"
                for name, check in item["checks"].items()
                if check is False
            )
    gross_passes: list[bool] = []
    for item, sleeve in zip(gross9, GROSS9_SLEEVES, strict=True):
        passed = _validate_gross9_result(item, sleeve)
        gross_passes.append(passed)
        failed_checks.extend(
            f"gross9:{sleeve}:{name}"
            for name, check in item["checks"].items()
            if check is False
        )
    passed = all((*prior_passes, *gross_passes))
    if (
        novelty_payload.get("passed") is not passed
        or novelty_payload.get("terminal") is not (not passed)
        or novelty_payload.get("failed_checks") != sorted(failed_checks)
    ):
        raise NoveltyTerminalError(
            "TUSI-168 aggregate novelty result drift"
        )
    return passed


def canonical_report_bytes(
    payload: Mapping[str, Any],
    *,
    registration: Mapping[str, Any] | None = None,
) -> bytes:
    """Serialize an exact self-consistent schema, without artifact authenticity."""

    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    novelty = payload.get("novelty")
    active_registry = (
        prereg.frozen_comparator_registry()
        if registration is None
        else frozen_registry(registration)
    )
    expected_comparators = _expected_comparator_contracts(active_registry)
    passed = (
        _validate_novelty_results(novelty, active_registry)
        if isinstance(novelty, Mapping)
        else None
    )
    expected_terminal = passed is False
    expected_status = (
        "novelty_passed" if passed is True else "retired_after_novelty"
    )
    expected_decision = (
        "NOVELTY_PASS_OPEN_STRICT_ECONOMICS"
        if passed is True
        else "RETIRE_TUSI_168_UNCHANGED_AFTER_NOVELTY"
    )
    source_binding = payload.get("source_support")
    gross9_binding = payload.get("gross9_clock_artifact")
    registry = payload.get("registry")
    candidate = payload.get("candidate_clock")
    comparator_groups = (
        registry.get("comparator_groups")
        if isinstance(registry, Mapping)
        else None
    )
    accepted_intervals = (
        candidate.get("accepted_intervals")
        if isinstance(candidate, Mapping)
        else None
    )
    if (
        set(payload) != REPORT_KEYS
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("policy_id") != POLICY_ID
        or type(passed) is not bool
        or payload.get("terminal") is not expected_terminal
        or payload.get("status") != expected_status
        or payload.get("decision") != expected_decision
        or payload.get("preregistration")
        != _expected_preregistration_binding()
        or not isinstance(novelty, Mapping)
        or set(novelty)
        != {
            "prior_source_comparators",
            "gross9_sleeves",
            "passed",
            "terminal",
            "failed_checks",
        }
        or novelty.get("terminal") is not expected_terminal
        or not isinstance(source_binding, Mapping)
        or set(source_binding) != {"path", "sha256", "manifest_hash", "passed"}
        or source_binding.get("path") != DEFAULT_SOURCE_SUPPORT_PATH.as_posix()
        or source_binding.get("passed") is not True
        or not isinstance(gross9_binding, Mapping)
        or set(gross9_binding)
        != {
            "path",
            "sha256",
            "manifest_hash",
            "authority_hash",
            "authentication_mode",
        }
        or gross9_binding.get("path") != DEFAULT_GROSS9_CLOCKS_PATH.as_posix()
        or gross9_binding.get("authority_hash") != GROSS9_AUTHORITY_SHA256
        or gross9_binding.get("authentication_mode")
        not in {"authoritative_esdi_production", "injected_synthetic"}
        or not isinstance(registry, Mapping)
        or dict(registry)
        != {
            "artifacts": 18,
            "comparator_groups": len(expected_comparators),
            "comparator_ids": sorted(expected_comparators),
            "canonical_compact_sorted_sha256": COMPARATOR_REGISTRY_SHA256,
        }
        or type(comparator_groups) is not int
        or comparator_groups <= 0
        or not isinstance(candidate, Mapping)
        or set(candidate) != {"path", "sha256", "accepted_intervals"}
        or candidate.get("path") != DEFAULT_PRIMARY_CLOCK_PATH.as_posix()
        or type(accepted_intervals) is not int
        or accepted_intervals <= 0
        or payload.get("evidence_boundary") != NOVELTY_EVIDENCE_BOUNDARY
        or payload.get("manifest_hash") != canonical_hash(core)
    ):
        raise NoveltyTerminalError(
            "TUSI-168 novelty report exact schema or manifest drift"
        )
    attempt = payload.get("attempt_claim")
    if not (
        attempt == {"mode": "synthetic_only"}
        or isinstance(attempt, Mapping)
        and set(attempt) == {"path", "sha256", "claim_hash"}
        and attempt.get("path") == DEFAULT_ATTEMPT_CLAIM_PATH.as_posix()
    ):
        raise NoveltyTerminalError(
            "TUSI-168 novelty attempt-claim report binding drift"
        )
    if isinstance(attempt, Mapping) and "sha256" in attempt:
        esdi_novelty._validate_hash(
            attempt.get("sha256"), "novelty attempt claim file hash"
        )
        esdi_novelty._validate_hash(
            attempt.get("claim_hash"), "novelty attempt claim hash"
        )
    for binding, label in (
        (source_binding, "source-support"),
        (gross9_binding, "Gross9"),
        (candidate, "candidate"),
    ):
        for key in ("sha256", "manifest_hash"):
            if key in binding:
                esdi_novelty._validate_hash(
                    binding[key], f"novelty report {label} {key}"
                )
    try:
        return (
            json.dumps(
                payload,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise NoveltyTerminalError(
            "TUSI-168 novelty report is not canonical JSON"
        ) from error


def _revalidated_candidate_intervals(
    candidate: VerifiedCandidateClock,
    source_support: VerifiedSourceSupport,
    *,
    production: bool,
) -> tuple[SignedInterval, ...]:
    expected_mode = (
        "canonical_nofollow_committed" if production else candidate.authentication_mode
    )
    if (
        not isinstance(candidate, VerifiedCandidateClock)
        or candidate.authentication_mode != expected_mode
        or candidate.source_support_sha256 != source_support.sha256
        or candidate.sha256 != sha256_bytes(candidate.raw_bytes)
        or production
        and candidate.path != DEFAULT_PRIMARY_CLOCK_PATH
    ):
        raise NoveltyTerminalError(
            "TUSI-168 authenticated candidate clock binding drift"
        )
    reparsed = _parse_candidate_clock_bytes(
        candidate.raw_bytes,
        path=candidate.path,
        source_support=source_support,
        authentication_mode=candidate.authentication_mode,
    )
    if (
        reparsed.sha256 != candidate.sha256
        or reparsed.source_support_sha256 != candidate.source_support_sha256
        or reparsed.intervals != candidate.intervals
    ):
        raise NoveltyTerminalError(
            "TUSI-168 authenticated candidate clock reproduction drift"
        )
    return reparsed.intervals


def _revalidated_comparator_clocks(
    comparators: VerifiedComparatorClocks,
    registry: Mapping[str, Mapping[str, Any]],
    *,
    production: bool,
) -> Mapping[str, ComparatorClock]:
    if not isinstance(comparators, VerifiedComparatorClocks):
        raise NoveltyTerminalError(
            "TUSI-168 authenticated comparator context is missing"
        )
    if comparators.registry_sha256 != canonical_hash(registry):
        raise NoveltyTerminalError(
            "TUSI-168 authenticated comparator registry drift"
        )
    if comparators.authentication_mode == "injected_synthetic":
        if production:
            raise NoveltyTerminalError(
                "TUSI-168 production requires no-follow comparator bytes"
            )
        expected_ids = set(_expected_comparator_contracts(registry))
        if set(comparators.clocks) != expected_ids:
            raise NoveltyTerminalError(
                "TUSI-168 synthetic comparator inventory drift"
            )
        return comparators.clocks
    if comparators.authentication_mode != "canonical_nofollow_registry":
        raise NoveltyTerminalError(
            "TUSI-168 comparator authentication mode drift"
        )
    expected_hashes = {
        artifact_name: cast(str, spec["sha256"])
        for artifact_name, spec in registry.items()
    }
    observed_hashes = {
        artifact_name: sha256_bytes(raw)
        for artifact_name, raw in comparators.artifact_bytes.items()
    }
    if (
        dict(comparators.artifact_sha256) != expected_hashes
        or observed_hashes != expected_hashes
    ):
        raise NoveltyTerminalError(
            "TUSI-168 authenticated comparator byte hash drift"
        )
    reparsed = _comparator_clocks_from_artifact_bytes(
        registry,
        comparators.artifact_bytes,
    )
    if reparsed != dict(comparators.clocks):
        raise NoveltyTerminalError(
            "TUSI-168 authenticated comparator clock reproduction drift"
        )
    return MappingProxyType(reparsed)


def reproduce_report_from_authenticated_inputs(
    inputs: AuthenticatedNoveltyInputs,
    *,
    production: bool,
    repository_root: str | Path = REPOSITORY_ROOT,
    authoritative_gross9_authenticator: (
        Callable[[bytes], esdi_novelty.VerifiedGross9Clocks] | None
    ) = None,
) -> dict[str, Any]:
    """Recompute every numeric result from retained authenticated clock bytes."""

    if not isinstance(inputs, AuthenticatedNoveltyInputs):
        raise NoveltyTerminalError(
            "TUSI-168 authenticated novelty inputs are missing"
        )
    if inputs.production is not production:
        raise NoveltyTerminalError(
            "TUSI-168 novelty reproduction mode drift"
        )
    source = inputs.source_support
    if production:
        if source.production_authenticated is not True:
            raise NoveltyTerminalError(
                "TUSI-168 production source support is not authenticated"
            )
        reauthenticated_source = parse_passed_source_support_bytes(
            source.raw_bytes,
            path=DEFAULT_SOURCE_SUPPORT_PATH.as_posix(),
            production=True,
            repository_root=repository_root,
            protocol_paths=inputs.protocol_paths,
        )
        if (
            reauthenticated_source.sha256 != source.sha256
            or reauthenticated_source.manifest_hash != source.manifest_hash
        ):
            raise NoveltyTerminalError(
                "TUSI-168 source-support publication authentication drift"
            )
        source = reauthenticated_source
    candidate = _revalidated_candidate_intervals(
        inputs.candidate,
        source,
        production=production,
    )
    registry = frozen_registry(inputs.registration)
    comparators = _revalidated_comparator_clocks(
        inputs.comparators,
        registry,
        production=production,
    )
    gross9 = inputs.gross9_artifact
    if production:
        if gross9.authentication_mode != "authoritative_esdi_production":
            raise NoveltyTerminalError(
                "TUSI-168 production report requires authoritative ESDI Gross9"
            )
        reauthenticated_gross9 = _authenticate_gross9_exact_bytes(
            gross9.raw_bytes,
            authoritative_authenticator=authoritative_gross9_authenticator,
            repository_root=repository_root,
        )
        if (
            reauthenticated_gross9.sha256 != gross9.sha256
            or reauthenticated_gross9.manifest_hash != gross9.manifest_hash
            or reauthenticated_gross9.authority_hash != gross9.authority_hash
            or reauthenticated_gross9.clocks != gross9.clocks
            or reauthenticated_gross9.payload != gross9.payload
        ):
            raise NoveltyTerminalError(
                "TUSI-168 authenticated Gross9 reproduction drift"
            )
        gross9 = reauthenticated_gross9
    return _build_report_from_clocks(
        source_support=source,
        candidate=candidate,
        gross9_artifact=gross9,
        comparators=comparators,
        registration=inputs.registration,
        attempt_claim=inputs.attempt_claim,
    )


def require_exact_authenticated_reproduction(
    payload: Mapping[str, Any],
    inputs: AuthenticatedNoveltyInputs,
    *,
    production: bool,
    repository_root: str | Path = REPOSITORY_ROOT,
    authoritative_gross9_authenticator: (
        Callable[[bytes], esdi_novelty.VerifiedGross9Clocks] | None
    ) = None,
) -> None:
    reproduced = reproduce_report_from_authenticated_inputs(
        inputs,
        production=production,
        repository_root=repository_root,
        authoritative_gross9_authenticator=authoritative_gross9_authenticator,
    )
    if _thaw_json(payload) != reproduced:
        raise NoveltyTerminalError(
            "TUSI-168 authenticated full-report reproduction drift"
        )


def validate_report_payload(
    payload: Mapping[str, Any],
    *,
    registration: Mapping[str, Any] | None = None,
    production: bool,
    repository_root: str | Path = REPOSITORY_ROOT,
    authenticated_inputs: AuthenticatedNoveltyInputs | None = None,
    authoritative_gross9_authenticator: (
        Callable[[bytes], esdi_novelty.VerifiedGross9Clocks] | None
    ) = None,
) -> bytes:
    raw = canonical_report_bytes(payload, registration=registration)
    if production:
        if authenticated_inputs is None:
            raise NoveltyTerminalError(
                "TUSI-168 production publication requires authenticated inputs"
            )
        require_exact_authenticated_reproduction(
            payload,
            authenticated_inputs,
            production=True,
            repository_root=repository_root,
            authoritative_gross9_authenticator=(
                authoritative_gross9_authenticator
            ),
        )
        authenticate_attempt_claim_for_report(
            payload,
            repository_root=repository_root,
        )
    return raw


def _write_once_json(
    payload: Mapping[str, Any],
    path: Path,
    *,
    registration: Mapping[str, Any] | None = None,
) -> str:
    canonical = canonical_report_bytes(payload, registration=registration)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir() or path.is_symlink():
        raise NoveltyTerminalError("TUSI-168 novelty output path is unsafe")
    if path.exists():
        if path.read_bytes() != canonical:
            raise NoveltyTerminalError("TUSI-168 novelty output drift")
        return "verified_existing"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".staged", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o444)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != canonical:
                raise NoveltyTerminalError(
                    "TUSI-168 novelty publication race drift"
                ) from None
            return "verified_existing"
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return "created"
    finally:
        temporary.unlink(missing_ok=True)


def write_once_novelty_json(
    payload: Mapping[str, Any],
    output: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    authenticated_inputs: AuthenticatedNoveltyInputs,
    registration: Mapping[str, Any] | None = None,
    repository_root: str | Path = REPOSITORY_ROOT,
    authoritative_gross9_authenticator: (
        Callable[[bytes], esdi_novelty.VerifiedGross9Clocks] | None
    ) = None,
) -> str:
    _canonical_relative_path(output, DEFAULT_OUTPUT_PATH, "novelty output")
    validate_report_payload(
        payload,
        registration=registration,
        production=True,
        repository_root=repository_root,
        authenticated_inputs=authenticated_inputs,
        authoritative_gross9_authenticator=authoritative_gross9_authenticator,
    )
    return _write_once_json(
        payload,
        Path(repository_root) / DEFAULT_OUTPUT_PATH,
        registration=registration,
    )


def write_once_novelty_json_for_test(
    payload: Mapping[str, Any],
    output: str | Path,
    *,
    registration: Mapping[str, Any] | None = None,
) -> str:
    return _write_once_json(
        payload,
        Path(output),
        registration=registration,
    )


def _claim_payload(
    source_support: VerifiedSourceSupport,
    candidate_sha256: str,
) -> dict[str, Any]:
    core = {
        "protocol_version": ATTEMPT_CLAIM_PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "status": "claimed_before_gross9_and_comparator_access",
        "one_shot": True,
        "retry_or_repair_after_failure": False,
        "preregistration": _expected_preregistration_binding(),
        "source_support": {
            "path": DEFAULT_SOURCE_SUPPORT_PATH.as_posix(),
            "sha256": source_support.sha256,
            "manifest_hash": source_support.manifest_hash,
        },
        "candidate_clock": {
            "path": DEFAULT_PRIMARY_CLOCK_PATH.as_posix(),
            "sha256": candidate_sha256,
        },
        "expected_gross9": {
            "path": DEFAULT_GROSS9_CLOCKS_PATH.as_posix(),
            "protocol_version": GROSS9_CLOCKS_PROTOCOL_VERSION,
            "policy_id": esdi_novelty.POLICY_ID,
            "preregistration": {
                "path": ESDI_PREREGISTRATION_PATH.as_posix(),
                "sha256": ESDI_PREREGISTRATION_SHA256,
                "manifest_hash": ESDI_PREREGISTRATION_MANIFEST_HASH,
            },
            "authority_hash": GROSS9_AUTHORITY_SHA256,
        },
        "canonical_output": DEFAULT_OUTPUT_PATH.as_posix(),
    }
    return {**core, "claim_hash": canonical_hash(core)}


def _claim_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _open_parent_directory(
    relative: Path,
    *,
    repository_root: str | Path,
    label: str,
) -> int:
    parent = relative.parent
    root = Path(repository_root)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(root, flags)
    try:
        for part in parent.parts:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except OSError as error:
        os.close(descriptor)
        raise NoveltyTerminalError(
            f"TUSI-168 {label} parent is missing or unsafe"
        ) from error


def _create_attempt_claim(
    payload: Mapping[str, Any],
    *,
    path: str | Path = DEFAULT_ATTEMPT_CLAIM_PATH,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, str]:
    relative = _canonical_relative_path(
        path, DEFAULT_ATTEMPT_CLAIM_PATH, "novelty attempt claim"
    )
    raw = _claim_bytes(payload)
    directory_descriptor = _open_parent_directory(
        relative,
        repository_root=repository_root,
        label="novelty attempt claim",
    )
    temporary = (
        f".{relative.name}.{secrets.token_hex(16)}.staged"
    )
    descriptor = -1
    try:
        try:
            os.stat(
                relative.name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise NoveltyTerminalError(
                "TUSI-168 novelty attempt is already claimed"
            )
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_descriptor,
        )
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("TUSI-168 attempt-claim write stalled")
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(
            temporary,
            relative.name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        os.fsync(directory_descriptor)
    except FileExistsError:
        raise NoveltyTerminalError(
            "TUSI-168 novelty attempt is already claimed"
        ) from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory_descriptor)
        except FileNotFoundError:
            pass
        os.close(directory_descriptor)
    return {
        "path": DEFAULT_ATTEMPT_CLAIM_PATH.as_posix(),
        "sha256": sha256_bytes(raw),
        "claim_hash": str(payload["claim_hash"]),
    }


def _load_attempt_claim(
    expected: Mapping[str, Any],
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, str]:
    raw = _read_canonical_regular(
        DEFAULT_ATTEMPT_CLAIM_PATH.as_posix(),
        DEFAULT_ATTEMPT_CLAIM_PATH,
        "novelty attempt claim",
        repository_root=repository_root,
    )
    payload = _decode_json_bytes(raw, "novelty attempt claim")
    if payload != dict(expected) or raw != _claim_bytes(payload):
        raise NoveltyTerminalError("TUSI-168 novelty attempt claim drift")
    return {
        "path": DEFAULT_ATTEMPT_CLAIM_PATH.as_posix(),
        "sha256": sha256_bytes(raw),
        "claim_hash": str(payload["claim_hash"]),
    }


def authenticate_attempt_claim_for_report(
    report: Mapping[str, Any],
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> None:
    source = report.get("source_support")
    candidate = report.get("candidate_clock")
    attempt = report.get("attempt_claim")
    if (
        not isinstance(source, Mapping)
        or not isinstance(candidate, Mapping)
        or not isinstance(attempt, Mapping)
        or attempt.get("mode") == "synthetic_only"
    ):
        raise NoveltyTerminalError(
            "TUSI-168 production report requires an authenticated attempt claim"
        )
    support = VerifiedSourceSupport(
        path=DEFAULT_SOURCE_SUPPORT_PATH,
        raw_bytes=b"",
        sha256=cast(str, source["sha256"]),
        manifest_hash=cast(str, source["manifest_hash"]),
        payload={},
    )
    expected = _claim_payload(support, cast(str, candidate["sha256"]))
    observed = _load_attempt_claim(expected, repository_root=repository_root)
    if dict(attempt) != observed:
        raise NoveltyTerminalError(
            "TUSI-168 novelty report attempt-claim binding drift"
        )


def collect_claimed_authenticated_inputs(
    *,
    registration: Mapping[str, Any],
    source_support: VerifiedSourceSupport,
    candidate: VerifiedCandidateClock,
    gross9_loader: Callable[[], VerifiedGross9Clocks],
    comparator_loader: ComparatorLoader,
    claim_creator: Callable[[Mapping[str, Any]], Mapping[str, str]],
    production: bool,
    protocol_paths: Sequence[Path] = source_builder.PROTOCOL_PATHS,
) -> AuthenticatedNoveltyInputs:
    """Durably claim the attempt before any Gross9 or comparator access."""

    if production and source_support.production_authenticated is not True:
        raise NoveltyTerminalError(
            "TUSI-168 production requires fully authenticated source support"
        )
    candidate_sha256 = source_support.payload["clock_artifacts"][
        "primary_sha256"
    ]
    esdi_novelty._validate_hash(candidate_sha256, "candidate clock hash")
    claim_payload = _claim_payload(source_support, candidate_sha256)
    attempt_binding = dict(claim_creator(claim_payload))
    if set(attempt_binding) != {"path", "sha256", "claim_hash"}:
        raise NoveltyTerminalError(
            "TUSI-168 novelty attempt-claim binding schema drift"
        )
    gross9 = gross9_loader()
    registry = frozen_registry(registration)
    loaded_comparators = comparator_loader(registry)
    if isinstance(loaded_comparators, VerifiedComparatorClocks):
        comparators = loaded_comparators
    elif production:
        raise NoveltyTerminalError(
            "TUSI-168 production comparator loader did not retain exact bytes"
        )
    else:
        comparators = VerifiedComparatorClocks(
            clocks=MappingProxyType(dict(loaded_comparators)),
            artifact_bytes=MappingProxyType({}),
            artifact_sha256=MappingProxyType({}),
            registry_sha256=canonical_hash(registry),
            authentication_mode="injected_synthetic",
        )
    if (
        not isinstance(candidate, VerifiedCandidateClock)
        or production
        and candidate.authentication_mode != "canonical_nofollow_committed"
        or production
        and comparators.authentication_mode != "canonical_nofollow_registry"
        or production
        and gross9.authentication_mode != "authoritative_esdi_production"
    ):
        raise NoveltyTerminalError(
            "TUSI-168 claimed production inputs are not authenticated"
        )
    return AuthenticatedNoveltyInputs(
        registration=_thaw_json(registration),
        source_support=source_support,
        candidate=candidate,
        comparators=comparators,
        gross9_artifact=gross9,
        attempt_claim=attempt_binding,
        protocol_paths=tuple(protocol_paths),
        production=production,
    )


def build_report_from_authenticated_inputs(
    inputs: AuthenticatedNoveltyInputs,
) -> dict[str, Any]:
    candidate = _revalidated_candidate_intervals(
        inputs.candidate,
        inputs.source_support,
        production=inputs.production,
    )
    registry = frozen_registry(inputs.registration)
    comparators = _revalidated_comparator_clocks(
        inputs.comparators,
        registry,
        production=inputs.production,
    )
    return _build_report_from_clocks(
        source_support=inputs.source_support,
        candidate=candidate,
        gross9_artifact=inputs.gross9_artifact,
        comparators=comparators,
        registration=inputs.registration,
        attempt_claim=inputs.attempt_claim,
    )


def execute_claimed_novelty(
    *,
    registration: Mapping[str, Any],
    source_support: VerifiedSourceSupport,
    candidate: VerifiedCandidateClock,
    gross9_loader: Callable[[], VerifiedGross9Clocks],
    comparator_loader: ComparatorLoader,
    claim_creator: Callable[[Mapping[str, Any]], Mapping[str, str]],
    production: bool,
    protocol_paths: Sequence[Path] = source_builder.PROTOCOL_PATHS,
) -> dict[str, Any]:
    inputs = collect_claimed_authenticated_inputs(
        registration=registration,
        source_support=source_support,
        candidate=candidate,
        gross9_loader=gross9_loader,
        comparator_loader=comparator_loader,
        claim_creator=claim_creator,
        production=production,
        protocol_paths=protocol_paths,
    )
    return build_report_from_authenticated_inputs(inputs)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-support",
        default=DEFAULT_SOURCE_SUPPORT_PATH.as_posix(),
    )
    parser.add_argument(
        "--candidate-clock",
        default=DEFAULT_PRIMARY_CLOCK_PATH.as_posix(),
    )
    parser.add_argument(
        "--gross9-clocks",
        default=DEFAULT_GROSS9_CLOCKS_PATH.as_posix(),
    )
    arguments = parser.parse_args(argv)
    _canonical_relative_path(
        arguments.source_support,
        DEFAULT_SOURCE_SUPPORT_PATH,
        "source-support report",
    )
    _canonical_relative_path(
        arguments.candidate_clock,
        DEFAULT_PRIMARY_CLOCK_PATH,
        "primary clock artifact",
    )
    _canonical_relative_path(
        arguments.gross9_clocks,
        DEFAULT_GROSS9_CLOCKS_PATH,
        "Gross9 clock artifact",
    )
    if (REPOSITORY_ROOT / DEFAULT_OUTPUT_PATH).exists() or (
        REPOSITORY_ROOT / DEFAULT_ATTEMPT_CLAIM_PATH
    ).exists():
        raise NoveltyTerminalError(
            "TUSI-168 novelty singleton was already attempted"
        )
    registration = verify_preregistration()
    support = load_passed_source_support(production=True)
    candidate = load_candidate_clock_csv(
        DEFAULT_PRIMARY_CLOCK_PATH.as_posix(), support, production=True
    )
    authenticated_inputs = collect_claimed_authenticated_inputs(
        registration=registration,
        source_support=support,
        candidate=candidate,
        gross9_loader=lambda: load_gross9_clock_artifact(
            registration=registration,
            source_support=support,
            path=DEFAULT_GROSS9_CLOCKS_PATH.as_posix(),
            production=True,
        ),
        comparator_loader=load_comparator_artifacts,
        claim_creator=lambda payload: _create_attempt_claim(payload),
        production=True,
    )
    report = build_report_from_authenticated_inputs(authenticated_inputs)
    status = write_once_novelty_json(
        report,
        registration=registration,
        authenticated_inputs=authenticated_inputs,
    )
    print(
        json.dumps(
            {
                "status": status,
                "output": str(DEFAULT_OUTPUT_PATH),
                "passed": report["novelty"]["passed"],
                "manifest_hash": report["manifest_hash"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
