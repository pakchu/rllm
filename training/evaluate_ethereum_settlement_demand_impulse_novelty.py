"""Evaluate the frozen, candidate-outcome-blind ESDI-288 novelty gates.

This module never reconstructs Gross9.  It consumes five hash-bound signed
sleeve clocks reconstructed after source support and before novelty/economics.
Those clocks certify outcome-dependent Gross9 paths; no ESDI market, return, or
PnL metric is opened or computed here.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

from training import build_ethereum_settlement_demand_impulse_source as source_builder
from training import (
    evaluate_ethereum_settlement_demand_impulse_source_support as source_support_evaluator,
)
from training import preregister_ethereum_settlement_demand_impulse as prereg
from training.preregister_ethereum_settlement_demand_impulse import (
    bidirectional_entry_containment,
    entries_in_domain,
    exact_entry_jaccard,
    fraction_at_most,
    fraction_below,
    occupied_bar_jaccard,
    signed_exposure_5m,
    squared_signed_exposure_pearson,
)


PROTOCOL_VERSION = "ethereum_settlement_demand_impulse_novelty_v1"
POLICY_ID = "ESDI-288"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION_PATH = Path(
    "results/ethereum_settlement_demand_impulse_preregistration_2026-07-30.json"
)
PREREGISTRATION_SHA256 = (
    "2a481fc60044d3d468340457d50f92a91f2a52184a464e1a91badfb418bbcaba"
)
PREREGISTRATION_MANIFEST_HASH = (
    "d5279f95cc7b92757aa77ecbbc5835d8b1cc4ce34f5a81d6f279abdcf2fcfe8a"
)
DEFAULT_SOURCE_SUPPORT_PATH = Path(
    "results/ethereum_settlement_demand_impulse_source_support_2026-07-30.json"
)
SOURCE_SUPPORT_PROTOCOL_VERSION = (
    "ethereum_settlement_demand_impulse_source_support_v1"
)
GROSS9_CLOCKS_PROTOCOL_VERSION = (
    "ethereum_settlement_demand_impulse_gross9_clocks_v1"
)
DEFAULT_GROSS9_CLOCKS_PATH = Path(
    "results/ethereum_settlement_demand_impulse_gross9_clocks_2026-07-30.json"
)
DEFAULT_OUTPUT_PATH = Path(
    "results/ethereum_settlement_demand_impulse_novelty_2026-07-30.json"
)
DEFAULT_ATTEMPT_CLAIM_PATH = Path(
    "results/"
    "ethereum_settlement_demand_impulse_novelty_attempt_claim_2026-07-30.json"
)
ATTEMPT_CLAIM_PROTOCOL_VERSION = (
    "ethereum_settlement_demand_impulse_novelty_attempt_claim_v1"
)
GROSS9_DOMAIN = ("2023-06-01T00:00:00Z", "2026-06-01T00:00:00Z")
GROSS9_SLEEVES = tuple(prereg.GROSS9_WEIGHTS)
MINIMUM_GATING_ENTRIES = 10
SOURCE_SUPPORT_EVIDENCE_BOUNDARY = {
    "official_ethereum_raw_rows_opened": 0,
    "official_ethereum_epoch_rows_opened": 0,
    "synthetic_epoch_rows_processed": 0,
    "comparator_rows_opened": 0,
    "market_rows_opened": 0,
    "funding_rows_opened": 0,
    "outcome_rows_opened": 0,
    "outcomes_computed": False,
    "network_calls": 0,
}
GROSS9_CLOCK_EVIDENCE_BOUNDARY = {
    "gross9_runtime_market_and_feature_rows_opened": True,
    "gross9_outcome_dependent_path_rows_opened": True,
    "gross9_full_domain_clock_paths_reproduced": True,
    "full_domain_paths_used_for_preregistered_structural_novelty_veto": True,
    "future_rows_used_for_economic_weight_ranking": False,
    "future_rows_used_for_structural_candidate_veto": True,
    "gross9_portfolio_return_or_pnl_metrics_computed": False,
}
NOVELTY_EVIDENCE_BOUNDARY = {
    "candidate_market_rows_opened": False,
    "candidate_outcome_rows_opened": False,
    "gross9_clock_artifact_bytes_opened": True,
    "gross9_outcome_dependent_clock_paths_certified": True,
    "gross9_full_domain_paths_used_for_preregistered_structural_novelty_veto": True,
    "future_rows_used_for_economic_weight_ranking": False,
    "future_rows_used_for_structural_candidate_veto": True,
    "portfolio_return_or_pnl_metrics_computed": False,
}
SOURCE_SUPPORT_CHECK_NAMES = frozenset(
    {
        "source_exact_epochs",
        "source_missing_epochs_zero",
        "source_dual_replay_differences_zero",
        "source_boundary_header_differences_zero",
        "future_append_selection_differences_zero",
        "identity_clock_side_rank_tie_source_hash_reproducible",
        "selection_total_min",
        "selection_2023H2_min",
        "selection_2024H1_min",
        "selection_2024H2_min",
        "selection_each_side_min",
        "selection_maximum_month_share",
        "future25_total_min",
        "future25_each_side_min",
        "future25_maximum_month_share",
        "future26_total_min",
        "future26_each_side_min",
        "future26_maximum_month_share",
        "maximum_accepted_entry_gap_days",
        "maximum_same_side_run",
        "base_fee_one_epoch_stale_exact_entry_jaccard_strict",
        "base_fee_one_epoch_stale_candidate_24h_containment_strict",
        "gas_utilization_only_exact_entry_jaccard_strict",
        "gas_utilization_only_candidate_24h_containment_strict",
        "base_fee_no_tail_exact_entry_jaccard_strict",
        "base_fee_no_tail_candidate_24h_containment_strict",
    }
)
SOURCE_SUPPORT_ARTIFACT_KEYS = frozenset(
    {
        "protocol_version",
        "policy_id",
        "status",
        "terminal",
        "artifact_eligible",
        "decision",
        "support_passed",
        "preregistration",
        "attempt_claim",
        "source_contract",
        "feature_rows",
        "feature_rank_tie_state_sha256",
        "raw_candidate_counts",
        "accepted_clock_counts",
        "support_audit",
        "support_checks",
        "future_append_selection_invariance",
        "clock_artifacts",
        "evidence_boundary",
        "later_stage_artifacts_opened",
        "manifest_hash",
    }
)
SOURCE_SUPPORT_SOURCE_CONTRACT_KEYS = frozenset(
    {
        "columns",
        "rows",
        "artifact_eligible",
        "source_manifest_path",
        "source_manifest_sha256",
        "source_manifest_hash",
        "raw_source_path",
        "raw_source_bytes",
        "raw_source_rows_decoded",
        "raw_source_sha256",
        "epoch_csv_path",
        "epoch_csv_bytes",
        "epoch_csv_rows_decoded",
        "epoch_csv_sha256",
        "pre_replay_protocol_seal",
        "replay_claim",
        "missing_epochs",
        "dual_replay_differences",
        "boundary_header_differences",
    }
)
SOURCE_SUPPORT_AUDIT_KEYS = frozenset(
    {
        "clock_stats",
        "selection_report_counts",
        "maximum_accepted_entry_gap_seconds",
        "maximum_same_side_run",
        "independent_control_metrics",
    }
)
SOURCE_SUPPORT_APPEND_INVARIANCE_KEYS = frozenset(
    {
        "passed",
        "selection_end_utc",
        "full_rebuild_selection_rows",
        "prefix_rebuild_selection_rows",
        "full_rebuild_selection_sha256",
        "prefix_rebuild_selection_sha256",
    }
)
_INTEGER_TIMESTAMP = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_UTC_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")


class NoveltyTerminalError(RuntimeError):
    """A frozen-input or undefined-metric failure that terminates novelty."""


@dataclass(frozen=True, order=True)
class SignedInterval:
    entry: int
    exit: int
    side: int


@dataclass(frozen=True)
class ComparatorClock:
    comparator_id: str
    capability: str
    entries: tuple[int, ...]
    intervals: tuple[SignedInterval, ...] | None
    artifact_name: str
    group: str | None = None


@dataclass(frozen=True)
class VerifiedSourceSupport:
    path: Path
    raw_bytes: bytes
    sha256: str
    manifest_hash: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class VerifiedGross9Clocks:
    path: Path
    raw_bytes: bytes
    sha256: str
    manifest_hash: str
    authority_hash: str
    clocks: Mapping[str, tuple[SignedInterval, ...]]
    payload: Mapping[str, Any]


ComparatorLoader = Callable[
    [Mapping[str, Mapping[str, Any]]], Mapping[str, ComparatorClock]
]
Gross9SleeveClocks = Mapping[str, Sequence[SignedInterval | tuple[int, int, int]]]


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


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NoveltyTerminalError(f"invalid JSON artifact: {path}") from error
    if not isinstance(payload, dict):
        raise NoveltyTerminalError(f"JSON artifact is not an object: {path}")
    return payload


def verify_preregistration(
    path: str | Path = PREREGISTRATION_PATH,
) -> dict[str, Any]:
    candidate = Path(path)
    if sha256_file(candidate) != PREREGISTRATION_SHA256:
        raise NoveltyTerminalError("ESDI-288 preregistration artifact hash drift")
    payload = _read_json(candidate)
    if payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise NoveltyTerminalError("ESDI-288 preregistration manifest hash drift")
    core = _thaw_json(
        {
            key: value
            for key, value in payload.items()
            if key != "manifest_hash"
        }
    )
    if canonical_hash(core) != PREREGISTRATION_MANIFEST_HASH:
        raise NoveltyTerminalError(
            "ESDI-288 preregistration canonical manifest hash drift"
        )
    if payload.get("policy_id") != POLICY_ID:
        raise NoveltyTerminalError("ESDI-288 preregistration policy drift")
    identity = payload.get("frozen_preregistration", {}).get(
        "repository_identity", {}
    )
    producer_hash = identity.get("sha256", {}).get(
        "training/preregister_ethereum_settlement_demand_impulse.py"
    )
    if producer_hash is None or sha256_file(prereg.__file__) != producer_hash:
        raise NoveltyTerminalError(
            "ESDI-288 preregistered executable helper source drift"
        )
    return payload


def _expected_preregistration_binding() -> dict[str, str]:
    return {
        "path": str(PREREGISTRATION_PATH),
        "sha256": PREREGISTRATION_SHA256,
        "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
    }


def _decode_json_bytes(raw_bytes: bytes, label: str) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise NoveltyTerminalError(f"ESDI-288 invalid {label} JSON") from error
    if not isinstance(payload, dict):
        raise NoveltyTerminalError(f"ESDI-288 {label} is not an object")
    return payload


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _require_canonical_committed_clean(
    path: str | Path,
    expected: Path,
    label: str,
    raw_bytes: bytes,
) -> Path:
    candidate = Path(path)
    if candidate != expected:
        raise NoveltyTerminalError(
            f"ESDI-288 {label} must use its canonical path"
        )
    command_prefix = ["git", "-C", str(REPOSITORY_ROOT)]
    tracked = subprocess.run(
        [
            *command_prefix,
            "ls-files",
            "--error-unmatch",
            "--",
            str(expected),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    status = subprocess.run(
        [
            *command_prefix,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            str(expected),
        ],
        capture_output=True,
        check=False,
    )
    head_blob = subprocess.run(
        [
            *command_prefix,
            "rev-parse",
            f"HEAD:{expected}",
        ],
        capture_output=True,
        check=False,
    )
    supplied_blob = subprocess.run(
        [*command_prefix, "hash-object", "--stdin"],
        input=raw_bytes,
        capture_output=True,
        check=False,
    )
    resolved = REPOSITORY_ROOT / expected
    if (
        tracked.returncode != 0
        or status.returncode != 0
        or status.stdout
        or head_blob.returncode != 0
        or supplied_blob.returncode != 0
        or head_blob.stdout.strip() != supplied_blob.stdout.strip()
        or not resolved.is_file()
        or resolved.is_symlink()
    ):
        raise NoveltyTerminalError(
            f"ESDI-288 {label} must be committed and clean"
        )
    return resolved


def require_passed_source_support(
    payload: Mapping[str, Any],
    *,
    production: bool = False,
) -> None:
    """Require the sole exact source-support pass schema."""

    if not isinstance(payload, Mapping):
        raise NoveltyTerminalError("ESDI-288 source-support artifact is invalid")
    if set(payload) != SOURCE_SUPPORT_ARTIFACT_KEYS:
        raise NoveltyTerminalError("ESDI-288 source-support exact schema drift")
    internal_hash = payload.get("manifest_hash")
    _validate_hash(internal_hash, "source-support manifest hash")
    core = _thaw_json(
        {
            key: value
            for key, value in payload.items()
            if key != "manifest_hash"
        }
    )
    if internal_hash != canonical_hash(core):
        raise NoveltyTerminalError(
            "ESDI-288 source-support internal manifest drift"
        )
    if payload.get("support_passed") is not True:
        raise NoveltyTerminalError(
            "ESDI-288 source support did not pass before comparator access"
        )
    exact_values = {
        "protocol_version": SOURCE_SUPPORT_PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "status": "support_passed_terminal",
        "terminal": True,
        "artifact_eligible": True,
        "decision": "SOURCE_SUPPORT_PASS",
        "later_stage_artifacts_opened": False,
    }
    if any(payload.get(key) != value for key, value in exact_values.items()):
        raise NoveltyTerminalError("ESDI-288 source-support pass schema drift")
    if any(
        alias in payload
        for alias in ("source_support_passed", "passed", "support")
    ):
        raise NoveltyTerminalError("ESDI-288 source-support alias field forbidden")
    if payload.get("preregistration") != _expected_preregistration_binding():
        raise NoveltyTerminalError(
            "ESDI-288 source-support preregistration binding drift"
        )
    attempt_claim = payload.get("attempt_claim")
    if (
        not isinstance(attempt_claim, Mapping)
        or set(attempt_claim) != {"path", "sha256", "claim_hash"}
        or attempt_claim.get("path")
        != str(source_support_evaluator.DEFAULT_ATTEMPT_CLAIM)
    ):
        raise NoveltyTerminalError(
            "ESDI-288 source-support attempt-claim binding drift"
        )
    _validate_hash(
        attempt_claim.get("sha256"),
        "source-support attempt-claim file hash",
    )
    _validate_hash(
        attempt_claim.get("claim_hash"),
        "source-support attempt-claim hash",
    )
    checks = payload.get("support_checks")
    if (
        not isinstance(checks, Mapping)
        or set(checks) != SOURCE_SUPPORT_CHECK_NAMES
        or any(value is not True for value in checks.values())
    ):
        raise NoveltyTerminalError(
            "ESDI-288 every exact source-support check must pass"
        )
    clock_artifacts = payload.get("clock_artifacts")
    if (
        not isinstance(clock_artifacts, Mapping)
        or set(clock_artifacts) != {"primary_sha256", "controls_sha256"}
    ):
        raise NoveltyTerminalError("ESDI-288 source-support clock binding drift")
    _validate_hash(clock_artifacts["primary_sha256"], "primary clock hash")
    _validate_hash(clock_artifacts["controls_sha256"], "control clock hash")
    source_contract = payload.get("source_contract")
    if (
        not isinstance(source_contract, Mapping)
        or set(source_contract) != SOURCE_SUPPORT_SOURCE_CONTRACT_KEYS
        or source_contract.get("columns")
        != list(source_support_evaluator.SOURCE_COLUMNS)
        or source_contract.get("artifact_eligible") is not True
        or type(source_contract.get("rows")) is not int
        or source_contract["rows"] != source_builder.EPOCH_COUNT
        or source_contract.get("source_manifest_path")
        != str(source_support_evaluator.DEFAULT_SOURCE_MANIFEST)
        or source_contract.get("raw_source_path")
        != str(source_support_evaluator.DEFAULT_RAW_SOURCE)
        or source_contract.get("epoch_csv_path")
        != str(source_support_evaluator.DEFAULT_EPOCH_SOURCE)
        or type(source_contract.get("source_manifest_path")) is not str
        or type(source_contract.get("raw_source_bytes")) is not int
        or source_contract["raw_source_bytes"] <= 0
        or type(source_contract.get("epoch_csv_bytes")) is not int
        or source_contract["epoch_csv_bytes"] <= 0
        or type(source_contract.get("raw_source_rows_decoded")) is not int
        or source_contract["raw_source_rows_decoded"]
        != source_builder.REQUEST_COUNT
        or type(source_contract.get("epoch_csv_rows_decoded")) is not int
        or source_contract["epoch_csv_rows_decoded"] != source_contract["rows"]
        or source_contract.get("missing_epochs") != 0
        or source_contract.get("dual_replay_differences") != 0
        or source_contract.get("boundary_header_differences") != 0
        or not isinstance(
            source_contract.get("pre_replay_protocol_seal"), Mapping
        )
        or not isinstance(source_contract.get("replay_claim"), Mapping)
        or type(payload.get("feature_rows")) is not int
        or payload["feature_rows"] <= 0
        or not isinstance(payload.get("raw_candidate_counts"), Mapping)
        or not isinstance(payload.get("accepted_clock_counts"), Mapping)
        or not isinstance(payload.get("support_audit"), Mapping)
    ):
        raise NoveltyTerminalError("ESDI-288 source-support report body drift")
    for field in (
        "source_manifest_sha256",
        "source_manifest_hash",
        "raw_source_sha256",
        "epoch_csv_sha256",
    ):
        _validate_hash(source_contract[field], f"source contract {field}")
    replay_claim = source_contract["replay_claim"]
    if (
        set(replay_claim) != {"path", "sha256", "claim_hash"}
        or replay_claim.get("path")
        != source_support_evaluator.DEFAULT_REPLAY_CLAIM.as_posix()
    ):
        raise NoveltyTerminalError("ESDI-288 replay-claim binding drift")
    _validate_hash(replay_claim.get("sha256"), "replay-claim file hash")
    _validate_hash(replay_claim.get("claim_hash"), "replay-claim hash")
    expected_control_names = set(source_support_evaluator.CONTROL_ORDER)
    for field in ("raw_candidate_counts", "accepted_clock_counts"):
        counts = payload[field]
        if (
            set(counts) != expected_control_names
            or any(type(value) is not int or value < 0 for value in counts.values())
        ):
            raise NoveltyTerminalError(
                f"ESDI-288 source-support {field} drift"
            )
    support_audit = payload["support_audit"]
    if set(support_audit) != SOURCE_SUPPORT_AUDIT_KEYS:
        raise NoveltyTerminalError("ESDI-288 source-support audit schema drift")
    _validate_hash(
        payload.get("feature_rank_tie_state_sha256"),
        "feature rank/tie hash",
    )
    append_invariance = payload.get("future_append_selection_invariance")
    if (
        not isinstance(append_invariance, Mapping)
        or set(append_invariance) != SOURCE_SUPPORT_APPEND_INVARIANCE_KEYS
        or append_invariance.get("passed") is not True
        or append_invariance.get("selection_end_utc")
        != "2025-01-01T00:00:00Z"
        or type(append_invariance.get("full_rebuild_selection_rows")) is not int
        or type(append_invariance.get("prefix_rebuild_selection_rows")) is not int
        or append_invariance["full_rebuild_selection_rows"]
        != append_invariance["prefix_rebuild_selection_rows"]
    ):
        raise NoveltyTerminalError(
            "ESDI-288 source-support append-invariance drift"
        )
    _validate_hash(
        append_invariance.get("full_rebuild_selection_sha256"),
        "full selection rebuild hash",
    )
    _validate_hash(
        append_invariance.get("prefix_rebuild_selection_sha256"),
        "prefix selection rebuild hash",
    )
    if (
        append_invariance["full_rebuild_selection_sha256"]
        != append_invariance["prefix_rebuild_selection_sha256"]
    ):
        raise NoveltyTerminalError(
            "ESDI-288 source-support append-invariance hash drift"
        )
    expected_evidence_boundary = {
        **SOURCE_SUPPORT_EVIDENCE_BOUNDARY,
        "official_ethereum_raw_rows_opened": source_contract[
            "raw_source_rows_decoded"
        ],
        "official_ethereum_epoch_rows_opened": source_contract[
            "epoch_csv_rows_decoded"
        ],
    }
    if payload.get("evidence_boundary") != expected_evidence_boundary:
        raise NoveltyTerminalError(
            "ESDI-288 source-support evidence boundary drift"
        )
    if production:
        claim_path = (
            REPOSITORY_ROOT / source_support_evaluator.DEFAULT_ATTEMPT_CLAIM
        )
        try:
            claim_raw = claim_path.read_bytes()
        except OSError as error:
            raise NoveltyTerminalError(
                "ESDI-288 source-support attempt claim is unreadable"
            ) from error
        _require_canonical_committed_clean(
            source_support_evaluator.DEFAULT_ATTEMPT_CLAIM,
            source_support_evaluator.DEFAULT_ATTEMPT_CLAIM,
            "source-support attempt claim",
            claim_raw,
        )
        try:
            observed_attempt_claim = (
                source_support_evaluator.load_attempt_claim()
            )
        except Exception as error:
            raise NoveltyTerminalError(
                "ESDI-288 source-support attempt claim did not validate"
            ) from error
        if dict(attempt_claim) != observed_attempt_claim:
            raise NoveltyTerminalError(
                "ESDI-288 source-support attempt claim differs"
            )
        try:
            frame, source_audit = source_support_evaluator.load_source_manifest()
        except Exception as error:
            raise NoveltyTerminalError(
                "ESDI-288 source-support source authentication failed"
            ) from error
        expected_source_contract = {
            "columns": list(source_support_evaluator.SOURCE_COLUMNS),
            "rows": len(frame),
            **source_audit,
        }
        if dict(source_contract) != expected_source_contract:
            raise NoveltyTerminalError(
                "ESDI-288 source-support source contract is not producer-exact"
            )


def parse_passed_source_support_bytes(
    raw_bytes: bytes,
    *,
    path: str | Path,
    production: bool,
) -> VerifiedSourceSupport:
    if not isinstance(raw_bytes, bytes):
        raise NoveltyTerminalError("ESDI-288 source-support bytes are invalid")
    if production:
        _require_canonical_committed_clean(
            path,
            DEFAULT_SOURCE_SUPPORT_PATH,
            "source-support artifact",
            raw_bytes,
        )
        artifact_path = DEFAULT_SOURCE_SUPPORT_PATH
    else:
        artifact_path = Path(path)
    payload = _decode_json_bytes(raw_bytes, "source-support artifact")
    if raw_bytes != source_support_evaluator._json_bytes(payload):
        raise NoveltyTerminalError(
            "ESDI-288 source-support serialization is not producer-canonical"
        )
    require_passed_source_support(payload, production=production)
    return VerifiedSourceSupport(
        path=artifact_path,
        raw_bytes=raw_bytes,
        sha256=sha256_bytes(raw_bytes),
        manifest_hash=payload["manifest_hash"],
        payload=_freeze_json(payload),
    )


def load_passed_source_support(
    path: str | Path = DEFAULT_SOURCE_SUPPORT_PATH,
    *,
    production: bool = True,
) -> VerifiedSourceSupport:
    if production and Path(path) != DEFAULT_SOURCE_SUPPORT_PATH:
        raise NoveltyTerminalError(
            "ESDI-288 source-support artifact must use its canonical path"
        )
    resolved = (
        REPOSITORY_ROOT / DEFAULT_SOURCE_SUPPORT_PATH
        if production
        else Path(path)
    )
    try:
        raw_bytes = resolved.read_bytes()
    except OSError as error:
        raise NoveltyTerminalError(
            "ESDI-288 source-support artifact is unreadable"
        ) from error
    return parse_passed_source_support_bytes(
        raw_bytes,
        path=path,
        production=production,
    )


def _parse_timestamp(value: str) -> int:
    if not isinstance(value, str) or value != value.strip() or not value:
        raise NoveltyTerminalError("ESDI-288 timestamp is not canonical")
    if _INTEGER_TIMESTAMP.fullmatch(value):
        return int(value)
    if not _UTC_TIMESTAMP.fullmatch(value):
        raise NoveltyTerminalError("ESDI-288 timestamp is not exact UTC seconds")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise NoveltyTerminalError("ESDI-288 timestamp is invalid") from error
    return int(parsed.timestamp())


def _parse_side(value: str) -> int:
    if value == "LONG":
        return 1
    if value == "SHORT":
        return -1
    raise NoveltyTerminalError("ESDI-288 side must be exactly LONG or SHORT")


def _domain_seconds(domain: Sequence[str]) -> tuple[int, int]:
    if (
        not isinstance(domain, (list, tuple))
        or len(domain) != 2
        or not all(isinstance(item, str) for item in domain)
    ):
        raise NoveltyTerminalError("ESDI-288 comparator domain is invalid")
    start, end = (_parse_timestamp(item) for item in domain)
    if start % 300 or end % 300 or end <= start:
        raise NoveltyTerminalError("ESDI-288 comparator domain is not a 5m range")
    return start, end


def _validate_hash(value: Any, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise NoveltyTerminalError(f"ESDI-288 invalid {label}")


def validate_registry(
    registry: Mapping[str, Mapping[str, Any]],
    expected: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Validate the exhaustive 18-artifact registry and every frozen field."""

    frozen = prereg.frozen_comparator_registry() if expected is None else expected
    if not isinstance(registry, Mapping) or dict(registry) != dict(frozen):
        raise NoveltyTerminalError("ESDI-288 comparator registry drift")
    if expected is None and len(registry) != 18:
        raise NoveltyTerminalError("ESDI-288 comparator registry is not exhaustive")
    for name, spec in registry.items():
        if not isinstance(name, str) or not isinstance(spec, Mapping):
            raise NoveltyTerminalError("ESDI-288 comparator registry is malformed")
        _validate_hash(spec.get("sha256"), f"{name} artifact hash")
        _validate_hash(spec.get("header_line_sha256"), f"{name} header hash")
        _domain_seconds(spec.get("comparison_domain", ()))
        required = spec.get("required_columns")
        filters = spec.get("filters")
        if (
            not isinstance(required, list)
            or not required
            or required != sorted(set(required))
            or not isinstance(filters, Mapping)
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in filters.items()
            )
            or not set(filters).issubset(required)
        ):
            raise NoveltyTerminalError(f"ESDI-288 invalid schema for {name}")
        capability = spec.get("capability")
        if capability == "directional_interval":
            for field in ("entry_column", "exit_column", "side_column"):
                if spec.get(field) not in required:
                    raise NoveltyTerminalError(
                        f"ESDI-288 missing {field} binding for {name}"
                    )
            group_column = spec.get("group_column")
            groups = spec.get("groups")
            if group_column is None:
                if groups != []:
                    raise NoveltyTerminalError(f"ESDI-288 unexpected groups for {name}")
            elif (
                group_column not in required
                or not isinstance(groups, list)
                or not groups
                or len(groups) != len(set(groups))
                or spec.get("each_group_is_a_separate_comparator") is not True
            ):
                raise NoveltyTerminalError(f"ESDI-288 invalid groups for {name}")
        elif capability is None:
            group_column = spec.get("group_column")
            capability_column = spec.get("capability_column")
            directional = spec.get("directional_interval_groups")
            timestamp_only = spec.get("timestamp_only_groups")
            if (
                group_column not in required
                or capability_column not in required
                or not isinstance(directional, list)
                or not isinstance(timestamp_only, list)
                or not directional
                or not timestamp_only
                or set(directional) & set(timestamp_only)
                or len(directional) != len(set(directional))
                or len(timestamp_only) != len(set(timestamp_only))
                or spec.get("each_group_is_a_separate_comparator") is not True
            ):
                raise NoveltyTerminalError(f"ESDI-288 invalid capability bundle {name}")
        else:
            raise NoveltyTerminalError(f"ESDI-288 unknown capability for {name}")


def frozen_registry(
    registration: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    novelty = registration.get("novelty")
    registry = novelty.get("frozen_comparator_artifacts") if isinstance(
        novelty, Mapping
    ) else None
    if not isinstance(registry, Mapping):
        raise NoveltyTerminalError("ESDI-288 preregistration registry is missing")
    copied = json.loads(json.dumps(registry))
    validate_registry(copied)
    return copied


def _artifact_path(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str):
        raise NoveltyTerminalError("ESDI-288 comparator path is invalid")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise NoveltyTerminalError("ESDI-288 comparator path escapes repository")
    return root / path


def _decoded_csv(
    artifact_name: str,
    spec: Mapping[str, Any],
    root: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    path = _artifact_path(root, spec.get("path"))
    try:
        compressed = path.read_bytes()
    except OSError as error:
        raise NoveltyTerminalError(
            f"ESDI-288 comparator artifact missing: {artifact_name}"
        ) from error
    if sha256_bytes(compressed) != spec["sha256"]:
        raise NoveltyTerminalError(
            f"ESDI-288 comparator artifact hash drift: {artifact_name}"
        )
    try:
        decompressed = gzip.decompress(compressed)
    except (OSError, EOFError) as error:
        raise NoveltyTerminalError(
            f"ESDI-288 comparator gzip invalid: {artifact_name}"
        ) from error
    header = decompressed.splitlines(keepends=True)[:1]
    if not header or sha256_bytes(header[0]) != spec["header_line_sha256"]:
        raise NoveltyTerminalError(
            f"ESDI-288 comparator header hash drift: {artifact_name}"
        )
    try:
        text = decompressed.decode("utf-8")
    except UnicodeDecodeError as error:
        raise NoveltyTerminalError(
            f"ESDI-288 comparator is not UTF-8: {artifact_name}"
        ) from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    columns = reader.fieldnames
    if (
        columns is None
        or len(columns) != len(set(columns))
        or not set(spec["required_columns"]).issubset(columns)
    ):
        raise NoveltyTerminalError(
            f"ESDI-288 comparator required columns missing: {artifact_name}"
        )
    try:
        rows = list(reader)
    except csv.Error as error:
        raise NoveltyTerminalError(
            f"ESDI-288 comparator CSV invalid: {artifact_name}"
        ) from error
    if any(None in row for row in rows):
        raise NoveltyTerminalError(
            f"ESDI-288 comparator row width drift: {artifact_name}"
        )
    return columns, rows


def _filtered_rows(
    rows: Iterable[Mapping[str, str]], filters: Mapping[str, str]
) -> list[Mapping[str, str]]:
    return [
        row
        for row in rows
        if all(row.get(column) == value for column, value in filters.items())
    ]


def _clock_from_rows(
    *,
    comparator_id: str,
    artifact_name: str,
    group: str | None,
    capability: str,
    rows: Sequence[Mapping[str, str]],
    spec: Mapping[str, Any],
) -> ComparatorClock:
    entry_column = spec["entry_column"]
    entries = tuple(_parse_timestamp(row[entry_column]) for row in rows)
    try:
        entries_in_domain(entries, 0, 2**63 - 1)
    except ValueError as error:
        raise NoveltyTerminalError(
            f"ESDI-288 duplicate or unsorted entries: {comparator_id}"
        ) from error
    if capability == "timestamp_only":
        return ComparatorClock(
            comparator_id, capability, entries, None, artifact_name, group
        )
    if capability != "directional_interval":
        raise NoveltyTerminalError(
            f"ESDI-288 unknown comparator capability: {comparator_id}"
        )
    intervals = tuple(
        SignedInterval(
            entry,
            _parse_timestamp(row[spec["exit_column"]]),
            _parse_side(row[spec["side_column"]]),
        )
        for entry, row in zip(entries, rows)
    )
    return ComparatorClock(
        comparator_id, capability, entries, intervals, artifact_name, group
    )


def load_comparator_artifacts(
    registry: Mapping[str, Mapping[str, Any]],
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
    expected_registry: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, ComparatorClock]:
    """Load every hash-bound comparator and split every frozen group."""

    validate_registry(registry, expected_registry)
    root = Path(repository_root)
    clocks: dict[str, ComparatorClock] = {}
    for artifact_name, spec in registry.items():
        _, physical_rows = _decoded_csv(artifact_name, spec, root)
        rows = _filtered_rows(physical_rows, spec["filters"])
        if not rows:
            raise NoveltyTerminalError(
                f"ESDI-288 comparator has no rows after filters: {artifact_name}"
            )
        if spec.get("capability") == "directional_interval":
            group_column = spec.get("group_column")
            groups = spec.get("groups") if group_column is not None else [None]
            observed = (
                {row[group_column] for row in rows}
                if group_column is not None
                else {None}
            )
            if observed != set(groups):
                raise NoveltyTerminalError(
                    f"ESDI-288 comparator groups drift: {artifact_name}"
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
                clocks[comparator_id] = _clock_from_rows(
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
                    f"ESDI-288 comparator bundle groups drift: {artifact_name}"
                )
            for group, capability in capabilities.items():
                group_rows = [row for row in rows if row[group_column] == group]
                if any(row[capability_column] != capability for row in group_rows):
                    raise NoveltyTerminalError(
                        f"ESDI-288 frozen capability drift: {artifact_name}:{group}"
                    )
                comparator_id = f"{artifact_name}:{group}"
                clocks[comparator_id] = _clock_from_rows(
                    comparator_id=comparator_id,
                    artifact_name=artifact_name,
                    group=group,
                    capability=capability,
                    rows=group_rows,
                    spec=spec,
                )
    return clocks


def _canonical_intervals(
    values: Sequence[SignedInterval | tuple[int, int, int]],
    label: str,
) -> tuple[SignedInterval, ...]:
    intervals: list[SignedInterval] = []
    for value in values:
        if isinstance(value, SignedInterval):
            interval = value
        elif (
            isinstance(value, tuple)
            and len(value) == 3
            and all(type(item) is int for item in value)
        ):
            interval = SignedInterval(*value)
        else:
            raise NoveltyTerminalError(f"ESDI-288 invalid interval: {label}")
        intervals.append(interval)
    entries = tuple(interval.entry for interval in intervals)
    try:
        entries_in_domain(entries, 0, 2**63 - 1)
    except ValueError as error:
        raise NoveltyTerminalError(
            f"ESDI-288 duplicate or unsorted entries: {label}"
        ) from error
    previous_exit = 0
    for interval in intervals:
        if (
            type(interval.entry) is not int
            or type(interval.exit) is not int
            or type(interval.side) is not int
            or interval.entry < 0
            or interval.entry % 300
            or interval.exit % 300
            or interval.entry >= interval.exit
            or interval.side not in {-1, 1}
            or interval.entry < previous_exit
        ):
            raise NoveltyTerminalError(
                f"ESDI-288 invalid sorted nonoverlap interval: {label}"
            )
        previous_exit = interval.exit
    return tuple(intervals)


def _intervals_in_domain(
    values: Sequence[SignedInterval | tuple[int, int, int]],
    start: int,
    end: int,
    label: str,
) -> tuple[SignedInterval, ...]:
    canonical = _canonical_intervals(values, label)
    selected = tuple(
        interval
        for interval in canonical
        if start <= interval.entry and interval.exit <= end
    )
    try:
        signed_exposure_5m(
            ((row.entry, row.exit, row.side) for row in selected), start, end
        )
    except ValueError as error:
        raise NoveltyTerminalError(
            f"ESDI-288 invalid contained interval clock: {label}"
        ) from error
    return selected


def _fraction_payload(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _pearson_payload(value: Fraction) -> dict[str, Any]:
    return {
        "squared_exact": _fraction_payload(value),
        "absolute_correlation": math.sqrt(
            value.numerator / value.denominator
        ),
    }


def inclusive_fraction_gate(value: Fraction, numerator: int, denominator: int) -> bool:
    """Expose the preregistered inclusive rational gate without float coercion."""

    return fraction_at_most(value, numerator, denominator)


def strict_fraction_gate(value: Fraction, numerator: int, denominator: int) -> bool:
    """Expose the preregistered strict rational gate for reusable audit checks."""

    return fraction_below(value, numerator, denominator)


def evaluate_prior_comparator(
    candidate: Sequence[SignedInterval | tuple[int, int, int]],
    comparator: ComparatorClock,
    domain: Sequence[str],
) -> dict[str, Any]:
    start, end = _domain_seconds(domain)
    candidate_rows = _intervals_in_domain(candidate, start, end, "candidate")
    candidate_entries = tuple(row.entry for row in candidate_rows)
    if comparator.capability == "timestamp_only":
        try:
            comparator_entries = entries_in_domain(
                comparator.entries, start, end
            )
        except ValueError as error:
            raise NoveltyTerminalError(
                f"ESDI-288 invalid timestamp-only clock: "
                f"{comparator.comparator_id}"
            ) from error
        comparator_rows: tuple[SignedInterval, ...] | None = None
    elif comparator.capability == "directional_interval":
        if comparator.intervals is None:
            raise NoveltyTerminalError(
                f"ESDI-288 directional intervals missing: {comparator.comparator_id}"
            )
        canonical_comparator_rows = _canonical_intervals(
            comparator.intervals,
            comparator.comparator_id,
        )
        if (
            tuple(row.entry for row in canonical_comparator_rows)
            != comparator.entries
        ):
            raise NoveltyTerminalError(
                f"ESDI-288 comparator entry/interval mismatch: "
                f"{comparator.comparator_id}"
            )
        comparator_rows = _intervals_in_domain(
            canonical_comparator_rows,
            start,
            end,
            comparator.comparator_id,
        )
        comparator_entries = tuple(row.entry for row in comparator_rows)
    else:
        raise NoveltyTerminalError(
            f"ESDI-288 unknown capability: {comparator.comparator_id}"
        )
    try:
        jaccard = exact_entry_jaccard(candidate_entries, comparator_entries)
        containment = bidirectional_entry_containment(
            candidate_entries, comparator_entries, 24 * 60 * 60
        )
    except ValueError as error:
        raise NoveltyTerminalError(
            f"ESDI-288 undefined entry metric: {comparator.comparator_id}"
        ) from error
    metrics: dict[str, Any] = {
        "exact_entry_jaccard": _fraction_payload(jaccard),
        "candidate_24h_containment": _fraction_payload(containment),
    }
    checks = {
        "exact_entry_jaccard": fraction_at_most(jaccard, 1, 5),
        "candidate_24h_containment": fraction_at_most(containment, 1, 2),
    }
    if comparator.capability == "timestamp_only":
        metrics["squared_signed_exposure_pearson"] = {
            "applicable": False,
            "reason": "frozen_timestamp_only_capability",
        }
    else:
        assert comparator_rows is not None
        try:
            candidate_exposure = signed_exposure_5m(
                ((row.entry, row.exit, row.side) for row in candidate_rows),
                start,
                end,
            )
            comparator_exposure = signed_exposure_5m(
                ((row.entry, row.exit, row.side) for row in comparator_rows),
                start,
                end,
            )
            pearson = squared_signed_exposure_pearson(
                candidate_exposure, comparator_exposure
            )
        except ValueError as error:
            raise NoveltyTerminalError(
                f"ESDI-288 undefined exposure metric: {comparator.comparator_id}"
            ) from error
        metrics["squared_signed_exposure_pearson"] = {
            "applicable": True,
            **_pearson_payload(pearson),
        }
        checks["squared_signed_exposure_pearson"] = fraction_at_most(
            pearson, 4, 25
        )
    gating = len(comparator_entries) >= MINIMUM_GATING_ENTRIES
    return {
        "comparator_id": comparator.comparator_id,
        "artifact_name": comparator.artifact_name,
        "group": comparator.group,
        "capability": comparator.capability,
        "comparison_domain": list(domain),
        "candidate_entries": len(candidate_entries),
        "comparator_entries": len(comparator_entries),
        "minimum_count_after_common_domain_filter": True,
        "gating": gating,
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()) if gating else True,
        "would_pass_if_gating": all(checks.values()),
    }


def gross9_frozen_contract_validation(
    registration: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        reconstruction = registration["gross9"]["authority"][
            "clock_reconstruction"
        ]
        common_domain = registration["novelty"]["gross9_common_domain"]
    except (KeyError, TypeError) as error:
        raise NoveltyTerminalError(
            "ESDI-288 Gross9 frozen reconstruction contract is missing"
        ) from error
    expected_reconstruction = {
        "stage": "after ESDI source-support pass and before ESDI economics",
        "five_signed_sleeves_required": True,
        "exact_runtime_config_and_transitive_hash_validation_required": True,
        "failure_or_missing_dependency_is_terminal": True,
    }
    if dict(reconstruction) != expected_reconstruction:
        raise NoveltyTerminalError(
            "ESDI-288 Gross9 frozen reconstruction contract drift"
        )
    if list(common_domain) != list(GROSS9_DOMAIN):
        raise NoveltyTerminalError("ESDI-288 Gross9 common domain drift")
    return {
        **expected_reconstruction,
        "source_support_passed_before_reconstruction": True,
        "exact_runtime_config_and_transitive_hash_validation_passed": True,
        "five_signed_sleeves_validated": True,
        "gross9_common_domain": list(GROSS9_DOMAIN),
        "evidence_boundary": dict(GROSS9_CLOCK_EVIDENCE_BOUNDARY),
    }


def _gross9_source_support_binding(
    source_support: VerifiedSourceSupport,
) -> dict[str, str]:
    return {
        "path": str(source_support.path),
        "sha256": source_support.sha256,
        "manifest_hash": source_support.manifest_hash,
    }


def parse_gross9_clock_artifact_bytes(
    raw_bytes: bytes,
    *,
    path: str | Path,
    registration: Mapping[str, Any],
    source_support: VerifiedSourceSupport,
    production: bool,
) -> VerifiedGross9Clocks:
    """Validate the write-once Gross9 clock artifact without reconstruction."""

    if not isinstance(source_support, VerifiedSourceSupport):
        raise NoveltyTerminalError(
            "ESDI-288 Gross9 clocks require verified source support"
        )
    verified_support = parse_passed_source_support_bytes(
        source_support.raw_bytes,
        path=source_support.path,
        production=False,
    )
    if verified_support.sha256 != source_support.sha256:
        raise NoveltyTerminalError("ESDI-288 source-support immutable hash drift")
    if not isinstance(raw_bytes, bytes):
        raise NoveltyTerminalError("ESDI-288 Gross9 clock bytes are invalid")
    if production:
        _require_canonical_committed_clean(
            path,
            DEFAULT_GROSS9_CLOCKS_PATH,
            "Gross9 clock artifact",
            raw_bytes,
        )
        artifact_path = DEFAULT_GROSS9_CLOCKS_PATH
    else:
        artifact_path = Path(path)
    payload = _decode_json_bytes(raw_bytes, "Gross9 clock artifact")
    required_keys = {
        "protocol_version",
        "policy_id",
        "preregistration",
        "source_support",
        "authority_hash",
        "clocks",
        "frozen_contract_validation",
        "manifest_hash",
    }
    if set(payload) != required_keys:
        raise NoveltyTerminalError("ESDI-288 Gross9 clock artifact schema drift")
    internal_hash = payload.get("manifest_hash")
    _validate_hash(internal_hash, "Gross9 clock manifest hash")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if internal_hash != canonical_hash(core):
        raise NoveltyTerminalError("ESDI-288 Gross9 clock manifest drift")
    if (
        payload.get("protocol_version") != GROSS9_CLOCKS_PROTOCOL_VERSION
        or payload.get("policy_id") != POLICY_ID
        or payload.get("preregistration") != _expected_preregistration_binding()
        or payload.get("source_support")
        != _gross9_source_support_binding(verified_support)
    ):
        raise NoveltyTerminalError("ESDI-288 Gross9 clock binding drift")
    try:
        authority = registration["gross9"]["authority"]
    except (KeyError, TypeError) as error:
        raise NoveltyTerminalError("ESDI-288 Gross9 authority is missing") from error
    authority_hash = canonical_hash(authority)
    if payload.get("authority_hash") != authority_hash:
        raise NoveltyTerminalError("ESDI-288 Gross9 authority hash drift")
    frozen_validation = gross9_frozen_contract_validation(registration)
    if payload.get("frozen_contract_validation") != frozen_validation:
        raise NoveltyTerminalError(
            "ESDI-288 Gross9 frozen-contract validation drift"
        )
    raw_clocks = payload.get("clocks")
    if not isinstance(raw_clocks, Mapping) or set(raw_clocks) != set(
        GROSS9_SLEEVES
    ):
        raise NoveltyTerminalError(
            "ESDI-288 Gross9 artifact requires exactly five clocks"
        )
    clocks: dict[str, tuple[SignedInterval, ...]] = {}
    for sleeve_name in GROSS9_SLEEVES:
        clock = raw_clocks[sleeve_name]
        if not isinstance(clock, Mapping) or set(clock) != {
            "intervals",
            "sha256",
        }:
            raise NoveltyTerminalError(
                f"ESDI-288 Gross9 sleeve schema drift: {sleeve_name}"
            )
        rows = clock["intervals"]
        if not isinstance(rows, list):
            raise NoveltyTerminalError(
                f"ESDI-288 Gross9 sleeve intervals missing: {sleeve_name}"
            )
        clock_core = {"intervals": rows}
        _validate_hash(clock["sha256"], f"{sleeve_name} sleeve hash")
        if clock["sha256"] != canonical_hash(clock_core):
            raise NoveltyTerminalError(
                f"ESDI-288 Gross9 sleeve hash drift: {sleeve_name}"
            )
        intervals: list[SignedInterval] = []
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {
                "entry",
                "exit",
                "side",
            }:
                raise NoveltyTerminalError(
                    f"ESDI-288 Gross9 interval schema drift: {sleeve_name}"
                )
            intervals.append(
                SignedInterval(
                    _parse_timestamp(row["entry"]),
                    _parse_timestamp(row["exit"]),
                    _parse_side(row["side"]),
                )
            )
        clocks[sleeve_name] = _canonical_intervals(
            intervals,
            f"Gross9:{sleeve_name}",
        )
    validate_gross9_sleeves(clocks)
    if production:
        try:
            from training import (
                evaluate_ethereum_settlement_demand_impulse_economics
                as economics_evaluator,
            )

            if (
                economics_evaluator.GROSS9_CLOCK_PROTOCOL
                != GROSS9_CLOCKS_PROTOCOL_VERSION
                or economics_evaluator.GROSS9_SLEEVES != GROSS9_SLEEVES
            ):
                raise RuntimeError("Gross9 producer identity drift")
            validation = economics_evaluator.validate_frozen_contract(
                registration
            )
            evaluator_identity = (
                economics_evaluator._validate_evaluator_source_identity()
            )
            attempt_payload = (
                economics_evaluator._gross9_attempt_claim_payload(
                    source_support_binding=_gross9_source_support_binding(
                        verified_support
                    ),
                    evaluator_source=evaluator_identity,
                )
            )
            gross9_claim_path = (
                REPOSITORY_ROOT / economics_evaluator.GROSS9_ATTEMPT_CLAIM
            )
            gross9_claim_raw = gross9_claim_path.read_bytes()
            _require_canonical_committed_clean(
                economics_evaluator.GROSS9_ATTEMPT_CLAIM,
                economics_evaluator.GROSS9_ATTEMPT_CLAIM,
                "Gross9 attempt claim",
                gross9_claim_raw,
            )
            economics_evaluator._load_exact_attempt_claim(
                economics_evaluator.GROSS9_ATTEMPT_CLAIM,
                attempt_payload,
                label="Gross9 reconstruction",
            )
            if (
                validation.get("validated") is not True
                or not isinstance(evaluator_identity, Mapping)
                or not isinstance(
                    evaluator_identity.get("manifest_hash"), str
                )
                or len(evaluator_identity["manifest_hash"]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in evaluator_identity["manifest_hash"]
                )
            ):
                raise RuntimeError("Gross9 completion evidence drift")
        except Exception as error:
            raise NoveltyTerminalError(
                "ESDI-288 Gross9 production completion did not authenticate"
            ) from error
    return VerifiedGross9Clocks(
        path=artifact_path,
        raw_bytes=raw_bytes,
        sha256=sha256_bytes(raw_bytes),
        manifest_hash=internal_hash,
        authority_hash=authority_hash,
        clocks=MappingProxyType(clocks),
        payload=_freeze_json(payload),
    )


def load_gross9_clock_artifact(
    *,
    registration: Mapping[str, Any],
    source_support: VerifiedSourceSupport,
    path: str | Path = DEFAULT_GROSS9_CLOCKS_PATH,
    production: bool = True,
) -> VerifiedGross9Clocks:
    if production and Path(path) != DEFAULT_GROSS9_CLOCKS_PATH:
        raise NoveltyTerminalError(
            "ESDI-288 Gross9 clock artifact must use its canonical path"
        )
    resolved = (
        REPOSITORY_ROOT / DEFAULT_GROSS9_CLOCKS_PATH
        if production
        else Path(path)
    )
    try:
        raw_bytes = resolved.read_bytes()
    except OSError as error:
        raise NoveltyTerminalError(
            "ESDI-288 Gross9 clock artifact is unreadable"
        ) from error
    return parse_gross9_clock_artifact_bytes(
        raw_bytes,
        path=path,
        registration=registration,
        source_support=source_support,
        production=production,
    )


def validate_gross9_sleeves(sleeves: Gross9SleeveClocks) -> None:
    if not isinstance(sleeves, Mapping) or set(sleeves) != set(GROSS9_SLEEVES):
        raise NoveltyTerminalError(
            "ESDI-288 requires exactly all five Gross9 signed sleeve clocks"
        )
    if any(prereg.GROSS9_WEIGHTS[name] <= 0 for name in sleeves):
        raise NoveltyTerminalError("ESDI-288 Gross9 sleeve weight is not positive")


def evaluate_gross9_sleeve(
    candidate: Sequence[SignedInterval | tuple[int, int, int]],
    sleeve_name: str,
    sleeve: Sequence[SignedInterval | tuple[int, int, int]],
) -> dict[str, Any]:
    start, end = _domain_seconds(GROSS9_DOMAIN)
    candidate_rows = _intervals_in_domain(candidate, start, end, "candidate")
    sleeve_rows = _intervals_in_domain(sleeve, start, end, sleeve_name)
    candidate_entries = tuple(row.entry for row in candidate_rows)
    sleeve_entries = tuple(row.entry for row in sleeve_rows)
    try:
        candidate_exposure = signed_exposure_5m(
            ((row.entry, row.exit, row.side) for row in candidate_rows), start, end
        )
        sleeve_exposure = signed_exposure_5m(
            ((row.entry, row.exit, row.side) for row in sleeve_rows), start, end
        )
        jaccard = exact_entry_jaccard(candidate_entries, sleeve_entries)
        containment = bidirectional_entry_containment(
            candidate_entries, sleeve_entries, 6 * 60 * 60
        )
        occupied = occupied_bar_jaccard(
            candidate_exposure, sleeve_exposure
        )
        pearson = squared_signed_exposure_pearson(
            candidate_exposure, sleeve_exposure
        )
    except ValueError as error:
        raise NoveltyTerminalError(
            f"ESDI-288 undefined Gross9 metric: {sleeve_name}"
        ) from error
    checks = {
        "exact_entry_jaccard": fraction_at_most(jaccard, 1, 10),
        "candidate_6h_containment": fraction_at_most(
            containment, 7, 20
        ),
        "occupied_bar_jaccard": fraction_at_most(occupied, 1, 4),
        "squared_signed_exposure_pearson": fraction_at_most(
            pearson, 49, 400
        ),
    }
    return {
        "sleeve": sleeve_name,
        "weight": prereg.GROSS9_WEIGHTS[sleeve_name],
        "comparison_domain": list(GROSS9_DOMAIN),
        "candidate_entries": len(candidate_entries),
        "sleeve_entries": len(sleeve_entries),
        "metrics": {
            "exact_entry_jaccard": _fraction_payload(jaccard),
            "candidate_6h_containment": _fraction_payload(containment),
            "occupied_bar_jaccard": _fraction_payload(occupied),
            "squared_signed_exposure_pearson": _pearson_payload(pearson),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def evaluate_novelty(
    candidate: Sequence[SignedInterval | tuple[int, int, int]],
    comparators: Mapping[str, ComparatorClock],
    registry: Mapping[str, Mapping[str, Any]],
    gross9_artifact: VerifiedGross9Clocks,
) -> dict[str, Any]:
    if not isinstance(gross9_artifact, VerifiedGross9Clocks):
        raise NoveltyTerminalError(
            "ESDI-288 novelty requires the verified Gross9 clock artifact"
        )
    gross9_sleeves = gross9_artifact.clocks
    validate_gross9_sleeves(gross9_sleeves)
    expected_ids: set[str] = set()
    domains: dict[str, Sequence[str]] = {}
    for artifact_name, spec in registry.items():
        if spec.get("capability") == "directional_interval":
            groups = spec["groups"] if spec.get("group_column") else [None]
        else:
            groups = [
                *spec["directional_interval_groups"],
                *spec["timestamp_only_groups"],
            ]
        for group in groups:
            comparator_id = (
                artifact_name if group is None else f"{artifact_name}:{group}"
            )
            expected_ids.add(comparator_id)
            domains[comparator_id] = spec["comparison_domain"]
    if set(comparators) != expected_ids:
        raise NoveltyTerminalError(
            "ESDI-288 did not receive every frozen comparator group exactly once"
        )
    prior_results = [
        evaluate_prior_comparator(candidate, comparators[name], domains[name])
        for name in sorted(comparators)
    ]
    gross9_results = [
        evaluate_gross9_sleeve(candidate, name, gross9_sleeves[name])
        for name in GROSS9_SLEEVES
    ]
    passed = all(item["passed"] for item in prior_results + gross9_results)
    failed_checks: list[str] = []
    for category, results in (
        ("prior", prior_results),
        ("gross9", gross9_results),
    ):
        for item in results:
            if category == "prior" and not item["gating"]:
                continue
            identity = item.get("comparator_id", item.get("sleeve"))
            failed_checks.extend(
                f"{category}:{identity}:{check}"
                for check, check_passed in item["checks"].items()
                if not check_passed
            )
    return {
        "prior_source_comparators": prior_results,
        "gross9_sleeves": gross9_results,
        "passed": passed,
        "terminal": not passed,
        "failed_checks": sorted(failed_checks),
    }


def build_report_after_source_support(
    *,
    source_support: VerifiedSourceSupport,
    candidate: Sequence[SignedInterval | tuple[int, int, int]],
    gross9_artifact: VerifiedGross9Clocks,
    comparator_loader: ComparatorLoader = load_comparator_artifacts,
    registration: Mapping[str, Any] | None = None,
    attempt_claim: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Consume immutable support/Gross9 bytes before comparator access."""

    if not isinstance(source_support, VerifiedSourceSupport):
        raise NoveltyTerminalError(
            "ESDI-288 requires an immutable verified source-support artifact"
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
        raise NoveltyTerminalError("ESDI-288 source-support immutable binding drift")
    active_registration = (
        verify_preregistration() if registration is None else registration
    )
    registration_core = {
        key: value
        for key, value in active_registration.items()
        if key != "manifest_hash"
    }
    if (
        active_registration.get("policy_id") != POLICY_ID
        or active_registration.get("manifest_hash")
        != PREREGISTRATION_MANIFEST_HASH
        or canonical_hash(registration_core) != PREREGISTRATION_MANIFEST_HASH
    ):
        raise NoveltyTerminalError("ESDI-288 supplied preregistration drift")
    if not isinstance(gross9_artifact, VerifiedGross9Clocks):
        raise NoveltyTerminalError(
            "ESDI-288 requires the hash-bound Gross9 clock artifact"
        )
    verified_gross9 = parse_gross9_clock_artifact_bytes(
        gross9_artifact.raw_bytes,
        path=gross9_artifact.path,
        registration=active_registration,
        source_support=verified_support,
        production=False,
    )
    if (
        verified_gross9.sha256 != gross9_artifact.sha256
        or verified_gross9.manifest_hash != gross9_artifact.manifest_hash
    ):
        raise NoveltyTerminalError("ESDI-288 Gross9 immutable binding drift")
    registry = frozen_registry(active_registration)
    comparators = comparator_loader(registry)
    novelty = evaluate_novelty(
        candidate,
        comparators,
        registry,
        verified_gross9,
    )
    core: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "preregistration": _expected_preregistration_binding(),
        "attempt_claim": (
            dict(attempt_claim)
            if attempt_claim is not None
            else {"mode": "synthetic_only"}
        ),
        "source_support": {
            "path": str(verified_support.path),
            "sha256": verified_support.sha256,
            "manifest_hash": verified_support.manifest_hash,
            "passed": True,
            "artifact": _thaw_json(verified_support.payload),
        },
        "gross9_clock_artifact": {
            "path": str(verified_gross9.path),
            "sha256": verified_gross9.sha256,
            "manifest_hash": verified_gross9.manifest_hash,
            "authority_hash": verified_gross9.authority_hash,
        },
        "registry_artifacts": len(registry),
        "registry_comparator_groups": len(comparators),
        "novelty": novelty,
        "evidence_boundary": dict(NOVELTY_EVIDENCE_BOUNDARY),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def canonical_report_bytes(payload: Mapping[str, Any]) -> bytes:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise NoveltyTerminalError("ESDI-288 novelty report manifest hash drift")
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


def _write_once_novelty_json(payload: Mapping[str, Any], path: Path) -> str:
    canonical = canonical_report_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise NoveltyTerminalError(
            "ESDI-288 novelty output parent is unsafe"
        )
    if path.is_symlink():
        raise NoveltyTerminalError("ESDI-288 novelty output is unsafe")
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as error:
            raise NoveltyTerminalError(
                "ESDI-288 novelty output is unsafe"
            ) from error
        if existing != canonical:
            raise NoveltyTerminalError("ESDI-288 novelty output drift")
        return "verified_existing"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".staged",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        try:
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(canonical)
                handle.flush()
                os.fsync(handle.fileno())
                os.fchmod(handle.fileno(), 0o444)
            try:
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError:
                if path.is_symlink():
                    raise NoveltyTerminalError(
                        "ESDI-288 novelty output is unsafe"
                    ) from None
                try:
                    existing = path.read_bytes()
                except OSError as error:
                    raise NoveltyTerminalError(
                        "ESDI-288 novelty output is unsafe"
                    ) from error
                if existing != canonical:
                    raise NoveltyTerminalError(
                        "ESDI-288 novelty output drift"
                    )
                return "verified_existing"
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
            return "created"
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_once_novelty_json(
    payload: Mapping[str, Any],
    output: str | Path = DEFAULT_OUTPUT_PATH,
) -> str:
    """Write only the frozen production singleton path."""

    if Path(output) != DEFAULT_OUTPUT_PATH:
        raise NoveltyTerminalError(
            "ESDI-288 novelty must use the canonical output path"
        )
    return _write_once_novelty_json(
        payload,
        REPOSITORY_ROOT / DEFAULT_OUTPUT_PATH,
    )


def write_once_novelty_json_for_test(
    payload: Mapping[str, Any],
    output: str | Path,
) -> str:
    """Explicit synthetic-only writer for isolated temporary paths."""

    return _write_once_novelty_json(payload, Path(output))


def _attempt_claim_payload(
    *,
    source_support: VerifiedSourceSupport,
    gross9_artifact: VerifiedGross9Clocks,
    candidate_clock: Mapping[str, str],
) -> dict[str, Any]:
    core = {
        "protocol_version": ATTEMPT_CLAIM_PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "status": "claimed_before_comparator_access",
        "one_shot": True,
        "retry_or_repair_after_failure": False,
        "preregistration": _expected_preregistration_binding(),
        "source_support": {
            "path": str(source_support.path),
            "sha256": source_support.sha256,
            "manifest_hash": source_support.manifest_hash,
        },
        "gross9_clock_artifact": {
            "path": str(gross9_artifact.path),
            "sha256": gross9_artifact.sha256,
            "manifest_hash": gross9_artifact.manifest_hash,
            "authority_hash": gross9_artifact.authority_hash,
        },
        "candidate_clock": dict(candidate_clock),
        "canonical_output": str(DEFAULT_OUTPUT_PATH),
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


def _claim_binding(
    raw: bytes,
    payload: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "path": str(DEFAULT_ATTEMPT_CLAIM_PATH),
        "sha256": sha256_bytes(raw),
        "claim_hash": str(payload["claim_hash"]),
    }


def _create_attempt_claim(payload: Mapping[str, Any]) -> dict[str, str]:
    path = REPOSITORY_ROOT / DEFAULT_ATTEMPT_CLAIM_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise NoveltyTerminalError("ESDI-288 novelty claim parent is unsafe")
    if path.exists() or path.is_symlink():
        raise NoveltyTerminalError("ESDI-288 novelty attempt is already claimed")
    raw = _claim_bytes(payload)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o444,
    )
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("novelty attempt-claim write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return _claim_binding(raw, payload)


def load_attempt_claim(
    expected_payload: Mapping[str, Any],
) -> dict[str, str]:
    path = REPOSITORY_ROOT / DEFAULT_ATTEMPT_CLAIM_PATH
    if path.is_symlink() or not path.is_file():
        raise NoveltyTerminalError("ESDI-288 novelty attempt claim is invalid")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NoveltyTerminalError(
            "ESDI-288 novelty attempt claim is invalid"
        ) from error
    if (
        not isinstance(payload, Mapping)
        or dict(payload) != dict(expected_payload)
        or raw != _claim_bytes(payload)
    ):
        raise NoveltyTerminalError("ESDI-288 novelty attempt claim drift")
    return _claim_binding(raw, payload)


def _load_completed_novelty(
    *,
    attempt_claim: Mapping[str, Any],
    source_support: VerifiedSourceSupport,
    gross9_artifact: VerifiedGross9Clocks,
) -> dict[str, Any]:
    path = REPOSITORY_ROOT / DEFAULT_OUTPUT_PATH
    if path.is_symlink() or not path.is_file():
        raise NoveltyTerminalError(
            "ESDI-288 claimed novelty attempt lacks a completion artifact"
        )
    raw = path.read_bytes()
    payload = _decode_json_bytes(raw, "novelty completion artifact")
    if raw != canonical_report_bytes(payload):
        raise NoveltyTerminalError(
            "ESDI-288 novelty completion serialization drift"
        )
    if (
        payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("policy_id") != POLICY_ID
        or payload.get("attempt_claim") != dict(attempt_claim)
        or payload.get("source_support", {}).get("sha256")
        != source_support.sha256
        or payload.get("gross9_clock_artifact", {}).get("sha256")
        != gross9_artifact.sha256
    ):
        raise NoveltyTerminalError(
            "ESDI-288 novelty completion binding drift"
        )
    return payload


def load_candidate_clock_csv(
    path: str | Path,
    source_support: VerifiedSourceSupport,
) -> tuple[SignedInterval, ...]:
    """Load the source evaluator's primary clock under its support hash."""

    if not isinstance(source_support, VerifiedSourceSupport):
        raise NoveltyTerminalError(
            "ESDI-288 primary clock requires verified source support"
        )
    verified_support = parse_passed_source_support_bytes(
        source_support.raw_bytes,
        path=source_support.path,
        production=False,
    )
    expected_hash = verified_support.payload["clock_artifacts"]["primary_sha256"]
    _validate_hash(expected_hash, "primary clock hash")
    candidate = Path(path)
    try:
        compressed = candidate.read_bytes()
    except OSError as error:
        raise NoveltyTerminalError("ESDI-288 primary clock is missing") from error
    if sha256_bytes(compressed) != expected_hash:
        raise NoveltyTerminalError("ESDI-288 primary clock hash drift")
    try:
        text = gzip.decompress(compressed).decode("utf-8")
    except (OSError, EOFError, UnicodeDecodeError) as error:
        raise NoveltyTerminalError("ESDI-288 primary clock gzip is invalid") from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    required = {
        "policy_id",
        "control",
        "entry_time_utc",
        "exit_time_utc",
        "side",
    }
    if (
        reader.fieldnames is None
        or len(reader.fieldnames) != len(set(reader.fieldnames))
        or not required.issubset(reader.fieldnames)
    ):
        raise NoveltyTerminalError("ESDI-288 primary clock columns are missing")
    intervals: list[SignedInterval] = []
    try:
        for row in reader:
            if (
                None in row
                or row["policy_id"] != POLICY_ID
                or row["control"] != "primary"
            ):
                raise NoveltyTerminalError("ESDI-288 primary clock row drift")
            intervals.append(
                SignedInterval(
                    _parse_timestamp(row["entry_time_utc"]),
                    _parse_timestamp(row["exit_time_utc"]),
                    _parse_side(row["side"]),
                )
            )
    except csv.Error as error:
        raise NoveltyTerminalError("ESDI-288 primary clock CSV is invalid") from error
    return _canonical_intervals(intervals, "ESDI-288 primary clock")


def load_reproduced_novelty_for_economics(
    path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """Authenticate and independently reproduce the committed novelty report."""

    if Path(path) != DEFAULT_OUTPUT_PATH:
        raise NoveltyTerminalError(
            "ESDI-288 novelty completion path is canonical"
        )
    output_path = REPOSITORY_ROOT / DEFAULT_OUTPUT_PATH
    try:
        raw = output_path.read_bytes()
    except OSError as error:
        raise NoveltyTerminalError(
            "ESDI-288 novelty completion artifact is unreadable"
        ) from error
    _require_canonical_committed_clean(
        DEFAULT_OUTPUT_PATH,
        DEFAULT_OUTPUT_PATH,
        "novelty completion artifact",
        raw,
    )
    registration = verify_preregistration()
    source_support = load_passed_source_support(production=True)
    candidate_relative = (
        source_support_evaluator.DEFAULT_PRIMARY_CLOCK_OUTPUT
    )
    candidate_path = REPOSITORY_ROOT / candidate_relative
    try:
        candidate_raw = candidate_path.read_bytes()
    except OSError as error:
        raise NoveltyTerminalError(
            "ESDI-288 candidate clock is unreadable"
        ) from error
    _require_canonical_committed_clean(
        candidate_relative,
        candidate_relative,
        "candidate clock artifact",
        candidate_raw,
    )
    candidate = load_candidate_clock_csv(
        candidate_path,
        source_support,
    )
    gross9_artifact = load_gross9_clock_artifact(
        registration=registration,
        source_support=source_support,
        path=DEFAULT_GROSS9_CLOCKS_PATH,
        production=True,
    )
    candidate_binding = {
        "path": str(candidate_relative),
        "sha256": sha256_bytes(candidate_raw),
    }
    attempt_payload = _attempt_claim_payload(
        source_support=source_support,
        gross9_artifact=gross9_artifact,
        candidate_clock=candidate_binding,
    )
    claim_path = REPOSITORY_ROOT / DEFAULT_ATTEMPT_CLAIM_PATH
    try:
        claim_raw = claim_path.read_bytes()
    except OSError as error:
        raise NoveltyTerminalError(
            "ESDI-288 novelty attempt claim is unreadable"
        ) from error
    _require_canonical_committed_clean(
        DEFAULT_ATTEMPT_CLAIM_PATH,
        DEFAULT_ATTEMPT_CLAIM_PATH,
        "novelty attempt claim",
        claim_raw,
    )
    attempt_claim = load_attempt_claim(attempt_payload)
    observed = _decode_json_bytes(raw, "novelty completion artifact")
    if raw != canonical_report_bytes(observed):
        raise NoveltyTerminalError(
            "ESDI-288 novelty completion serialization drift"
        )
    expected = build_report_after_source_support(
        source_support=source_support,
        candidate=candidate,
        gross9_artifact=gross9_artifact,
        registration=registration,
        attempt_claim=attempt_claim,
    )
    if observed != expected:
        raise NoveltyTerminalError(
            "ESDI-288 committed novelty report did not reproduce exactly"
        )
    return observed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-support",
        type=Path,
        default=DEFAULT_SOURCE_SUPPORT_PATH,
    )
    parser.add_argument("--candidate-clock", type=Path, required=True)
    parser.add_argument(
        "--gross9-clocks",
        type=Path,
        default=DEFAULT_GROSS9_CLOCKS_PATH,
    )
    arguments = parser.parse_args()
    if arguments.candidate_clock != (
        source_support_evaluator.DEFAULT_PRIMARY_CLOCK_OUTPUT
    ):
        raise NoveltyTerminalError(
            "ESDI-288 production candidate clock path is canonical"
        )
    claim_path = REPOSITORY_ROOT / DEFAULT_ATTEMPT_CLAIM_PATH
    output_path = REPOSITORY_ROOT / DEFAULT_OUTPUT_PATH
    claim_exists = claim_path.exists() or claim_path.is_symlink()
    output_exists = output_path.exists() or output_path.is_symlink()
    if claim_exists and not output_exists:
        raise NoveltyTerminalError(
            "ESDI-288 novelty attempt was claimed without completion"
        )
    if output_exists and not claim_exists:
        raise NoveltyTerminalError(
            "ESDI-288 novelty output exists without its attempt claim"
        )
    registration = verify_preregistration()
    source_support = load_passed_source_support(
        arguments.source_support,
        production=True,
    )
    candidate = load_candidate_clock_csv(
        arguments.candidate_clock,
        source_support,
    )
    gross9_artifact = load_gross9_clock_artifact(
        registration=registration,
        source_support=source_support,
        path=arguments.gross9_clocks,
        production=True,
    )
    candidate_path = REPOSITORY_ROOT / arguments.candidate_clock
    candidate_binding = {
        "path": str(arguments.candidate_clock),
        "sha256": sha256_bytes(candidate_path.read_bytes()),
    }
    attempt_payload = _attempt_claim_payload(
        source_support=source_support,
        gross9_artifact=gross9_artifact,
        candidate_clock=candidate_binding,
    )
    if claim_exists:
        attempt_claim = load_attempt_claim(attempt_payload)
        report = _load_completed_novelty(
            attempt_claim=attempt_claim,
            source_support=source_support,
            gross9_artifact=gross9_artifact,
        )
        status = "verified_existing"
    else:
        attempt_claim = _create_attempt_claim(attempt_payload)
        report = build_report_after_source_support(
            source_support=source_support,
            candidate=candidate,
            gross9_artifact=gross9_artifact,
            registration=registration,
            attempt_claim=attempt_claim,
        )
        status = write_once_novelty_json(report)
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


if __name__ == "__main__":
    main()
