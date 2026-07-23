"""Seal URCD-72 before source values, comparator rows, or outcomes are opened."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


POLICY_ID = "URCD-72"
PROTOCOL_VERSION = "usdc_recipient_concentration_dislocation_prereg_v1"
AS_OF_DATE = "2026-07-23"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path("training/preregister_usdc_recipient_concentration_dislocation.py")
TEST_PATH = Path("tests/test_preregister_usdc_recipient_concentration_dislocation.py")
BOUNDARY_DOCUMENT = Path(
    "docs/usdc-recipient-concentration-dislocation-boundary-2026-07-23.md"
)
BOUNDARY_DOCUMENT_SHA256 = (
    "baf461fd03c39f80cb02e14d61bfaf7104cc59b087013e3d2cbf6a9582ac0707"
)
MECHANISM_DOCUMENT = Path(
    "docs/usdc-recipient-concentration-dislocation-mechanism-decision-2026-07-23.md"
)
MECHANISM_DOCUMENT_SHA256 = (
    "8acbbf0da8343a5fa8ee23d7faeeb63fffe5ba7ebe8cb0baeb5f768060d1a02c"
)
DEFAULT_OUTPUT = Path(
    "results/usdc_recipient_concentration_dislocation_preregistration_2026-07-23.json"
)

SOURCE_CSV = Path(
    "data/ethereum_stablecoin_issuance_redemption_2020_2023/"
    "ethereum_usdt_usdc_issuance_redemption_2020_2023.csv.gz"
)
SOURCE_CSV_SHA256 = "70ba3799ba84dc671051623a8d167b1731f043cf84a686b9878a67fcd52e5901"
SOURCE_MANIFEST = Path(
    "results/ethereum_stablecoin_issuance_redemption_source_manifest_2026-07-21.json"
)
SOURCE_MANIFEST_SHA256 = (
    "8ec9ab08c413bf6f5f8170fb800b05105522d4cf1a7932943c214288701e31fe"
)
SOURCE_MANIFEST_HASH = (
    "a0c7740db64f7779fade68d76985c629cabe81983bf594e8258cef16a5725a1b"
)
SOURCE_HEADER = (
    "asset",
    "contract_address",
    "event",
    "event_sign",
    "amount_raw",
    "decimals",
    "indexed_address_1",
    "indexed_address_2",
    "data_address",
    "block_number",
    "block_hash",
    "block_timestamp",
    "transaction_hash",
    "transaction_index",
    "log_index",
    "confirmation_block_number",
    "confirmation_block_hash",
    "available_at",
)
SOURCE_ALLOWED_COLUMNS = (
    "asset",
    "event",
    "event_sign",
    "amount_raw",
    "decimals",
    "indexed_address_2",
    "block_number",
    "block_hash",
    "block_timestamp",
    "transaction_hash",
    "transaction_index",
    "log_index",
    "available_at",
)


def _comparator(
    *,
    candidate: str,
    controls: Sequence[str],
    path: str,
    sha256: str,
    header_line_sha256: str,
    start: str,
    end: str,
) -> Mapping[str, Any]:
    return {
        "candidate": candidate,
        "controls": tuple(controls),
        "path": Path(path),
        "sha256": sha256,
        "header_line_sha256": header_line_sha256,
        "comparison_start": start,
        "comparison_end_exclusive": end,
    }


FULL_START = "2021-01-01T00:00:00Z"
FULL_END = "2024-01-01T00:00:00Z"
LATE_START = "2023-09-01T00:00:00Z"
COMPARATOR_SPECS: tuple[Mapping[str, Any], ...] = (
    _comparator(
        candidate="AMTR-48",
        controls=("primary", "cross_minter"),
        path="data/authorized_minter_turnaround_relay_clocks_2020_2023.csv.gz",
        sha256="30875029daa4d6e2eff9a59f53d45eda57dbced05988df089c38a6c81abfa0f6",
        header_line_sha256="423287fbc7a50bd00c0ca1de8580c983df1a2d128c1cc497d68e1bc74c224ac8",
        start=FULL_START,
        end=FULL_END,
    ),
    _comparator(
        candidate="UGCI-288",
        controls=("primary",),
        path="data/usdc_gross_clearing_imbalance_clocks_2021_2023.csv.gz",
        sha256="a0f861c69ac171e1efa665dc90a916d0351413ca07e5e46783bb8abd662175fd",
        header_line_sha256="b79639e44ce1b4488fdf6991e60831221cbc9a48565fa42d053faeb71156ad91",
        start=FULL_START,
        end=FULL_END,
    ),
    _comparator(
        candidate="WCDR-2016",
        controls=("primary",),
        path=(
            "data/wrapped_collateral_dollar_liquidity_rotation_2021_2023/"
            "wcdr2016_support_clocks_2021_2023.csv.gz"
        ),
        sha256="241d96a64a654ba2faeda2d4a8460131269acf21d0bbbf31177d35d1ecd63b3c",
        header_line_sha256="e67cd52d0cadded15fd49f4ed809707e5d1601260416a93949f452dd7638680e",
        start=FULL_START,
        end=FULL_END,
    ),
    _comparator(
        candidate="WTSL-168-SOURCE-SEEN",
        controls=("primary",),
        path=(
            "data/wbtc_turnover_stablecoin_liquidity_2021_2023/"
            "wtsl168_support_clocks_2021_2023.csv.gz"
        ),
        sha256="df8cb085d439c9ee9e89334cb891b9e3b04f54c2a8e70bd4f552a90648ea8b6d",
        header_line_sha256="f206f15f5410c3bb568df4f64c0cffafcf077b5ef08dc8c427ac3af33d873937",
        start=FULL_START,
        end=FULL_END,
    ),
    _comparator(
        candidate="WSCF-72-SOURCE-FAMILY-SEEN",
        controls=("primary",),
        path=(
            "data/wbtc_stablecoin_finalized_confirmation_relay_2021_2023/"
            "wscf72_support_clocks_2021_2023.csv.gz"
        ),
        sha256="86565774ae97a1024c5a66b4d59a1f5413bf4608398623359dd3ee24572f0ef3",
        header_line_sha256="adb55cd822efbdcd8469018a51c2b037514758633599a403fae1a1868ef2e9f3",
        start=FULL_START,
        end=FULL_END,
    ),
    _comparator(
        candidate="FCCM-72",
        controls=("primary",),
        path=(
            "data/funding_currency_custody_mobility_consensus_2021_2023/"
            "fccm72_support_clocks_2021_2023.csv.gz"
        ),
        sha256="71180862d9dcc4d76e055c52fd72a2424ee12387a6b8062af8a9382675af3810",
        header_line_sha256="ffec7a169e71d896d348e875e4753c880050c8011b52eb058eee6932a5d4a6d5",
        start=FULL_START,
        end=FULL_END,
    ),
    _comparator(
        candidate="SQFD-6",
        controls=("primary", "no_usdt_lag", "no_participation"),
        path="data/stablecoin_quote_flow_diffusion_clocks_2023_2026.csv.gz",
        sha256="a81e144eea1e80ae5439fc66db1fad5bbd00cd9ac177e25142b5cfb5a07bcc5b",
        header_line_sha256="2e6d34c734ddc66d15c7718cc0aed3f2c8903fc02370bd9a2446054ff96a2071",
        start=LATE_START,
        end=FULL_END,
    ),
    _comparator(
        candidate="SDDR-12",
        controls=("primary",),
        path="data/stablecoin_denominator_dislocation_clocks_2023.csv.gz",
        sha256="eaf2d6c187af9855e76474d2951fcdc12267174980a72649b73d068982ca8c69",
        header_line_sha256="91e4b4187dccbba5c9a6407316c4205d17422b1900b319a7ef800a541e1f3550",
        start=LATE_START,
        end=FULL_END,
    ),
    _comparator(
        candidate="UCBR-12",
        controls=("primary",),
        path="data/usdt_collateral_breadth_relay_clocks_2023.csv.gz",
        sha256="20b3ee9f82696222a3adbde0045dfde53e0e240e85162e463166aa8fe90b1a8f",
        header_line_sha256="a66cd7a33793d7d0b1056171526dd67c9de5cb95b8847435a8ad1c220757ef10",
        start=LATE_START,
        end=FULL_END,
    ),
)


@dataclass(frozen=True)
class PolicyConfig:
    anchor_hours_utc: tuple[int, ...] = (0, 6, 12, 18)
    current_window_hours: int = 24
    minimum_current_events: int = 4
    minimum_current_recipients: int = 3
    prior_daily_endpoints: int = 180
    minimum_valid_prior_windows: int = 120
    diffuse_quantile: str = "1/5"
    concentrated_quantile: str = "4/5"
    materiality_quantile: str = "1/2"
    entry_delay_minutes: int = 10
    hold_elapsed_hours: int = 72
    bar_minutes: int = 5
    train_start: str = FULL_START
    train_end_exclusive: str = "2023-01-01T00:00:00Z"
    selection_start: str = "2023-01-01T00:00:00Z"
    selection_end_exclusive: str = FULL_END


FROZEN_CONFIG = PolicyConfig()

SOURCE_CONTROLS = (
    "primary",
    "event_count_hhi",
    "equal_recipient_breadth",
    "no_materiality",
    "stale_24h",
    "recipient_year_permutation",
    "amount_year_permutation",
    "direction_flip",
)

SUPPORT_GATES: Mapping[str, Any] = {
    "train_total_minimum": 80,
    "each_train_year_minimum": 30,
    "each_train_half_minimum": 12,
    "train_each_side_minimum": 16,
    "selection_total_minimum": 30,
    "each_selection_half_minimum": 10,
    "selection_each_side_minimum": 6,
    "minimum_each_side_share": "1/5",
    "maximum_month_share": "1/5",
    "maximum_quarter_share": "2/5",
    "maximum_gap_days": 60,
    "maximum_same_side_run": 12,
    "permutation_maximum_exact_jaccard": "7/20",
    "permutation_maximum_exact_same_side_reproduction": "3/5",
    "failure_action": "retire_URCD_72_before_comparators_and_outcomes",
}

NOVELTY_GATES: Mapping[str, Any] = {
    "maximum_exact_entry_jaccard": "1/10",
    "tolerance_elapsed_hours": 6,
    "maximum_bidirectional_containment": "2/5",
    "minimum_candidate_entries": 10,
    "minimum_comparator_entries": 5,
    "timestamp_only": True,
    "side_is_integrity_only": True,
}

FORBIDDEN_COMPARATOR_HEADER_TOKENS = (
    "return",
    "price",
    "funding",
    "future",
    "label",
    "pnl",
    "cagr",
    "mdd",
)

STATIC_BOUNDARY: Mapping[str, Any] = {
    "boundary_and_mechanism_bytes_hashed": 0,
    "source_file_bytes_hashed": 0,
    "source_manifest_metadata_parsed": 0,
    "source_header_lines_decoded": 0,
    "source_value_rows_decoded": 0,
    "comparator_file_bytes_hashed": 0,
    "comparator_header_lines_decoded": 0,
    "comparator_value_rows_decoded": 0,
    "urcd_features_or_incidence_computed": 0,
    "btc_market_rows_decoded": 0,
    "funding_rows_decoded": 0,
    "future_return_rows_decoded": 0,
    "pnl_cagr_mdd_values_decoded": 0,
    "post_2023_source_value_rows_decoded": 0,
    "network_calls": 0,
    "git_protocol_subprocess_calls": 0,
}
VERIFIED_UNCOMMITTED_BOUNDARY: Mapping[str, Any] = {
    **STATIC_BOUNDARY,
    "boundary_and_mechanism_bytes_hashed": 2,
    "source_file_bytes_hashed": 1,
    "source_manifest_metadata_parsed": 1,
    "source_header_lines_decoded": 1,
    "comparator_file_bytes_hashed": len(COMPARATOR_SPECS),
    "comparator_header_lines_decoded": len(COMPARATOR_SPECS),
}
EXPECTED_BOUNDARY: Mapping[str, Any] = {
    **VERIFIED_UNCOMMITTED_BOUNDARY,
    "git_protocol_subprocess_calls": 2,
}


@dataclass(frozen=True)
class Config:
    output: str = str(DEFAULT_OUTPUT)


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeError("URCD path must be repository-relative")
    resolved = (REPOSITORY_ROOT / candidate).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("URCD path must remain repository-relative") from exc
    return resolved


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def serialized_payload(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _read_gzip_header_line(path: str | Path) -> tuple[bytes, tuple[str, ...]]:
    with gzip.open(_repository_path(path), "rb") as handle:
        line = handle.readline()
    if not line.endswith(b"\n"):
        raise RuntimeError("URCD gzip comparator/source header lacks LF")
    header = tuple(next(csv.reader([line.decode("utf-8")])))
    return line, header


def _validate_hash(path: str | Path, expected: str, label: str) -> None:
    if sha256_file(path) != expected:
        raise RuntimeError(f"URCD {label} hash mismatch")


def _validate_config(cfg: PolicyConfig = FROZEN_CONFIG) -> None:
    if cfg != FROZEN_CONFIG:
        raise RuntimeError("URCD frozen policy drift")
    if cfg.anchor_hours_utc != (0, 6, 12, 18):
        raise RuntimeError("URCD anchor grid drift")
    if cfg.hold_elapsed_hours * 60 % cfg.bar_minutes:
        raise RuntimeError("URCD hold does not align to five-minute bars")
    if cfg.entry_delay_minutes != 2 * cfg.bar_minutes:
        raise RuntimeError("URCD entry latency drift")


def _verify_source_binding() -> Mapping[str, Any]:
    for path, digest, label in (
        (BOUNDARY_DOCUMENT, BOUNDARY_DOCUMENT_SHA256, "boundary document"),
        (MECHANISM_DOCUMENT, MECHANISM_DOCUMENT_SHA256, "mechanism document"),
        (SOURCE_CSV, SOURCE_CSV_SHA256, "source CSV"),
        (SOURCE_MANIFEST, SOURCE_MANIFEST_SHA256, "source manifest"),
    ):
        _validate_hash(path, digest, label)
    _, header = _read_gzip_header_line(SOURCE_CSV)
    if header != SOURCE_HEADER:
        raise RuntimeError("URCD source header drift")
    manifest = json.loads(_repository_path(SOURCE_MANIFEST).read_text("utf-8"))
    output = manifest.get("output", {})
    boundary = manifest.get("outcome_boundary", {})
    if (
        manifest.get("manifest_hash") != SOURCE_MANIFEST_HASH
        or output.get("path") != str(SOURCE_CSV)
        or output.get("sha256") != SOURCE_CSV_SHA256
        or output.get("rows") != 266_362
        or boundary.get("source_only") is not True
        or boundary.get("pnl_cagr_mdd_opened") is not False
    ):
        raise RuntimeError("URCD source manifest contract drift")
    for field in (
        "btc_market_rows_read",
        "funding_rows_read",
        "future_return_rows_read",
        "post_2023_contract_event_rows_read",
    ):
        if boundary.get(field) != 0:
            raise RuntimeError(f"URCD source manifest opened {field}")
    return {
        "csv": str(SOURCE_CSV),
        "csv_sha256": SOURCE_CSV_SHA256,
        "manifest": str(SOURCE_MANIFEST),
        "manifest_sha256": SOURCE_MANIFEST_SHA256,
        "manifest_hash": SOURCE_MANIFEST_HASH,
        "header": list(SOURCE_HEADER),
        "allowed_columns": list(SOURCE_ALLOWED_COLUMNS),
        "manifest_rows": 266_362,
        "value_rows_read_during_preregistration": 0,
    }


def _source_binding_static() -> Mapping[str, Any]:
    return {
        "csv": str(SOURCE_CSV),
        "csv_sha256": SOURCE_CSV_SHA256,
        "manifest": str(SOURCE_MANIFEST),
        "manifest_sha256": SOURCE_MANIFEST_SHA256,
        "manifest_hash": SOURCE_MANIFEST_HASH,
        "header": list(SOURCE_HEADER),
        "allowed_columns": list(SOURCE_ALLOWED_COLUMNS),
        "manifest_rows": 266_362,
        "value_rows_read_during_preregistration": 0,
    }


def _comparator_bindings(*, verify: bool) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    required = {"candidate", "control", "entry_time", "side"}
    for spec in COMPARATOR_SPECS:
        if verify:
            _validate_hash(spec["path"], spec["sha256"], spec["candidate"])
            line, header = _read_gzip_header_line(spec["path"])
            if hashlib.sha256(line).hexdigest() != spec["header_line_sha256"]:
                raise RuntimeError(f"URCD comparator header drift: {spec['candidate']}")
            if not required.issubset(header) or len(header) != len(set(header)):
                raise RuntimeError(f"URCD comparator schema drift: {spec['candidate']}")
            lowered = tuple(field.lower() for field in header)
            if any(
                token in field
                for field in lowered
                for token in FORBIDDEN_COMPARATOR_HEADER_TOKENS
            ):
                raise RuntimeError(
                    f"URCD comparator outcome field forbidden: {spec['candidate']}"
                )
        output.append(
            {
                "candidate": spec["candidate"],
                "controls": list(spec["controls"]),
                "path": str(spec["path"]),
                "sha256": spec["sha256"],
                "header_line_sha256": spec["header_line_sha256"],
                "comparison_start": spec["comparison_start"],
                "comparison_end_exclusive": spec["comparison_end_exclusive"],
                "allowed_columns": ["candidate", "control", "entry_time", "side"],
                "value_rows_read_during_preregistration": 0,
            }
        )
    return output


def policy_payload() -> dict[str, Any]:
    _validate_config()
    return {
        "candidate": POLICY_ID,
        "singleton": True,
        "mutable_parameters": [],
        "config": asdict(FROZEN_CONFIG),
        "source": {
            "asset": "usdc_eth",
            "event": "mint",
            "event_sign": 1,
            "decimals": 6,
            "clock": "available_at",
            "sealed_from": FULL_END,
            "post_seal_values": "timestamp pre-screened and not decoded",
            "recipient": "lowercase indexed_address_2 operational endpoint",
            "full_period_recipient_membership_feature_forbidden": True,
        },
        "current_state": {
            "anchors_utc": list(FROZEN_CONFIG.anchor_hours_utc),
            "window": "D-24h < available_at <= D",
            "minimum_events": FROZEN_CONFIG.minimum_current_events,
            "minimum_distinct_recipients": FROZEN_CONFIG.minimum_current_recipients,
            "amount_aggregation": "sum integer amount_raw by recipient",
            "hhi": "sum(a_r*a_r)/(sum(a_r)*sum(a_r)) exact rational",
            "binary_float_forbidden": True,
        },
        "prior_panel": {
            "endpoints": "D-1d through D-180d inclusive at same UTC hour",
            "daily_endpoints": FROZEN_CONFIG.prior_daily_endpoints,
            "minimum_valid": FROZEN_CONFIG.minimum_valid_prior_windows,
            "reference_windows_nonoverlapping": True,
            "invalid_windows_excluded_but_counted": True,
            "q20_hhi": "nearest-rank ascending exact rational",
            "q80_hhi": "nearest-rank ascending exact rational",
            "q50_amount": "independently sorted ascending integer amount",
            "ties": "earlier reference endpoint",
            "current_or_same_later_excluded": True,
        },
        "state_transition": {
            "diffuse": "valid and material and HHI<=q20_hhi",
            "concentrated": "valid and material and HHI>=q80_hhi",
            "equal_hhi_thresholds": "neutral",
            "material": "current total mint amount>=strict-prior q50 amount",
            "long": "enter DIFFUSE from immediately prior non-DIFFUSE",
            "short": "enter CONCENTRATED from immediately prior non-CONCENTRATED",
            "invalid_prior_anchor": "neutral; no carry",
        },
        "execution": {
            "entry": "D+10 elapsed minutes",
            "exit": "entry+72 elapsed hours",
            "hold_bars_5m": 864,
            "split_reservation": "independent per split and control",
            "split_filter": "split_start<=entry and exit<=split_end before reservation",
            "crossing_candidate_advances_reservation": False,
            "acceptance": "entry>=prior accepted exit",
            "stop_takeprofit_pyramiding": False,
        },
        "identity": {
            "algorithm": "SHA256 canonical UTF-8 JSON",
            "json": "sort_keys=True,separators=(',',':'),ensure_ascii=True,no newline",
            "fields": [
                "candidate",
                "control",
                "decision_time",
                "row_identities",
                "side",
            ],
            "row_identity_sort": "(block_hash,transaction_hash,integer(log_index))",
        },
        "controls": {
            "order": list(SOURCE_CONTROLS),
            "audit_only": [
                "event_count_hhi",
                "equal_recipient_breadth",
                "no_materiality",
                "stale_24h",
            ],
            "routing_selectivity": [
                "recipient_year_permutation",
                "amount_year_permutation",
            ],
            "later_economic": ["direction_flip"],
            "permutation": (
                "within available_at UTC year; independent SHA256 source/destination "
                "ordering; selected field only; no RNG or tunable seed"
            ),
        },
        "support_gates": dict(SUPPORT_GATES),
        "novelty_gates": dict(NOVELTY_GATES),
        "windows": {
            "train": [
                FROZEN_CONFIG.train_start,
                FROZEN_CONFIG.train_end_exclusive,
            ],
            "selection": [
                FROZEN_CONFIG.selection_start,
                FROZEN_CONFIG.selection_end_exclusive,
            ],
        },
        "oos_extension": {
            "current_scope_ends_pre2024": True,
            "same_contract_topic_abi_and_n_plus_64": True,
            "dual_archive_replay_and_independent_headers": True,
            "contiguous_from": FULL_END,
            "each_stage_source_frozen_before_corresponding_outcomes": True,
        },
        "rllm_boundary": {
            "deterministic_economics_must_pass_first": True,
            "allowed_actions": ["TRADE_FIXED_SIDE", "ABSTAIN"],
            "may_create_retime_reverse_or_repair": False,
        },
    }


def _build_preregistration(
    *, verify_bindings: bool, artifact_eligible: bool = False, git_calls: int = 0
) -> dict[str, Any]:
    policy = policy_payload()
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": POLICY_ID,
        "as_of_date": AS_OF_DATE,
        "verification_mode": (
            "verified_hashes_headers_and_commit_guard"
            if artifact_eligible
            else (
                "verified_hashes_and_headers_uncommitted"
                if verify_bindings
                else "static_test_fixture"
            )
        ),
        "artifact_eligible": artifact_eligible,
        "boundary_document": {
            "path": str(BOUNDARY_DOCUMENT),
            "sha256": BOUNDARY_DOCUMENT_SHA256,
        },
        "mechanism_document": {
            "path": str(MECHANISM_DOCUMENT),
            "sha256": MECHANISM_DOCUMENT_SHA256,
        },
        "source_binding": (
            _verify_source_binding() if verify_bindings else _source_binding_static()
        ),
        "comparator_bindings": _comparator_bindings(verify=verify_bindings),
        "policy": policy,
        "policy_hash": canonical_hash(policy),
        "outcome_boundary": dict(
            EXPECTED_BOUNDARY
            if artifact_eligible
            else (
                VERIFIED_UNCOMMITTED_BOUNDARY if verify_bindings else STATIC_BOUNDARY
            )
        ),
        "source_values_or_incidence_opened": False,
        "comparator_rows_opened_during_preregistration": False,
        "outcomes_opened": False,
        "performance_values_opened": False,
        "failure_action": "retire_URCD_72_without_repair",
        "artifact_output": str(DEFAULT_OUTPUT),
        "preregistration_source": str(SCRIPT_PATH),
        "preregistration_source_sha256": sha256_file(SCRIPT_PATH),
        "preregistration_test": str(TEST_PATH),
        "preregistration_test_sha256": sha256_file(TEST_PATH),
        "git_protocol_subprocess_calls": git_calls,
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def build_preregistration(*, verify_bindings: bool = True) -> dict[str, Any]:
    return _build_preregistration(verify_bindings=verify_bindings)


def _validate_preregistration(
    payload: Mapping[str, Any], *, verify_bindings: bool, allow_eligible: bool
) -> None:
    if payload.get("candidate") != POLICY_ID:
        raise RuntimeError("URCD preregistration candidate drift")
    if payload.get("artifact_eligible") and not allow_eligible:
        raise RuntimeError("URCD eligible artifact is validated only by the write path")
    eligible = bool(payload.get("artifact_eligible"))
    expected_git_calls = 2 if eligible else 0
    if payload.get("git_protocol_subprocess_calls") != expected_git_calls:
        raise RuntimeError("URCD git protocol call counter drift")
    if payload.get("artifact_output") != str(DEFAULT_OUTPUT):
        raise RuntimeError("URCD artifact output binding drift")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("URCD preregistration manifest hash mismatch")
    expected = _build_preregistration(
        verify_bindings=verify_bindings,
        artifact_eligible=eligible,
        git_calls=expected_git_calls,
    )
    if payload != expected:
        raise RuntimeError("URCD preregistration binding drift")
    boundary = payload.get("outcome_boundary", {})
    for field in (
        "source_value_rows_decoded",
        "comparator_value_rows_decoded",
        "urcd_features_or_incidence_computed",
        "btc_market_rows_decoded",
        "funding_rows_decoded",
        "future_return_rows_decoded",
        "pnl_cagr_mdd_values_decoded",
        "post_2023_source_value_rows_decoded",
        "network_calls",
    ):
        if boundary.get(field) != 0:
            raise RuntimeError(f"URCD preregistration boundary opened: {field}")


def validate_preregistration(
    payload: Mapping[str, Any], *, verify_bindings: bool = True
) -> None:
    _validate_preregistration(
        payload, verify_bindings=verify_bindings, allow_eligible=False
    )


def _git_check(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_protocol_committed() -> None:
    paths = (str(SCRIPT_PATH), str(TEST_PATH))
    tracked = _git_check("ls-files", "--error-unmatch", "--", *paths)
    if tracked.returncode:
        raise RuntimeError("URCD preregistration protocol is not committed")
    clean = _git_check("diff", "--quiet", "HEAD", "--", *paths)
    if clean.returncode:
        raise RuntimeError("URCD preregistration protocol differs from HEAD")


def write_preregistration(cfg: Config = Config()) -> tuple[dict[str, Any], str]:
    if Path(cfg.output) != DEFAULT_OUTPUT:
        raise RuntimeError("URCD eligible artifact output must equal DEFAULT_OUTPUT")
    _assert_protocol_committed()
    payload = _build_preregistration(
        verify_bindings=True, artifact_eligible=True, git_calls=2
    )
    _validate_preregistration(payload, verify_bindings=True, allow_eligible=True)
    raw = serialized_payload(payload)
    destination = _repository_path(cfg.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != raw:
            raise RuntimeError("URCD preregistration artifact is immutable")
        return payload, "verified_existing"
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, destination)
    except FileExistsError:
        if destination.read_bytes() != raw:
            raise RuntimeError("URCD preregistration artifact raced with different bytes")
        return payload, "verified_existing"
    finally:
        temporary.unlink(missing_ok=True)
    return payload, "created"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    payload, status = write_preregistration(Config(output=args.output))
    print(json.dumps({"status": status, **payload}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
