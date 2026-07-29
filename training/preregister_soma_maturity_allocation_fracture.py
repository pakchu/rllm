"""Freeze SMAF-72 before source incidence, overlap, or market outcomes."""

from __future__ import annotations

import argparse
from datetime import date
import errno
from fractions import Fraction
import gzip
import hashlib
import json
import os
import re
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
import secrets
from typing import Any, Mapping


POLICY_ID = "SMAF-72"
PROTOCOL_VERSION = "soma_maturity_allocation_fracture_preregistration_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path("training/preregister_soma_maturity_allocation_fracture.py")
DEFAULT_OUTPUT = Path(
    "results/soma_maturity_allocation_fracture_"
    "preregistration_2026-07-30.json"
)

PREREGISTRATION_DOCUMENT = Path(
    "docs/soma-maturity-allocation-fracture-"
    "preregistration-2026-07-30.md"
)
PREREGISTRATION_DOCUMENT_SHA256 = (
    "0ca0b00c77bd55e3360abe1f36409938a8e95dc450a599b89370d1265b4491f9"
)
SOURCE_AUDIT = Path(
    "docs/new-york-fed-securities-lending-source-audit-2026-07-23.md"
)
SOURCE_AUDIT_SHA256 = (
    "c812998be0bd44efc09b8120d9bc0b2a96f4e1e95f9414a4c8458d97319307bc"
)
COMMON_WINDOW_POLICY = Path(
    "docs/novelty-comparator-common-window-policy-2026-07-23.md"
)
COMMON_WINDOW_POLICY_SHA256 = (
    "928bce6e04fb34001478b4b4ea84156580b661c88a0f0338065a891c009bd580"
)
SOURCE_BUILDER = Path("training/build_new_york_fed_securities_lending.py")
SOURCE_BUILDER_SHA256 = (
    "2f0b5b3daca253ca015c7f691faf0ab75d11c200c11f5bc1c47b34ed1b85ef45"
)

SOURCE_ROOT = Path("data/new_york_fed_securities_lending_2019_2023")
OPERATIONS = (
    SOURCE_ROOT
    / "new_york_fed_securities_lending_operations_2019_2023.csv.gz"
)
OPERATIONS_SHA256 = (
    "99eb8c37c05417789dfad7452c7b2ddc5b6b640078b87451f1c945158af77906"
)
OPERATIONS_HEADER_SHA256 = (
    "c0d63795e5e53cef816c50472c6941069cb018f30ad1f745f250daa0fa6b9200"
)
OPERATIONS_USECOLS = (
    "operation_id",
    "operation_date",
    "available_at_utc",
    "total_par_submitted",
    "total_par_accepted",
)
DETAILS = (
    SOURCE_ROOT
    / "new_york_fed_securities_lending_details_2019_2023.csv.gz"
)
DETAILS_SHA256 = (
    "27178d8738cb50c4e6c13f1e5940fcfdf4009e6979b006c42fb86fb399d0716d"
)
DETAILS_HEADER_SHA256 = (
    "9f4d54dff4b9c9f0c47c0a85e0bf245276e5a3cb764b3c084017f679586b76dd"
)
DETAILS_USECOLS = (
    "operation_id",
    "operation_date",
    "available_at_utc",
    "cusip",
    "security_description",
    "par_submitted",
    "par_accepted",
    "actual_available_to_borrow",
)
SOURCE_MANIFEST = SOURCE_ROOT / "build_manifest.json"
SOURCE_MANIFEST_SHA256 = (
    "58b9eb56728065d919978b8969e9bbb4bcb291f723a290d22045fe2ca3da2019"
)
SOURCE_MANIFEST_CANONICAL_HASH = (
    "748db33b3ea40eb48d126d0e9882b05e1994741bf851a8f3c7b89d5166db969c"
)

SLCS_CLOCK = Path(
    "results/soma_lending_collateral_scarcity_clocks_2026-07-23.csv.gz"
)
SLCS_CLOCK_SHA256 = (
    "b3fe0dc8c895f9a8974cdf08b5bed9d58ff693b8aca7ed59c224627de930a948"
)
SLCS_HEADER_SHA256 = (
    "45a24e800b79a30047ffeb5f45c69cf4817262e57b0af1cf5e046332536e5e94"
)
SLCS_USECOLS = ("control", "entry_time", "exit_time", "side")
SLCS_GROUP_VOCABULARY = (
    "primary",
    "carry_intensity_only",
    "constant_long",
    "constant_short",
    "demand_breadth_only",
    "demand_intensity_only",
    "deterministic_random_side",
    "exact_direction_flip",
    "five_operation_stale",
    "mean_without_consensus",
    "one_operation_stale",
    "same_sign_without_magnitude",
    "weighted_fee_only",
    "year_component_permutation",
)
SLCS_GROUPS = (
    "primary",
    "demand_intensity_only",
    "weighted_fee_only",
    "carry_intensity_only",
    "demand_breadth_only",
)

SCAF_CLOCK = Path(
    "data/soma_collateral_allocation_fracture_clocks_2020_2023.csv.gz"
)
SCAF_CLOCK_SHA256 = (
    "64e07005d70442bfa7a110b1e6bea9802ee94be16d95f6e7db9228f4790a28e6"
)
SCAF_HEADER_SHA256 = (
    "770965eb9e07bbca6f6b3f3c3165fe5c04301ef6573da86dacd161582cfa8c8f"
)
SCAF_USECOLS = ("control", "entry_time", "exit_time", "side")
SCAF_GROUP_VOCABULARY = (
    "primary",
    "inventory_mismatch_only",
    "award_distortion_only",
    "unmet_demand_mass_only",
    "fee_distortion_only",
    "mean_change_without_consensus",
    "two_of_four_without_opposition",
    "one_batch_stale",
    "five_batch_stale",
    "within_batch_demand_permutation",
    "exact_direction_flip",
    "deterministic_random_side",
    "constant_long",
    "constant_short",
)
SCAF_GROUPS = (
    "primary",
    "inventory_mismatch_only",
    "award_distortion_only",
    "unmet_demand_mass_only",
    "fee_distortion_only",
)

MARKET_MANIFEST = Path(
    "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
)
MARKET_MANIFEST_SHA256 = (
    "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
)
MARKET_DATA = Path(
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)
MARKET_DATA_SHA256 = (
    "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
)
MARKET_HEADER_SHA256 = (
    "5e8d51e7e1218929db6a54ca59280eb4306171b81d5d0880467a85cf9d23eff2"
)
FUNDING_MANIFEST = Path(
    "results/binance_um_btcusdt_funding_marks_"
    "2020_2023_manifest_2026-07-17.json"
)
FUNDING_MANIFEST_SHA256 = (
    "a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b"
)
FUNDING_DATA = Path(
    "data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz"
)
FUNDING_DATA_SHA256 = (
    "3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6"
)
FUNDING_HEADER_SHA256 = (
    "71b2b1395313f631969674c43e569c8f1619a9fb23c8316e2e0478c32f01d61f"
)
HYDRATION_MANIFEST = Path(
    "results/smaf_72_economic_artifact_hydration_2026-07-30.json"
)
HYDRATION_LOCAL_LOG = Path(
    ".omx/local/smaf-72-hydration-source-path.json"
)

DESCRIPTION_REGEX = (
    r"\A(?P<label>[A-Z][A-Z0-9/-]{0,15}) "
    r"(?P<coupon>[0-9]{1,2}(?:\.[0-9]{1,6})?) "
    r"(?P<maturity>(?:0[1-9]|1[0-2])/"
    r"(?:0[1-9]|[12][0-9]|3[01])/[0-9]{2})\Z"
)
DESCRIPTION_PATTERN = re.compile(DESCRIPTION_REGEX, flags=re.ASCII)
AMOUNT_REGEX = r"\A(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z"
AMOUNT_PATTERN = re.compile(AMOUNT_REGEX, flags=re.ASCII)
OPERATION_DATE_PATTERN = re.compile(
    r"\A[0-9]{4}-(?:0[1-9]|1[0-2])-"
    r"(?:0[1-9]|[12][0-9]|3[01])\Z",
    flags=re.ASCII,
)

SOURCE_CONTROL_ORDER = (
    "primary",
    "submitted_inventory_tilt",
    "submitted_award_tilt",
    "award_inventory_tilt",
    "aggregate_demand_intensity",
)
OUTCOME_CONTROL_ORDER = (
    "exact_direction_flip",
    "deterministic_random_side",
    "constant_long",
    "constant_short",
    "one_extra_bar_delay",
    "one_operation_delay",
)


@dataclass(frozen=True)
class Policy:
    policy_id: str = POLICY_ID
    history_operations: int = 126
    lower_tail_numerator: int = 1
    upper_tail_numerator: int = 9
    tail_denominator: int = 10
    entry_latency_minutes: int = 5
    hold_bars: int = 864
    hold_hours: int = 72
    leverage: float = 0.5
    base_cost_bp_per_notional_side: float = 6.0
    stress_cost_bp_per_notional_side: float = 10.0
    cluster_signflip_draws: int = 20_000
    train_start: str = "2020-01-01T00:00:00Z"
    train_end: str = "2023-01-01T00:00:00Z"
    selection_end: str = "2024-01-01T00:00:00Z"


@dataclass(frozen=True)
class ParsedDescription:
    label: str
    coupon: Fraction
    maturity: date


def parse_exact_decimal(value: str) -> Fraction:
    if AMOUNT_PATTERN.fullmatch(value) is None:
        raise ValueError("SMAF-72 invalid exact decimal")
    return Fraction(value)


def parse_security_description(value: str) -> ParsedDescription:
    match = DESCRIPTION_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError("SMAF-72 invalid security_description grammar")
    month, day, year = (
        int(token) for token in match.group("maturity").split("/")
    )
    try:
        maturity = date(2000 + year, month, day)
    except ValueError as error:
        raise ValueError(
            "SMAF-72 invalid security_description calendar date"
        ) from error
    return ParsedDescription(
        label=match.group("label"),
        coupon=Fraction(match.group("coupon")),
        maturity=maturity,
    )


def maturity_distance(operation_date: str, description: str) -> int:
    if OPERATION_DATE_PATTERN.fullmatch(operation_date) is None:
        raise ValueError("SMAF-72 invalid operation_date grammar")
    try:
        operation = date.fromisoformat(operation_date)
    except ValueError as error:
        raise ValueError("SMAF-72 invalid operation_date") from error
    tau = (parse_security_description(description).maturity - operation).days
    if not 1 <= tau <= 18_263:
        raise ValueError("SMAF-72 maturity distance outside frozen range")
    return tau


def _path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        raise RuntimeError("SMAF-72 path must be repository-relative")
    resolved = (REPOSITORY_ROOT / candidate).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise RuntimeError(
            "SMAF-72 path must remain inside repository"
        ) from error
    return resolved


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_header_bytes(path: str | Path) -> bytes:
    source = _path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rb") as handle:
        header = handle.readline()
    if not header.endswith(b"\n") or b"\n" in header[:-1]:
        raise RuntimeError("SMAF-72 CSV header is not one canonical line")
    return header


def csv_header(path: str | Path) -> list[str]:
    text = csv_header_bytes(path).decode("utf-8")
    columns = text.removesuffix("\n").removesuffix("\r").split(",")
    if not columns or any(not column for column in columns):
        raise RuntimeError("SMAF-72 CSV header contains empty column")
    if len(columns) != len(set(columns)):
        raise RuntimeError("SMAF-72 CSV header contains duplicates")
    return columns


def sha256_csv_header(path: str | Path) -> str:
    return hashlib.sha256(csv_header_bytes(path)).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def active_frozen_dependencies() -> dict[str, str]:
    """Dependencies safe to hash without opening sealed economic rows."""

    return {
        str(PREREGISTRATION_DOCUMENT): PREREGISTRATION_DOCUMENT_SHA256,
        str(SOURCE_AUDIT): SOURCE_AUDIT_SHA256,
        str(COMMON_WINDOW_POLICY): COMMON_WINDOW_POLICY_SHA256,
        str(SOURCE_BUILDER): SOURCE_BUILDER_SHA256,
        str(OPERATIONS): OPERATIONS_SHA256,
        str(DETAILS): DETAILS_SHA256,
        str(SOURCE_MANIFEST): SOURCE_MANIFEST_SHA256,
        str(SLCS_CLOCK): SLCS_CLOCK_SHA256,
        str(SCAF_CLOCK): SCAF_CLOCK_SHA256,
    }


def validate_frozen_dependencies() -> None:
    for path, expected in active_frozen_dependencies().items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"SMAF-72 frozen dependency changed: {path}")
    for path, expected, allowlist in (
        (OPERATIONS, OPERATIONS_HEADER_SHA256, OPERATIONS_USECOLS),
        (DETAILS, DETAILS_HEADER_SHA256, DETAILS_USECOLS),
        (SLCS_CLOCK, SLCS_HEADER_SHA256, SLCS_USECOLS),
        (SCAF_CLOCK, SCAF_HEADER_SHA256, SCAF_USECOLS),
    ):
        if sha256_csv_header(path) != expected:
            raise RuntimeError(f"SMAF-72 frozen header changed: {path}")
        if not set(allowlist).issubset(csv_header(path)):
            raise RuntimeError(f"SMAF-72 allowlist missing from header: {path}")


def _csv_contract(
    *,
    path: Path,
    sha256: str,
    header_sha256: str,
    usecols: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256,
        "header_sha256": header_sha256,
        "read_csv": {
            "usecols": list(usecols),
            "dtype": "string",
            "keep_default_na": False,
            "na_filter": False,
        },
    }


def _comparator_contracts() -> list[dict[str, Any]]:
    return [
        {
            "id": "SLCS",
            **_csv_contract(
                path=SLCS_CLOCK,
                sha256=SLCS_CLOCK_SHA256,
                header_sha256=SLCS_HEADER_SHA256,
                usecols=SLCS_USECOLS,
            ),
            "group_column": "control",
            "allowed_groups": list(SLCS_GROUP_VOCABULARY),
            "selected_groups": list(SLCS_GROUPS),
            "side_map": {"1": "LONG", "-1": "SHORT"},
            "minimum_contained_rows_each_group": 20,
        },
        {
            "id": "SCAF",
            **_csv_contract(
                path=SCAF_CLOCK,
                sha256=SCAF_CLOCK_SHA256,
                header_sha256=SCAF_HEADER_SHA256,
                usecols=SCAF_USECOLS,
            ),
            "group_column": "control",
            "allowed_groups": list(SCAF_GROUP_VOCABULARY),
            "selected_groups": list(SCAF_GROUPS),
            "side_map": {"LONG": "LONG", "SHORT": "SHORT"},
            "minimum_contained_rows_each_group": 20,
        },
    ]


def _sealed_economic_artifacts() -> dict[str, dict[str, Any]]:
    return {
        "market_manifest": {
            "path": str(MARKET_MANIFEST),
            "sha256": MARKET_MANIFEST_SHA256,
        },
        "market_data": {
            "path": str(MARKET_DATA),
            "sha256": MARKET_DATA_SHA256,
            "header_sha256": MARKET_HEADER_SHA256,
        },
        "funding_manifest": {
            "path": str(FUNDING_MANIFEST),
            "sha256": FUNDING_MANIFEST_SHA256,
        },
        "funding_data": {
            "path": str(FUNDING_DATA),
            "sha256": FUNDING_DATA_SHA256,
            "header_sha256": FUNDING_HEADER_SHA256,
        },
    }


def _core_manifest() -> dict[str, Any]:
    policy = Policy()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "policy": asdict(policy),
        "singleton": True,
        "research_status": "source_seen_outcome_unseen_preincidence",
        "frozen_documents": {
            "preregistration": {
                "path": str(PREREGISTRATION_DOCUMENT),
                "sha256": PREREGISTRATION_DOCUMENT_SHA256,
            },
            "source_audit": {
                "path": str(SOURCE_AUDIT),
                "sha256": SOURCE_AUDIT_SHA256,
            },
            "common_window_policy": {
                "path": str(COMMON_WINDOW_POLICY),
                "sha256": COMMON_WINDOW_POLICY_SHA256,
            },
            "source_builder": {
                "path": str(SOURCE_BUILDER),
                "sha256": SOURCE_BUILDER_SHA256,
            },
        },
        "source_contracts": {
            "operations": _csv_contract(
                path=OPERATIONS,
                sha256=OPERATIONS_SHA256,
                header_sha256=OPERATIONS_HEADER_SHA256,
                usecols=OPERATIONS_USECOLS,
            ),
            "details": _csv_contract(
                path=DETAILS,
                sha256=DETAILS_SHA256,
                header_sha256=DETAILS_HEADER_SHA256,
                usecols=DETAILS_USECOLS,
            ),
            "manifest": {
                "path": str(SOURCE_MANIFEST),
                "sha256": SOURCE_MANIFEST_SHA256,
                "canonical_hash": SOURCE_MANIFEST_CANONICAL_HASH,
            },
            "operation_maturity_date_forbidden": True,
            "raw_api_json_forbidden": True,
            "network_calls": 0,
        },
        "probe_disclosure": {
            "selection": (
                "first eight normalized detail rows from the first "
                "operation_id in file order"
            ),
            "operation_id": "SL 010219 1",
            "operation_date": "2019-01-02",
            "rows": [
                ["912810EC8", "T 08.875 02/15/19"],
                ["912810ED6", "T 08.125 08/15/19"],
                ["912810EE4", "T 08.500 02/15/20"],
                ["912810EF1", "T 08.750 05/15/20"],
                ["912810EG9", "T 08.750 08/15/20"],
                ["912810EH7", "T 07.875 02/15/21"],
                ["912810EJ3", "T 08.125 05/15/21"],
                ["912810EK0", "T 08.125 08/15/21"],
            ],
            "amount_rate_availability_fields_read": False,
            "incidence_or_count_computed": False,
            "failure_action": "retire unchanged; parser repair forbidden",
        },
        "parser_contract": {
            "security_description_regex": DESCRIPTION_REGEX,
            "regex_flags": ["ASCII"],
            "exact_ascii_spaces": True,
            "coupon_validation_only": True,
            "label_validation_only": True,
            "maturity_year": "2000 + YY",
            "gregorian_validation": True,
            "tau": "maturity_date - operation_date in integer calendar days",
            "tau_min": 1,
            "tau_max": 18_263,
            "amount_regex": AMOUNT_REGEX,
            "numeric_arithmetic": "exact rational from decimal digits",
            "float_forbidden": True,
            "trim_or_normalize": False,
        },
        "complete_operation_contract": {
            "operation_id_unique": True,
            "operation_cusip_unique": True,
            "every_detail_joins_exactly_one_operation": True,
            "joined_date_and_availability_exact": True,
            "every_operation_has_detail": True,
            "accepted_not_above_submitted_each_detail": True,
            "detail_operation_totals_reconcile_exactly": [
                "par_submitted",
                "par_accepted",
            ],
            "strictly_positive_centroid_weights": [
                "par_submitted",
                "par_accepted",
                "actual_available_to_borrow",
            ],
            "one_invalid_detail_invalidates_operation": True,
            "row_deletion_or_imputation": False,
            "reason_counts_required": True,
        },
        "causal_batch_contract": {
            "key": "exact available_at_utc",
            "exactly_one_complete_operation_required": True,
            "current_batch_mutually_prior": False,
            "invalid_or_multi_operation_batch_resets": [
                "rolling_history",
                "prior_LOW_state",
                "prior_HIGH_state",
            ],
            "weekend_holiday_or_no_operation_gap_resets": False,
        },
        "feature_contract": {
            "centroid": "sum(W_i*tau_i)/sum(W_i)",
            "weights": {
                "S": "par_submitted",
                "A": "par_accepted",
                "V": "actual_available_to_borrow",
            },
            "primary": "2*C(S)-C(V)-C(A)",
            "decomposition": "(C(S)-C(V))+(C(S)-C(A))",
            "meaning": (
                "submitted maturity relative to the average of "
                "available-to-borrow and accepted-award centroids"
            ),
            "not_duration_dv01_risk_or_causal": True,
            "high_side": "SHORT",
            "low_side": "LONG",
            "polarity_repair_forbidden": True,
        },
        "rank_and_onset": {
            "history": "latest 126 strictly prior complete operations",
            "integer_cross_multiplication_only": True,
            "L": "count(prior < current)",
            "E": "count(prior == current)",
            "midrank": "(2*L+E)/252",
            "LOW": "10*(2*L+E) <= 252",
            "HIGH": "10*(2*L+E) >= 2268",
            "first_rank_ready_operation": "baseline only; cannot trigger",
            "trigger": "false-to-true onset of LOW or HIGH",
            "tail_persistence_emits": False,
            "canonical_signal_id": (
                "sha256_utf8(SMAF-72|<control>|<operation_id>|"
                "<available_at_utc>|<LOW_OR_HIGH>)"
            ),
        },
        "execution": {
            "decision_time": "available_at_utc",
            "ceil_5m": "smallest Unix-epoch multiple of 300s >= decision",
            "entry_time": "ceil_to_5m(decision_time)+5 elapsed minutes",
            "already_aligned_still_waits_minutes": 5,
            "hold_bars": policy.hold_bars,
            "hold_hours": policy.hold_hours,
            "exit_time": "entry_time+72 elapsed hours",
            "candidate_order": ["entry_time", "signal_id"],
            "reservation": "[entry_time,exit_time)",
            "acceptance": "entry_time >= previous accepted exit_time",
            "global_nonoverlap_before_split": True,
            "suppression": "no queue, replacement, or release",
            "source_state_independent_of_suppression": True,
        },
        "windows": {
            "warmup": ["2019-01-01T00:00:00Z", policy.train_start],
            "train": [policy.train_start, policy.train_end],
            "selection": [policy.train_end, policy.selection_end],
            "sealed": [policy.selection_end, None],
            "containment": (
                "operation_date, decision, entry, and exit in same split; "
                "exit may equal exclusive end"
            ),
            "crossing_interval": "report and exclude whole",
            "timezone": "UTC",
        },
        "controls": {
            "source_order": list(SOURCE_CONTROL_ORDER),
            "source_definitions": {
                "primary": "2*C(S)-C(V)-C(A)",
                "submitted_inventory_tilt": "C(S)-C(V)",
                "submitted_award_tilt": "C(S)-C(A)",
                "award_inventory_tilt": "C(A)-C(V)",
                "aggregate_demand_intensity": "sum(S)/sum(V)",
            },
            "source_clocks_reserve_independently": True,
            "outcome_order": list(OUTCOME_CONTROL_ORDER),
            "random_side": {
                "input": "SMAF-72|<primary_signal_id>|RANDOM_SIDE",
                "LONG": "first SHA256 byte < 128",
                "SHORT": "first SHA256 byte >= 128",
            },
            "one_extra_bar_delay": (
                "same event and side; entry and exit +5 elapsed minutes"
            ),
            "one_operation_delay": (
                "same event and side; entry at next complete causal "
                "operation in same uninterrupted segment scheduled entry; "
                "exit +72h; rerun reservation"
            ),
            "delayed_parent_set": "raw parent signal IDs and sides unchanged",
            "delayed_accepted_set": (
                "may shrink under independent overlap, crossing, or "
                "missing same-segment successor; every reason reported"
            ),
            "delayed_controls_may_create_parent_events": False,
        },
        "source_support_gates": {
            "undefined_or_empty": "fail",
            "operation_and_batch_window_attribution": "available_at_utc",
            "split_boundary_resets_rank_history": False,
            "coverage_windows": {
                "full": [
                    "2019-01-01T00:00:00Z",
                    "2024-01-01T00:00:00Z",
                ],
                "warmup": [
                    "2019-01-01T00:00:00Z",
                    "2020-01-01T00:00:00Z",
                ],
                "train": [policy.train_start, policy.train_end],
                "selection": [policy.train_end, policy.selection_end],
            },
            "coverage_formulas": {
                "description_parser_coverage": (
                    "valid parsed joined detail rows / "
                    "all joined detail rows"
                ),
                "complete_operation_share": (
                    "complete operations / all operation rows"
                ),
                "single_operation_batch_share": (
                    "valid one-complete-operation availability batches / "
                    "all distinct available_at_utc batches"
                ),
            },
            "coverage": {
                "each_ratio_exact_in_full_warmup_train_selection": 1.0,
                "train_rank_ready_min": 740,
                "selection_rank_ready_min": 240,
                "each_split_each_raw_tail_share_min": 0.05,
                "each_split_each_raw_tail_share_max": 0.20,
            },
            "train": {
                "events_min": 60,
                "events_max": 180,
                "each_year_min": 15,
                "each_half_min": 6,
                "each_quarter_min": 2,
                "each_side_min": 15,
                "each_side_share_min": 0.20,
                "active_months_min": 20,
                "maximum_month_share": 0.15,
                "maximum_quarter_share": 0.30,
                "maximum_elapsed_entry_gap_days": 90,
                "maximum_same_side_run": 8,
            },
            "selection": {
                "events_min": 18,
                "events_max": 70,
                "each_half_min": 7,
                "each_quarter_min": 3,
                "each_side_min": 5,
                "each_side_share_min": 0.20,
                "active_months_min": 8,
                "maximum_month_share": 0.25,
                "maximum_quarter_share": 0.45,
                "maximum_elapsed_entry_gap_days": 90,
                "maximum_same_side_run": 6,
            },
            "internal_component_distinctness": {
                "controls": list(SOURCE_CONTROL_ORDER[1:]),
                "train_entries_min_each": 30,
                "selection_entries_min_each": 10,
                "each_split_each_side_share_min": 0.20,
                "exact_entry_jaccard_max": 0.70,
                "same_entry_same_side_reproduction_max": 0.70,
                "absolute_signed_occupancy_pearson_max": 0.80,
                "undefined_or_nonfinite_correlation": "fail",
            },
            "failure_order": [
                "frozen_identity_and_exact_header",
                "schema_join_uniqueness_reconciliation",
                "parser_coverage_and_complete_operations",
                "singleton_causal_batches",
                "rank_coverage_and_tail_selectivity",
                "primary_event_support",
                "internal_component_distinctness",
            ],
            "failure_action": (
                "retire SMAF-72 unchanged before external comparators "
                "or outcomes"
            ),
        },
        "novelty_contract": {
            "opens_only_after_all_source_support_passes": True,
            "common_window": [policy.train_start, policy.selection_end],
            "full_interval_containment": True,
            "raw_validation_before_filter": True,
            "comparators": _comparator_contracts(),
            "groups_compared_separately": True,
            "thresholds_each_group": {
                "exact_entry_jaccard_max": 0.20,
                "same_entry_same_side_reproduction_max": 0.30,
                "candidate_24h_containment_max": 0.40,
                "comparator_24h_containment_max": 0.40,
                "absolute_signed_occupancy_pearson_max": 0.35,
            },
            "one_to_one_24h_match": {
                "inputs": "sorted distinct UTC entry times",
                "algorithm": (
                    "two pointers; discard only timestamp >24 elapsed "
                    "hours earlier; otherwise match and advance both"
                ),
                "matched_count": "maximum cardinality",
            },
            "occupancy": {
                "grid": (
                    "complete five-minute UTC grid over "
                    "[2020-01-01,2024-01-01)"
                ),
                "interval": "[entry,exit)",
                "LONG": 1,
                "SHORT": -1,
                "idle": 0,
            },
            "duplicate_entry_overlap_non5m_empty_or_nonfinite": "fail",
            "all_selected_groups_must_pass": True,
            "failure_action": "retire SMAF-72 unchanged before outcomes",
        },
        "sealed_economic_artifacts": _sealed_economic_artifacts(),
        "economic_hydration_contract": {
            "logical_paths_and_hashes_define_identity": True,
            "materialization_state_at_freeze": {
                "market_manifest": False,
                "market_data": False,
                "funding_manifest": True,
                "funding_data": True,
            },
            "matching_market_bytes_hash_checked_outside_worktree": True,
            "external_store_location_frozen_or_recorded_here": False,
            "before_economic_evaluator_commit": [
                "copy bytes into exact logical repository-relative paths",
                "reject symlink or non-regular file",
                "set hydrated files read-only",
                "validate all four full-file SHA256 identities",
                "validate both gzip header SHA256 identities",
                f"write exact canonical manifest {HYDRATION_MANIFEST}",
                "open zero market or funding rows until checks pass",
            ],
            "manifest": {
                "path": str(HYDRATION_MANIFEST),
                "protocol_version": (
                    "smaf_72_economic_artifact_hydration_v1"
                ),
                "encoding": "UTF-8",
                "trailing_lf_count": 1,
                "serialization": {
                    "sort_keys": True,
                    "indent": 2,
                    "ensure_ascii": True,
                    "allow_nan": False,
                },
                "top_level_keys_exact": [
                    "protocol_version",
                    "artifacts",
                    "market_rows_opened",
                    "funding_rows_opened",
                    "manifest_hash",
                ],
                "artifact_keys_exact": [
                    "logical_path",
                    "portable_source_locator",
                    "copied_at_utc",
                    "size_bytes",
                    "sha256",
                    "header_sha256",
                    "regular_file",
                    "symlink",
                    "mode",
                    "rows_opened_before_validation",
                ],
                "artifact_order": "ascending logical_path",
                "portable_source_locator": (
                    "local-cache:sha256:<artifact sha256>"
                ),
                "copied_at_utc": (
                    "UTC RFC3339 YYYY-MM-DDTHH:MM:SS.ffffffZ"
                ),
                "mode": "0444",
                "regular_file": True,
                "symlink": False,
                "rows_opened_before_validation": 0,
                "market_rows_opened": 0,
                "funding_rows_opened": 0,
                "manifest_hash": (
                    "SHA256 compact sorted-key JSON without manifest_hash; "
                    "separators comma/colon, ensure_ascii true, "
                    "allow_nan false"
                ),
            },
            "host_absolute_source_path": {
                "allowed_only_in_uncommitted_gitignored_log": str(
                    HYDRATION_LOCAL_LOG
                ),
                "forbidden_from_committed_manifest_evaluator_and_results": True,
            },
            "later_evaluator_freezes": [
                "hydration manifest file SHA256",
                "hydration internal manifest_hash",
            ],
            "later_evaluator_validates_before_rows": True,
            "evaluator_fallback_artifact_root": None,
            "evaluator_refuses_absent_logical_path": True,
        },
        "economic_contract": {
            "rows_open_authorized": False,
            "later_evaluator_must_validate_identity_and_header_first": True,
            "instrument": "Binance USD-M BTCUSDT perpetual",
            "initial_equity": 1.0,
            "side_sign": {"LONG": 1, "SHORT": -1},
            "entry_price": "exact 5m open at entry_time",
            "exit_price": "exact 5m open at exit_time",
            "quantity": "0.5*pre_entry_equity/entry_open; fixed through exit",
            "base_cost_bp_per_notional_side": (
                policy.base_cost_bp_per_notional_side
            ),
            "stress_cost_bp_per_notional_side": (
                policy.stress_cost_bp_per_notional_side
            ),
            "cost_cash": "abs(quantity)*execution_price*bp/10000",
            "funding_interval": "entry_time <= funding_time < exit_time",
            "funding_cash": (
                "-side_sign*quantity*funding_rate*settlement_mark_price"
            ),
            "full_calendar_cagr": (
                "includes idle cash; elapsed_seconds/(365.25*86400)"
            ),
            "strict_mdd": (
                "global/pre-entry HWM; entry fee; funding credits before "
                "favorable held-bar extreme; funding debits before adverse "
                "extreme; adverse virtual exit fee; scheduled exit fee"
            ),
            "mean_gross_underlying_bp": (
                "mean(side_sign*(exit_open/entry_open-1)*10000)"
            ),
            "weekly_cluster_signflip": {
                "cluster": "UTC ISO entry week base-cost net trade PnL",
                "draws": policy.cluster_signflip_draws,
                "draw_indices": "0..19999 formatted as five digits",
                "stage_tokens": [
                    "TRAIN_2020_2022",
                    "SELECTION_2023",
                ],
                "iso_year_format": "four digits",
                "iso_week_format": "two digits",
                "bit_input": (
                    "UTF8 no newline: SMAF-72|<STAGE_TOKEN>|"
                    "<DRAW_00000>|<ISO_YEAR_4>-W<ISO_WEEK_2>"
                ),
                "bit": "most significant bit of digest byte zero; &0x80",
                "bit_one_multiplier": -1,
                "bit_zero_multiplier": 1,
                "p_value": (
                    "(1+count(flipped_total>=observed_total))/20001"
                ),
            },
            "stops_take_profit_early_exit_or_overlap": False,
        },
        "economic_gates": {
            "train_2020_2022": {
                "executed_trades_min": 60,
                "each_year_min": 15,
                "each_side_min": 15,
                "base_and_stress_absolute_return_positive": True,
                "each_year_base_absolute_return_positive": True,
                "base_cagr_to_strict_mdd_min": 3.0,
                "stress_cagr_to_strict_mdd_min": 2.5,
                "base_and_stress_strict_mdd_max": 0.15,
                "mean_gross_underlying_bp_min": 35.0,
                "weekly_cluster_signflip_p_max": 0.10,
                "each_delay_absolute_return_positive": True,
                "direction_random_constant_margin": (
                    "primary base absolute return strictly greater"
                ),
                "mechanism_control_ratio_margin_min": 0.25,
                "mechanism_control_absolute_return_margin": (
                    "primary strictly greater than every source control"
                ),
            },
            "selection_2023": {
                "executed_trades_min": 18,
                "each_half_min": 7,
                "each_side_min": 5,
                "base_and_stress_absolute_return_positive": True,
                "each_half_base_absolute_return_positive": True,
                "base_cagr_to_strict_mdd_min": 3.0,
                "stress_cagr_to_strict_mdd_min": 2.5,
                "base_and_stress_strict_mdd_max": 0.15,
                "mean_gross_underlying_bp_min": 35.0,
                "weekly_cluster_signflip_p_max": 0.20,
                "each_delay_absolute_return_positive": True,
                "direction_random_constant_margin": (
                    "primary base absolute return strictly greater"
                ),
                "mechanism_control_ratio_margin_min": 0.25,
                "mechanism_control_absolute_return_margin": (
                    "primary strictly greater than every source control"
                ),
            },
            "zero_mdd_or_nonfinite_control_ratio": "fail margin comparison",
            "required_joint_report": [
                "absolute_return",
                "cagr",
                "strict_mdd",
                "cagr_to_strict_mdd",
                "trades",
                "long_trades",
                "short_trades",
                "subperiod_returns",
                "mean_gross_underlying_bp",
                "funding_cash",
                "cost_cash",
                "weekly_clusters",
                "weekly_cluster_signflip_p",
            ],
        },
        "strict_sequence": [
            "preregistration_document_commit",
            "write_once_preregistration_commit",
            "source_support_evaluator_commit",
            "source_support",
            "internal_component_distinctness",
            "external_comparator_novelty",
            "economic_evaluator_commit",
            "train_2020_2022_once",
            "selection_2023_once_if_train_passes",
            "post_2023_source_extension",
        ],
        "sequence_rules": {
            "stop_at_first_failure": True,
            "no_parameter_or_parser_repair": True,
            "selection_rows_loaded_during_train": False,
            "selection_opens_only_after_train_pass": True,
            "post_2023_sealed": True,
        },
        "rllm_boundary": {
            "activation_requires_deterministic_train_and_selection_pass": True,
            "actions": ["TRADE_FIXED_SIDE", "ABSTAIN"],
            "fit_window": "2020-2022 only",
            "selection_feedback_may_change_policy": False,
            "may_not_change": [
                "candidate clock",
                "side",
                "entry",
                "exit",
                "hold",
                "leverage",
                "accounting",
            ],
            "forbidden": [
                "raw maturity amount or rank numerator",
                "timestamp date or split identity",
                "price return funding reward or PnL",
                "CAGR MDD or evaluated outcome",
            ],
        },
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "candidate_comparator_overlap_opened": False,
        "economic_rows_opened": False,
        "evidence_boundary": {
            "historical_access_independently_auditable_from_pre_head": False,
            "disclosure_is_conservative_contamination_envelope": True,
            "pristine_source_claim": False,
            "description_probe_rows_read": 8,
            "probe_identity_rows_read": 8,
            "source_amount_rate_or_availability_rows_read": 0,
            "smaf_centroids_fractures_ranks_tails_or_events_derived": 0,
            "slcs_rows_scanned_for_group_inventory": 1_685,
            "scaf_rows_scanned_for_group_inventory": 5_809,
            "candidate_overlap_metrics_computed": 0,
            "btc_market_rows_loaded": 0,
            "funding_data_rows_loaded": 0,
            "forward_returns_pnl_cagr_or_mdd_opened": False,
            "network_calls": 0,
            "subprocess_calls": 0,
        },
    }


def build_manifest() -> dict[str, Any]:
    core = _core_manifest()
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_manifest(payload: Mapping[str, Any]) -> None:
    expected = build_manifest()
    if dict(payload) != expected:
        raise RuntimeError("SMAF-72 preregistration differs from code")
    core = {
        key: value for key, value in payload.items() if key != "manifest_hash"
    }
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("SMAF-72 preregistration manifest hash mismatch")
    for key in (
        "outcomes_opened",
        "source_incidence_opened",
        "candidate_comparator_overlap_opened",
        "economic_rows_opened",
    ):
        if payload.get(key) is not False:
            raise RuntimeError(f"SMAF-72 evidence boundary opened: {key}")
    boundary = payload["evidence_boundary"]
    forbidden = (
        "source_amount_rate_or_availability_rows_read",
        "smaf_centroids_fractures_ranks_tails_or_events_derived",
        "candidate_overlap_metrics_computed",
        "btc_market_rows_loaded",
        "funding_data_rows_loaded",
        "network_calls",
        "subprocess_calls",
    )
    if any(boundary[key] != 0 for key in forbidden):
        raise RuntimeError("SMAF-72 preregistration opened forbidden evidence")
    if boundary["forward_returns_pnl_cagr_or_mdd_opened"] is not False:
        raise RuntimeError("SMAF-72 preregistration opened outcomes")


def canonical_manifest_bytes(payload: Mapping[str, Any]) -> bytes:
    validate_manifest(payload)
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


def _output_relative(path: str | Path) -> Path:
    candidate = Path(path)
    raw = str(path)
    if (
        candidate.is_absolute()
        or raw.startswith("~")
        or ".." in candidate.parts
        or candidate.name in {"", ".", ".."}
    ):
        raise RuntimeError("SMAF-72 output must be repository-relative")
    return candidate


def _assert_secure_io_capabilities() -> None:
    for name in ("O_NOFOLLOW", "O_DIRECTORY"):
        value = getattr(os, name, None)
        if not isinstance(value, int) or value == 0:
            raise RuntimeError(f"SMAF-72 requires nonzero os.{name}")
    for function in (os.open, os.link, os.unlink):
        if function not in os.supports_dir_fd:
            raise RuntimeError(
                f"SMAF-72 requires dir_fd support for {function.__name__}"
            )
    if os.link not in os.supports_follow_symlinks:
        raise RuntimeError(
            "SMAF-72 requires follow_symlinks support for os.link"
        )


def _open_parent(candidate: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    current = os.open(REPOSITORY_ROOT, flags)
    try:
        for part in candidate.parent.parts:
            next_descriptor = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = next_descriptor
        return current
    except OSError as error:
        os.close(current)
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise RuntimeError(
                "SMAF-72 output parent path is unsafe"
            ) from error
        raise


def _read_regular(directory: int, name: str) -> bytes:
    flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=directory)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise RuntimeError("SMAF-72 output path is unsafe") from error
        raise
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError("SMAF-72 output is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _publish_temporary(
    parent: int,
    temporary_name: str,
    output_name: str,
) -> None:
    os.link(
        temporary_name,
        output_name,
        src_dir_fd=parent,
        dst_dir_fd=parent,
        follow_symlinks=False,
    )


def write_once(
    output: str | Path = DEFAULT_OUTPUT,
    payload: Mapping[str, Any] | None = None,
) -> str:
    candidate = _output_relative(output)
    _assert_secure_io_capabilities()
    validate_frozen_dependencies()
    expected = build_manifest() if payload is None else dict(payload)
    canonical = canonical_manifest_bytes(expected)
    parent = _open_parent(candidate)
    temporary_name = (
        f".{candidate.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    temporary_created = False
    try:
        try:
            existing = _read_regular(parent, candidate.name)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if existing != canonical:
                raise RuntimeError(
                    "SMAF-72 existing preregistration is noncanonical"
                )
            return "verified_existing"
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent,
        )
        temporary_created = True
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical)
            handle.flush()
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        try:
            _publish_temporary(
                parent,
                temporary_name,
                candidate.name,
            )
        except FileExistsError:
            if _read_regular(parent, candidate.name) != canonical:
                raise RuntimeError("SMAF-72 preregistration race drift")
            return "verified_existing"
        os.fsync(parent)
        return "created"
    finally:
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=parent)
            except FileNotFoundError:
                pass
            os.fsync(parent)
        os.close(parent)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_manifest()
    status = write_once(args.output, payload)
    print(
        json.dumps(
            {
                "status": status,
                "output": str(args.output),
                "manifest_hash": payload["manifest_hash"],
                "source_incidence_opened": payload[
                    "source_incidence_opened"
                ],
                "outcomes_opened": payload["outcomes_opened"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
