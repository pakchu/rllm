"""Freeze SCAF-48 before source incidence, comparators, or market outcomes."""

from __future__ import annotations

import argparse
import errno
import gzip
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import secrets
import stat
from typing import Any, Mapping


POLICY_ID = "SCAF-48"
PROTOCOL_VERSION = "soma_collateral_allocation_fracture_preregistration_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path("training/preregister_soma_collateral_allocation_fracture.py")
DEFAULT_OUTPUT = Path(
    "results/soma_collateral_allocation_fracture_"
    "preregistration_2026-07-24.json"
)

BOUNDARY = Path(
    "docs/soma-collateral-allocation-fracture-boundary-2026-07-24.md"
)
BOUNDARY_SHA256 = (
    "faade722ffa8f7ce67db50cb34e55d371a31c9d3770e96f1b9507e8470b340d3"
)
SCHEMA_AMENDMENT = Path(
    "docs/scaf-normalized-source-schema-preincidence-amendment-2026-07-24.md"
)
SCHEMA_AMENDMENT_SHA256 = (
    "8e6a2ef1be5c5e93c5e998cb8b9d7a9ddf0a3e931042d6d0f47238af0b39b5d2"
)
MECHANISM = Path(
    "docs/soma-collateral-allocation-fracture-"
    "mechanism-decision-2026-07-24.md"
)
MECHANISM_SHA256 = (
    "af203719ed3111880ff5528723b2d9c0878ec9ba659739baf7862df96d007728"
)
SOURCE_AUDIT = Path(
    "docs/new-york-fed-securities-lending-source-audit-2026-07-23.md"
)
SOURCE_AUDIT_SHA256 = (
    "c812998be0bd44efc09b8120d9bc0b2a96f4e1e95f9414a4c8458d97319307bc"
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
OPERATIONS_ALLOWLIST = (
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
DETAILS_ALLOWLIST = (
    "operation_id",
    "operation_date",
    "available_at_utc",
    "cusip",
    "par_submitted",
    "par_accepted",
    "weighted_average_rate",
    "actual_available_to_borrow",
)
SOURCE_MANIFEST = SOURCE_ROOT / "build_manifest.json"
SOURCE_MANIFEST_SHA256 = (
    "58b9eb56728065d919978b8969e9bbb4bcb291f723a290d22045fe2ca3da2019"
)

SLCS_CLOCK = Path(
    "results/soma_lending_collateral_scarcity_clocks_2026-07-23.csv.gz"
)
SLCS_CLOCK_SHA256 = (
    "b3fe0dc8c895f9a8974cdf08b5bed9d58ff693b8aca7ed59c224627de930a948"
)
SLCS_CLOCK_HEADER_SHA256 = (
    "45a24e800b79a30047ffeb5f45c69cf4817262e57b0af1cf5e046332536e5e94"
)
SLCS_USECOLS = ("control", "entry_time", "exit_time", "side")
SLCS_GROUPS = (
    "primary",
    "demand_intensity_only",
    "weighted_fee_only",
    "carry_intensity_only",
    "demand_breadth_only",
    "mean_without_consensus",
    "same_sign_without_magnitude",
)
SLCS_PREREGISTRATION = Path(
    "results/soma_lending_collateral_scarcity_"
    "preregistration_2026-07-23.json"
)
SLCS_PREREGISTRATION_SHA256 = (
    "517d53437db55773bd98d7513ee5722dd1c03769a519393985f978994b3edc1a"
)
SLCS_SUPPORT = Path(
    "results/soma_lending_collateral_scarcity_support_2026-07-23.json"
)
SLCS_SUPPORT_SHA256 = (
    "354f3edb9f1d9bdbac1f609e50882f2e4d1df6ee8cfa555287ca99a15148a738"
)
SLCS_REJECTION = Path(
    "docs/soma-lending-collateral-scarcity-"
    "support-rejection-2026-07-23.md"
)
SLCS_REJECTION_SHA256 = (
    "fb7e46c7f86ab6fd839e5d79fd26f3077547dbadfef241b1490bd09009f7521b"
)

CONTROL_ORDER = (
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
COMPONENT_ORDER = (
    "inventory_mismatch",
    "award_distortion",
    "unmet_demand_mass",
    "fee_distortion",
)


@dataclass(frozen=True)
class Policy:
    policy_id: str = POLICY_ID
    hold_bars: int = 576
    hold_minutes: int = 2_880
    entry_latency_minutes: int = 5
    notional: float = 0.5
    train_start: str = "2020-01-01T00:00:00Z"
    train_end: str = "2023-01-01T00:00:00Z"
    selection_end: str = "2024-01-01T00:00:00Z"


def _path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        raise RuntimeError("SCAF-48 path must be repository-relative")
    resolved = (REPOSITORY_ROOT / candidate).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise RuntimeError(
            "SCAF-48 path must remain inside repository"
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
        raise RuntimeError("SCAF-48 CSV header is not one canonical line")
    return header


def csv_header(path: str | Path) -> list[str]:
    text = csv_header_bytes(path).decode("utf-8")
    columns = text.removesuffix("\n").removesuffix("\r").split(",")
    if not columns or any(not column for column in columns):
        raise RuntimeError("SCAF-48 CSV header contains empty column")
    if len(columns) != len(set(columns)):
        raise RuntimeError("SCAF-48 CSV header contains duplicates")
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


def frozen_dependencies() -> dict[str, str]:
    return {
        str(BOUNDARY): BOUNDARY_SHA256,
        str(SCHEMA_AMENDMENT): SCHEMA_AMENDMENT_SHA256,
        str(MECHANISM): MECHANISM_SHA256,
        str(SOURCE_AUDIT): SOURCE_AUDIT_SHA256,
        str(SOURCE_BUILDER): SOURCE_BUILDER_SHA256,
        str(OPERATIONS): OPERATIONS_SHA256,
        str(DETAILS): DETAILS_SHA256,
        str(SOURCE_MANIFEST): SOURCE_MANIFEST_SHA256,
        str(SLCS_CLOCK): SLCS_CLOCK_SHA256,
        str(SLCS_PREREGISTRATION): SLCS_PREREGISTRATION_SHA256,
        str(SLCS_SUPPORT): SLCS_SUPPORT_SHA256,
        str(SLCS_REJECTION): SLCS_REJECTION_SHA256,
    }


def validate_frozen_dependencies() -> None:
    for path, expected in frozen_dependencies().items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"SCAF-48 frozen dependency changed: {path}")
    for path, expected, allowlist in (
        (OPERATIONS, OPERATIONS_HEADER_SHA256, OPERATIONS_ALLOWLIST),
        (DETAILS, DETAILS_HEADER_SHA256, DETAILS_ALLOWLIST),
        (SLCS_CLOCK, SLCS_CLOCK_HEADER_SHA256, SLCS_USECOLS),
    ):
        if sha256_csv_header(path) != expected:
            raise RuntimeError(f"SCAF-48 frozen header changed: {path}")
        if not set(allowlist).issubset(csv_header(path)):
            raise RuntimeError(f"SCAF-48 allowlist missing from header: {path}")


def _source_contract(
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


def _core_manifest() -> dict[str, Any]:
    policy = Policy()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "policy": asdict(policy),
        "singleton": True,
        "research_status": "source_seen_outcome_unseen_preincidence",
        "frozen_documents": {
            "boundary": {
                "path": str(BOUNDARY),
                "sha256": BOUNDARY_SHA256,
            },
            "schema_amendment": {
                "path": str(SCHEMA_AMENDMENT),
                "sha256": SCHEMA_AMENDMENT_SHA256,
            },
            "mechanism": {
                "path": str(MECHANISM),
                "sha256": MECHANISM_SHA256,
            },
            "source_audit": {
                "path": str(SOURCE_AUDIT),
                "sha256": SOURCE_AUDIT_SHA256,
            },
            "source_builder": {
                "path": str(SOURCE_BUILDER),
                "sha256": SOURCE_BUILDER_SHA256,
            },
        },
        "source_contracts": {
            "operations": _source_contract(
                path=OPERATIONS,
                sha256=OPERATIONS_SHA256,
                header_sha256=OPERATIONS_HEADER_SHA256,
                usecols=OPERATIONS_ALLOWLIST,
            ),
            "details": _source_contract(
                path=DETAILS,
                sha256=DETAILS_SHA256,
                header_sha256=DETAILS_HEADER_SHA256,
                usecols=DETAILS_ALLOWLIST,
            ),
            "manifest": {
                "path": str(SOURCE_MANIFEST),
                "sha256": SOURCE_MANIFEST_SHA256,
            },
            "forbidden_columns": [
                "security_description",
                "soma_holdings",
                "theoretical_available_to_borrow",
                "outstanding_loans",
                "settlement_date",
                "maturity_date",
                "note",
            ],
            "raw_api_json_forbidden": True,
            "network_calls": 0,
        },
        "batch_contract": {
            "key": "exact available_at_utc",
            "atom_identity": ["operation_id", "cusip"],
            "equal_cusips_across_operations_merged": False,
            "simultaneous_batches_are_mutually_prior": False,
            "reference": "immediately previous complete strictly earlier batch",
            "invalid_batch_breaks_continuity": True,
            "first_complete_after_break_can_trigger": False,
            "positive_batch_totals_required": [
                "par_submitted",
                "actual_available_to_borrow",
                "par_accepted",
                "par_accepted_times_weighted_average_rate",
            ],
            "operation_detail_reconciliation": "exact decimal",
        },
        "numeric_contract": {
            "amount_arithmetic": "arbitrary precision Decimal",
            "atom_order": ["operation_id", "cusip"],
            "jsd_logarithm": "natural log normalized by ln(2)",
            "jsd_accumulator": "math.fsum over binary64 shares",
            "zero_log_convention": "0*log(0/m)=0",
            "component_quantum": "1e-12",
            "rounding": "ROUND_HALF_EVEN",
            "unmet_mass_decimal_context_precision": 80,
            "unmet_mass_binary_float_forbidden": True,
        },
        "components": {
            "order": list(COMPONENT_ORDER),
            "inventory_mismatch": "JSD(submitted_share, actual_available_share)",
            "award_distortion": "JSD(submitted_share, accepted_share)",
            "unmet_demand_mass": (
                "sum(submitted where accepted==0) / sum(submitted)"
            ),
            "fee_distortion": (
                "JSD(accepted_share, accepted_fee_mass_share)"
            ),
            "higher_meaning": "greater cross-sectional allocation fracture",
            "aggregate_slcs_levels_used": False,
            "rolling_rank_or_fitted_scale_used": False,
        },
        "transition": {
            "comparison": "quantized current versus previous component",
            "tokens": ["UP", "DOWN", "FLAT"],
            "fracture": "count(UP) >= 3",
            "relief": "count(DOWN) >= 3",
            "neutral": "otherwise",
            "side": {
                "FRACTURE": "SHORT",
                "RELIEF": "LONG",
                "NEUTRAL": "ABSTAIN",
            },
            "magnitude_threshold": None,
            "every_consensus_batch_is_raw_opportunity": True,
        },
        "execution": {
            "canonical_signal_timestamp": "YYYY-MM-DDTHH:MM:SSZ",
            "fractional_source_seconds_allowed": False,
            "primary_signal_id": (
                "sha256_utf8(SCAF-48|timestamp|FRACTURE_OR_RELIEF)"
            ),
            "entry": "ceil_to_5m(signal) + 5 elapsed minutes",
            "hold_bars": policy.hold_bars,
            "hold_minutes": policy.hold_minutes,
            "interval": "[entry, exit)",
            "global_nonoverlap_before_split": True,
            "reservation_sort": [
                "entry_time",
                "signal_available_time",
                "signal_id",
            ],
            "acceptance": "entry >= previous accepted exit",
            "suppressed_events_queued": False,
            "entry_exit_same_split_required": True,
            "notional": policy.notional,
            "stop_or_take_profit": False,
        },
        "windows": {
            "warmup": ["2019-01-01T00:00:00Z", policy.train_start],
            "train": [policy.train_start, policy.train_end],
            "selection": [policy.train_end, policy.selection_end],
            "sealed": [policy.selection_end, None],
            "support_calendar_timezone": "UTC",
        },
        "controls": {
            "order": list(CONTROL_ORDER),
            "independent_reservation": list(CONTROL_ORDER[1:10]),
            "primary_clock_side_only": list(CONTROL_ORDER[10:]),
            "stale_history_reset_on_invalid_batch": True,
            "permutation": {
                "field": "submitted_share_p",
                "destination_hash_bytes": (
                    "SCAF-48\\0timestamp\\0operation_id\\0cusip"
                ),
                "encoding": "UTF-8",
                "destination_order": [
                    "digest_bytes",
                    "operation_id",
                    "cusip",
                ],
                "source_order": ["operation_id", "cusip"],
                "nul_in_identifier": "reject",
            },
            "random_side": {
                "input": "SCAF-48|<primary_signal_id>|RANDOM_SIDE",
                "digest_integer": "unsigned big-endian",
                "even": "LONG",
                "odd": "SHORT",
            },
        },
        "source_support_gate": {
            "coverage": {
                "train_complete_batches_min": 700,
                "selection_complete_batches_min": 220,
                "train_valid_transitions_min": 690,
                "selection_valid_transitions_min": 215,
                "each_split_raw_consensus_share_min": 0.10,
                "each_split_raw_consensus_share_max": 0.65,
            },
            "train": {
                "events_min": 120,
                "each_year_events_min": 30,
                "long_min": 30,
                "short_min": 30,
                "active_months_min": 30,
                "maximum_utc_calendar_gap_days": 30,
                "maximum_month_share": 0.12,
                "maximum_quarter_share": 0.25,
                "maximum_same_side_run": 12,
            },
            "selection": {
                "events_min": 35,
                "each_half_events_min": 15,
                "each_quarter_events_min": 7,
                "long_min": 10,
                "short_min": 10,
                "active_months_min": 10,
                "maximum_utc_calendar_gap_days": 30,
                "maximum_month_share": 0.20,
                "maximum_same_side_run": 8,
            },
            "every_control_nonempty_each_split": True,
            "empty_or_undefined": "fail",
            "failure_action": (
                "retire SCAF-48 unchanged before comparator rows or outcomes"
            ),
        },
        "composition_gate": {
            "each_split": {
                "each_component_raw_agreement_min": 0.55,
                "each_component_raw_agreement_max": 0.95,
                "four_of_four_share_min": 0.10,
                "four_of_four_share_max": 0.85,
                "exact_three_of_four_share_min": 0.15,
                "each_component_control_reproduction_max": 0.80,
                "mean_change_reproduction_max": 0.90,
                "each_stale_reproduction_max": 0.75,
                "random_side_reproduction_max": 0.60,
                "permutation_exact_entry_jaccard_max": 0.50,
                "permutation_same_side_reproduction_max": 0.65,
            },
            "raw_denominator": (
                "split-contained raw primary opportunities before reservation"
            ),
            "flat_component_counts_as_disagreement": True,
            "control_reproduction_denominator": "accepted primary",
        },
        "novelty_contract": {
            "opens_only_after_source_and_composition_pass": True,
            "common_window": [
                policy.train_start,
                policy.selection_end,
            ],
            "comparator": {
                "path": str(SLCS_CLOCK),
                "sha256": SLCS_CLOCK_SHA256,
                "header_sha256": SLCS_CLOCK_HEADER_SHA256,
                "read_csv": {
                    "usecols": list(SLCS_USECOLS),
                    "dtype": "string",
                    "keep_default_na": False,
                    "na_filter": False,
                },
                "group_column": "control",
                "groups": list(SLCS_GROUPS),
                "minimum_contained_rows_each": 20,
                "exact_group_equality": True,
            },
            "thresholds_each_group": {
                "exact_entry_jaccard_max": 0.25,
                "one_new_york_calendar_day_jaccard_max": 0.50,
                "same_entry_same_side_reproduction_max": 0.30,
                "absolute_signed_occupancy_pearson_max": 0.35,
            },
            "one_day_matching": {
                "timezone": "America/New_York",
                "criterion": "absolute local-date difference <= 1",
                "algorithm": "maximum-cardinality deterministic augmenting path",
                "jaccard_denominator": (
                    "candidate + comparator - matched"
                ),
            },
            "occupancy": {
                "grid": "complete five-minute common-window cells",
                "interval": "[entry, exit)",
                "LONG": 1,
                "SHORT": -1,
                "idle": 0,
                "self_overlap": "fail",
                "constant_or_nonfinite": "fail",
            },
            "bound_prior_artifacts": {
                "slcs_preregistration": {
                    "path": str(SLCS_PREREGISTRATION),
                    "sha256": SLCS_PREREGISTRATION_SHA256,
                },
                "slcs_support": {
                    "path": str(SLCS_SUPPORT),
                    "sha256": SLCS_SUPPORT_SHA256,
                },
                "slcs_rejection": {
                    "path": str(SLCS_REJECTION),
                    "sha256": SLCS_REJECTION_SHA256,
                },
            },
        },
        "live_fail_flat": {
            "capture_and_hash_official_response": True,
            "preorder_integrity_failure": "invalidate order and halt",
            "failure_while_flat": "remain flat and halt",
            "failure_while_positioned": (
                "one reduce-only market flatten request and halt"
            ),
            "emergency_flatten_is_alpha_exit": False,
            "restart": "audited new complete baseline batch",
        },
        "economic_rllm_sequence": {
            "economic_evaluator_authorized": False,
            "deterministic_source_train_selection_pass_required": True,
            "deterministic_cagr_to_strict_mdd_min": 3.0,
            "strict_mdd": (
                "global/pre-entry HWM plus intratrade adverse path and costs"
            ),
            "full_wall_clock_cagr": True,
            "exact_funding_and_executable_costs": True,
            "rllm_actions": ["TRADE_FIXED_SIDE", "ABSTAIN"],
            "rllm_tokens": [
                "four_component_UP_DOWN_FLAT",
                "FRACTURE_RELIEF",
                "three_vs_four_agreement",
                "prior_symbolic_relation",
                "batch_validity",
                "position_fixed_side_time_in_position",
                "fixed_risk_budget",
            ],
            "rllm_forbidden": [
                "raw_components",
                "amounts",
                "rates",
                "cusips",
                "operation_ids",
                "dates",
                "timestamps",
                "btc_prices",
                "future_paths",
                "split_labels",
                "ranks",
                "rewards",
                "evaluated_outcomes",
            ],
        },
        "strict_sequence": [
            "boundary_commit",
            "schema_amendment_commit",
            "mechanism_commit",
            "write_once_preregistration_commit",
            "source_support_evaluator_commit",
            "source_support",
            "relational_composition",
            "slcs_same_source_novelty",
            "economic_rllm_evaluator_commit",
            "train_2020_2022",
            "selection_2023",
            "post_2023_source_extension",
        ],
        "research_history_boundary": {
            "slcs_source_values_and_incidence_seen": True,
            "slcs_market_or_funding_outcomes_seen": False,
            "dclb_source_support_failure_seen": True,
            "scaf_source_incidence_seen": False,
            "scaf_comparator_rows_seen": False,
            "scaf_market_or_funding_outcomes_seen": False,
            "pristine_soma_source_claim": False,
            "outcome_unseen_scaf_claim": True,
        },
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "source_rows_decoded": False,
        "comparator_rows_decoded": False,
        "evidence_boundary": {
            "operation_value_rows_read": 0,
            "detail_value_rows_read": 0,
            "scaf_components_computed": 0,
            "scaf_batches_derived": 0,
            "scaf_transitions_derived": 0,
            "scaf_candidate_events_derived": 0,
            "slcs_comparator_value_rows_read": 0,
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "future_return_rows_read": 0,
            "pnl_cagr_mdd_opened": False,
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
        raise RuntimeError("SCAF-48 preregistration differs from code")
    core = {
        key: value for key, value in payload.items() if key != "manifest_hash"
    }
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("SCAF-48 preregistration manifest hash mismatch")
    for key in (
        "outcomes_opened",
        "source_incidence_opened",
        "source_rows_decoded",
        "comparator_rows_decoded",
    ):
        if payload.get(key) is not False:
            raise RuntimeError(f"SCAF-48 evidence boundary opened: {key}")
    boundary = payload["evidence_boundary"]
    for key, value in boundary.items():
        if value not in (0, False):
            raise RuntimeError(
                f"SCAF-48 preregistration decoded forbidden evidence: {key}"
            )


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
        raise RuntimeError("SCAF-48 output must be repository-relative")
    return candidate


def _open_parent(candidate: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    current = os.open(REPOSITORY_ROOT, flags)
    try:
        for part in candidate.parent.parts:
            next_descriptor = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = next_descriptor
        return current
    except Exception:
        os.close(current)
        raise


def _read_regular(directory: int, name: str) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=directory)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise RuntimeError("SCAF-48 output path is unsafe") from error
        raise
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError("SCAF-48 output is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def write_once(
    output: str | Path = DEFAULT_OUTPUT,
    payload: Mapping[str, Any] | None = None,
) -> str:
    validate_frozen_dependencies()
    candidate = build_manifest() if payload is None else dict(payload)
    encoded = canonical_manifest_bytes(candidate)
    relative = _output_relative(output)
    parent = _open_parent(relative)
    temporary = f".{relative.name}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
    temporary_created = False
    try:
        try:
            existing = _read_regular(parent, relative.name)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if existing != encoded:
                raise RuntimeError(
                    "SCAF-48 existing preregistration is noncanonical"
                )
            return "verified_existing"
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent,
        )
        temporary_created = True
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(
                temporary,
                relative.name,
                src_dir_fd=parent,
                dst_dir_fd=parent,
                follow_symlinks=False,
            )
        except FileExistsError:
            if _read_regular(parent, relative.name) != encoded:
                raise RuntimeError("SCAF-48 preregistration race drift")
            return "verified_existing"
        os.fsync(parent)
        return "created"
    finally:
        if temporary_created:
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
            os.fsync(parent)
        os.close(parent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    arguments = parser.parse_args()
    payload = build_manifest()
    status = write_once(arguments.output, payload)
    print(
        json.dumps(
            {
                "output": arguments.output,
                "status": status,
                "manifest_hash": payload["manifest_hash"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
