"""Freeze PIVOT-72 before decoding source incidence or market outcomes."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


DEFAULT_OUTPUT = (
    "results/paired_intrinsic_venue_orderflow_topology_"
    "preregistration_2026-07-24.json"
)
BOUNDARY_DOCUMENT = (
    "docs/paired-intrinsic-venue-orderflow-topology-boundary-2026-07-24.md"
)
BOUNDARY_DOCUMENT_SHA256 = (
    "dd06a3aea17596e2d1e451b5c3f8f3d98af5691d9ffe5f1ac26af59d8e8fcacb"
)
BOUNDARY_COMMIT = "00f9d3e"
MECHANISM_DOCUMENT = (
    "docs/paired-intrinsic-venue-orderflow-topology-"
    "mechanism-decision-2026-07-24.md"
)
MECHANISM_DOCUMENT_SHA256 = (
    "7d9cbf6ea3ad3ad938f52c80bf76bd1585d6adc8d55ac6bcb4df888112990d02"
)
MECHANISM_COMMIT = "69ff60c"

SOURCE = (
    "data/binance_cross_venue_minute_leadership_btc_2020_2023/"
    "BTCUSDT_cross_venue_minute_leadership_5m_2020-01_2023-12.csv.gz"
)
SOURCE_SHA256 = (
    "00ab6a55fc7bfeb3012584db5bc97a7d7b98dd995491acfd3f865c6bd41f92cc"
)
SOURCE_HEADER_SHA256 = (
    "b7c730d6fc2c37d6e94f6a436478fd09ff42d15d7fd81bf521c4ca36465ff49f"
)
SOURCE_MANIFEST = (
    "data/binance_cross_venue_minute_leadership_btc_2020_2023/"
    "build_manifest.json"
)
SOURCE_MANIFEST_SHA256 = (
    "544c2945a2b56be478a1edc4abbb93b762bda5afc32cbd0658dd6822ff6b70fa"
)
SOURCE_AUDIT = "results/binance_cross_venue_minute_leadership_audit_2026-07-14.json"
SOURCE_AUDIT_SHA256 = (
    "ffe0124ac9c5c0c3f1d1c284b672618cf910dc16cae36e65c1efe79710f039af"
)

MARKET_SOURCE = (
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)
MARKET_SOURCE_SHA256 = (
    "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
)
MARKET_HEADER_SHA256 = (
    "5e8d51e7e1218929db6a54ca59280eb4306171b81d5d0880467a85cf9d23eff2"
)
MARKET_MANIFEST = (
    "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
)
MARKET_MANIFEST_SHA256 = (
    "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
)

FUNDING_SOURCE = "data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz"
FUNDING_SOURCE_SHA256 = (
    "3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6"
)
FUNDING_HEADER_SHA256 = (
    "71b2b1395313f631969674c43e569c8f1619a9fb23c8316e2e0478c32f01d61f"
)
FUNDING_MANIFEST = (
    "results/binance_um_btcusdt_funding_marks_2020_2023_"
    "manifest_2026-07-17.json"
)
FUNDING_MANIFEST_SHA256 = (
    "a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b"
)

SOURCE_ALLOWLIST = (
    "date",
    "feature_available_time_utc",
    "trade_earliest_time_utc",
    "spot_quote_notional",
    "um_quote_notional",
    "spot_signed_quote_notional",
    "um_signed_quote_notional",
    "source_complete",
)

TOKEN_SCHEMA: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("leader", ("SPOT", "UM")),
    ("gap_q", ("Q0", "Q1", "Q2", "Q3")),
    ("early_session", ("S00_06", "S06_12", "S12_18", "S18_24")),
    ("laggard_progress_q", ("Q0", "Q1", "Q2", "Q3")),
    ("spot_early_sign", ("NEG", "ZERO", "POS")),
    ("um_early_sign", ("NEG", "ZERO", "POS")),
    ("spot_late_sign", ("NEG", "ZERO", "POS")),
    ("um_late_sign", ("NEG", "ZERO", "POS")),
    ("spot_late_abs_flow_q", ("Q0", "Q1", "Q2", "Q3")),
    ("um_late_abs_flow_q", ("Q0", "Q1", "Q2", "Q3")),
    ("gap_change", ("NARROW", "SAME", "WIDEN")),
    ("leader_change", ("SAME", "SWITCH")),
)
TOKEN_COLUMNS = tuple(name for name, _ in TOKEN_SCHEMA)
TOKEN_VOCABULARY = {name: values for name, values in TOKEN_SCHEMA}
SIGN_TOKEN_COLUMNS = (
    "spot_early_sign",
    "um_early_sign",
    "spot_late_sign",
    "um_late_sign",
)
ACTION_NAMES = ("ABSTAIN", "LONG", "SHORT")
ACTION_PRIORITY = {action: index for index, action in enumerate(ACTION_NAMES)}
FORBIDDEN_COMPARATOR_PATHS = (
    "data/premium_compression_breakout_relay_clocks_2020_2026.csv.gz",
    "data/premium_snapback_recenter_clocks_2020_2026.csv.gz",
)


@dataclass(frozen=True)
class Policy:
    policy_id: str = "PIVOT-72"
    reference_calendar_days: int = 28
    reference_complete_days_min: int = 21
    intrinsic_volume_fraction: float = 0.50
    latest_anchor_start_minute_utc: int = 23 * 60 + 50
    prior_base_states: int = 180
    prior_base_states_min: int = 90
    quartiles: tuple[float, float, float] = (0.25, 0.50, 0.75)
    buffer_bars: int = 1
    inference_order_bars: int = 1
    entry_delay_bars_from_late_anchor: int = 3
    hold_bars: int = 72
    leverage: float = 0.50
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010
    held_path_drawdown_penalty: float = 1.0 / 3.0
    trade_utility_hurdle_account: float = 0.0010
    preference_pair_margin: float = 0.0005
    cluster_signflip_draws: int = 100_000
    random_seed: int = 20_260_724


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def jsonable(payload: Any) -> Any:
    return json.loads(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
    )


def csv_header_bytes(path: str | Path) -> bytes:
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rb") as handle:
        header = handle.readline()
    if not header.endswith(b"\n") or b"\n" in header[:-1]:
        raise RuntimeError(f"PIVOT-72 CSV header is not one LF line: {path}")
    return header


def csv_header(path: str | Path) -> list[str]:
    header = csv_header_bytes(path).decode("utf-8")
    return next(csv.reader([header.rstrip("\n")]))


def sha256_csv_header(path: str | Path) -> str:
    return hashlib.sha256(csv_header_bytes(path)).hexdigest()


def sign_token(value: float) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("PIVOT sign input must be finite")
    if numeric < 0.0:
        return "NEG"
    if numeric > 0.0:
        return "POS"
    return "ZERO"


def prior_quartile_bucket(
    value: float,
    previous_values: Iterable[float],
    *,
    minimum: int = Policy().prior_base_states_min,
    maximum: int = Policy().prior_base_states,
) -> str:
    prior = np.asarray(list(previous_values), dtype=np.float64)
    if len(prior) < minimum:
        raise ValueError("PIVOT prior state history is not ready")
    if len(prior) > maximum:
        prior = prior[-maximum:]
    current = np.float64(value)
    if not np.isfinite(current) or not np.isfinite(prior).all():
        raise ValueError("PIVOT quartile values must be finite")
    thresholds = np.quantile(
        prior,
        np.asarray(Policy().quartiles, dtype=np.float64),
        method="linear",
    )
    bucket = int(np.searchsorted(thresholds, current, side="right"))
    return f"Q{bucket}"


def validate_tokens(tokens: Mapping[str, str]) -> dict[str, str]:
    if tuple(tokens) != TOKEN_COLUMNS:
        raise ValueError("PIVOT token order or schema changed")
    normalized = {name: str(tokens[name]) for name in TOKEN_COLUMNS}
    for name, value in normalized.items():
        if value not in TOKEN_VOCABULARY[name]:
            raise ValueError(f"PIVOT token level is invalid: {name}={value}")
    return normalized


def sign_mirror_tokens(tokens: Mapping[str, str]) -> dict[str, str]:
    mirrored = validate_tokens(tokens)
    sign_map = {"NEG": "POS", "ZERO": "ZERO", "POS": "NEG"}
    for name in SIGN_TOKEN_COLUMNS:
        mirrored[name] = sign_map[mirrored[name]]
    return mirrored


def venue_swap_tokens(tokens: Mapping[str, str]) -> dict[str, str]:
    swapped = validate_tokens(tokens)
    swapped["leader"] = {"SPOT": "UM", "UM": "SPOT"}[swapped["leader"]]
    for left, right in (
        ("spot_early_sign", "um_early_sign"),
        ("spot_late_sign", "um_late_sign"),
        ("spot_late_abs_flow_q", "um_late_abs_flow_q"),
    ):
        swapped[left], swapped[right] = swapped[right], swapped[left]
    return swapped


def sign_mirror_action(action: str) -> str:
    if action not in ACTION_NAMES:
        raise ValueError("PIVOT action is invalid")
    return {"ABSTAIN": "ABSTAIN", "LONG": "SHORT", "SHORT": "LONG"}[action]


def action_option_orders() -> tuple[tuple[str, str, str], ...]:
    return tuple(itertools.permutations(ACTION_NAMES))


def opportunity_times(late_anchor: pd.Timestamp | str) -> dict[str, pd.Timestamp]:
    anchor = pd.Timestamp(late_anchor)
    if anchor.tzinfo is None:
        anchor = anchor.tz_localize("UTC")
    else:
        anchor = anchor.tz_convert("UTC")
    bar = pd.Timedelta(minutes=5)
    return {
        "late_anchor": anchor,
        "state_completion": anchor + bar,
        "buffer_completion": anchor + 2 * bar,
        "decision_deadline": anchor + 3 * bar,
        "entry": anchor + 3 * bar,
        "exit": anchor + (3 + Policy().hold_bars) * bar,
    }


def _static_comparator_contracts() -> list[dict[str, Any]]:
    standard_header = [
        "signal_position",
        "entry_position",
        "exit_position",
        "signal_date",
        "entry_date",
        "exit_date",
        "side",
        "branch",
        "hold_bars",
    ]
    intrinsic_header = [
        "clock_name",
        "source_day",
        "decision_time",
        "entry_time",
        "exit_time",
        "side",
    ]
    return [
        {
            "id": "CVICR",
            "path": "data/cross_venue_intrinsic_clock_resolution_clocks_2020_2023.csv.gz",
            "sha256": "9f05b372686805539dbf56fb9b7ea7a8f90f8887d6731e1a8e1b1c1db14d8c0e",
            "header": [
                "control",
                "signal_id",
                "source_day",
                "causal_origin_time",
                "resolution_time",
                "signal_available_time",
                "decision_time",
                "entry_time",
                "exit_time",
                "side",
                "leader",
            ],
            "header_sha256": "94842895d19a3f6e0b20eebec5758abd5aad421c5b8d7cc9161270f1b8372ce8",
            "entry_column": "entry_time",
            "exit_column": "exit_time",
            "side_column": "side",
            "side_encoding": {"LONG": 1, "SHORT": -1},
            "group_column": "control",
            "selected_groups": [
                "primary",
                "gap_only",
                "initial_conflict_only",
                "late_alignment_only",
                "no_leader_persistence",
                "no_gap_tail",
                "fixed_expected_time_clocks",
                "stale_laggard_flow_24h",
                "exact_direction_flip",
                "deterministic_random_side",
                "one_bar_execution_delay",
                "one_hour_execution_delay",
            ],
            "declared_coverage": [
                "2020-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
            ],
            "six_hour_tolerant_gate": True,
        },
        {
            "id": "CATCH-12",
            "path": "results/cash_auction_transfer_catchup_handoff_clock_2026-07-14.csv",
            "sha256": "066bf8e08267a043cc191eb436f0aa33105ab948de9f9f1edfde4d9c30de46d1",
            "header": standard_header,
            "header_sha256": "3211ca0e50e607f39ef0c8ee72329828a267cdab15de029002b0b5beaeb2032c",
            "entry_column": "entry_date",
            "exit_column": "exit_date",
            "side_column": "side",
            "side_encoding": {"1": 1, "-1": -1},
            "group_column": "branch",
            "selected_groups": ["catch12"],
            "declared_coverage": [
                "2020-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
            ],
            "six_hour_tolerant_gate": False,
        },
        {
            "id": "CLASP-24",
            "path": "results/cash_late_arrival_spillover_propagation_clock_2026-07-14.csv",
            "sha256": "e166f4bd24afd5a2f129bcc26393ad4293ad0bc5792686b3b0fc4a805d53f9d5",
            "header": standard_header,
            "header_sha256": "3211ca0e50e607f39ef0c8ee72329828a267cdab15de029002b0b5beaeb2032c",
            "entry_column": "entry_date",
            "exit_column": "exit_date",
            "side_column": "side",
            "side_encoding": {"1": 1, "-1": -1},
            "group_column": "branch",
            "selected_groups": ["clasp24"],
            "declared_coverage": [
                "2020-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
            ],
            "six_hour_tolerant_gate": False,
        },
        {
            "id": "LURI-48",
            "path": "results/leveraged_um_inventory_release_handoff_clock_2026-07-14.csv",
            "sha256": "50765cfed0c3ec6a0d1df18857c4e0a3e574d1aa449538c9b89cfac1fff67095",
            "header": standard_header,
            "header_sha256": "3211ca0e50e607f39ef0c8ee72329828a267cdab15de029002b0b5beaeb2032c",
            "entry_column": "entry_date",
            "exit_column": "exit_date",
            "side_column": "side",
            "side_encoding": {"1": 1, "-1": -1},
            "group_column": "branch",
            "selected_groups": ["luri48"],
            "declared_coverage": [
                "2020-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
            ],
            "six_hour_tolerant_gate": False,
        },
        {
            "id": "CVTT-V01-V04",
            "path": "data/cross_venue_temporal_torsion_v2_support_clocks_2020_2022.csv.gz",
            "sha256": "8f933b9d387fbcb764645a7002a5eefa9ee159c9c1ce7e007dca0dc4c16ebe33",
            "header": [
                "policy_id",
                "route",
                "side",
                "hold_bars",
                "signal_date",
                "signal_row",
                "entry_date",
            ],
            "header_sha256": "a182548526587fa060072ca6fc2ab284167d12e9b56dcc1c6194c5d5f65cc683",
            "entry_column": "entry_date",
            "exit_column": None,
            "exit_derivation": "entry_date + hold_bars*5m",
            "side_column": "side",
            "side_encoding": {"1": 1, "-1": -1},
            "group_column": "policy_id",
            "selected_groups": ["V01", "V02", "V03", "V04"],
            "declared_coverage": [
                "2020-01-01T00:00:00Z",
                "2023-01-01T00:00:00Z",
            ],
            "selection_absence_is_not_failure": True,
            "six_hour_tolerant_gate": False,
        },
        {
            "id": "IVLIR-primary",
            "path": "data/intrinsic_volume_latent_impact_relay_clocks_2020_2023.csv.gz",
            "sha256": "523f24a0d955fe99cfb86c62078532c5fc9091234e6669ab9acff2a8f3367788",
            "header": intrinsic_header,
            "header_sha256": "0ad7d7a39f7d772de30d2c47056efd3c9b7740561eea9a1b69007b4870d5d495",
            "entry_column": "entry_time",
            "exit_column": "exit_time",
            "side_column": "side",
            "side_encoding": {"LONG": 1, "SHORT": -1},
            "group_column": "clock_name",
            "selected_groups": ["primary"],
            "declared_coverage": [
                "2020-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
            ],
            "six_hour_tolerant_gate": True,
        },
        {
            "id": "IVFHR-primary-and-any-handoff",
            "path": "data/intrinsic_volume_flow_handoff_relay_clocks_2020_2023.csv.gz",
            "sha256": "ab12762dec9a93d41c293766e46dfc80ade81914fb32753a5923faa6437c338e",
            "header": intrinsic_header,
            "header_sha256": "0ad7d7a39f7d772de30d2c47056efd3c9b7740561eea9a1b69007b4870d5d495",
            "entry_column": "entry_time",
            "exit_column": "exit_time",
            "side_column": "side",
            "side_encoding": {"LONG": 1, "SHORT": -1},
            "group_column": "clock_name",
            "selected_groups": ["primary", "any_handoff"],
            "compare_groups_separately": True,
            "declared_coverage": [
                "2020-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
            ],
            "six_hour_tolerant_gate": True,
        },
        {
            "id": "IVPLH-primary",
            "path": "data/intrinsic_volume_price_lag_handoff_clocks_2020_2023.csv.gz",
            "sha256": "2efca3b44b0512a9423da90171f43babcadec2316dc6148796f3e61f98138e80",
            "header": [
                "control",
                "signal_id",
                "source_day",
                "decision_time",
                "entry_time",
                "exit_time",
                "side",
            ],
            "header_sha256": "d5ae2566140aca706f84f916965352daed3aad058e4abcc9614f19f4950f0bbd",
            "entry_column": "entry_time",
            "exit_column": "exit_time",
            "side_column": "side",
            "side_encoding": {"LONG": 1, "SHORT": -1},
            "group_column": "control",
            "selected_groups": ["primary"],
            "declared_coverage": [
                "2021-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
            ],
            "six_hour_tolerant_gate": True,
        },
        {
            "id": "CCHR-live-pre2024",
            "path": "results/cchr_live_portfolio_pure_clocks_2026-07-21.csv.gz",
            "sha256": "73d6efbd35b3be64b0fa04fa9c8cb2db25866ef884f19b1ae673949e22a42b08",
            "header": [
                "candidate_id",
                "split",
                "decision_time",
                "entry_time",
                "exit_time",
                "side",
            ],
            "header_sha256": "da21cfc42a55581971c304cc30122a72f9d062a7601db59a9702ec35504acb9a",
            "entry_column": "entry_time",
            "exit_column": "exit_time",
            "side_column": "side",
            "side_encoding": {"1": 1, "-1": -1},
            "group_column": "candidate_id",
            "selected_groups": "every",
            "declared_coverage": [
                "2021-08-09T01:00:00Z",
                "2024-01-01T00:00:00Z",
            ],
            "rows": 440,
            "post_2023_row_policy": "hard fail; raw post-2023 sources forbidden",
            "six_hour_tolerant_gate": False,
        },
    ]


def _carta_contract() -> dict[str, Any]:
    return {
        "kind": "deferred_action_only_reproduction",
        "policies": ["relational_ridge", "naive_bayes"],
        "views": ["emitted", "executed"],
        "state_source": {
            "path": "training/preregister_causal_adaptive_relational_tokens.py",
            "sha256": "a3a0be1c8c4401bfb707176d9def951938471805597d51c66f92500bafc4f4af",
        },
        "policy_source": {
            "path": "training/causal_adaptive_relational_bandit.py",
            "sha256": "7cb4428b39c923dc909fbd380cef6bb8647c47a5acef099d75c8d5c22d518b68",
        },
        "evaluator_source": {
            "path": "training/evaluate_causal_adaptive_relational_baselines.py",
            "sha256": "130bc08767d6f4d71541215a66b4a88fdc160081e14849ab0000066bb7f3dc21",
        },
        "support_result": {
            "path": "results/causal_adaptive_relational_tokens_support_2026-07-14.json",
            "sha256": "77dfd1d0b0ad444744157972aa437f805901bc56428a4e5d76029bf64100d339",
        },
        "baseline_result": {
            "path": "results/causal_adaptive_relational_baseline_selection_2026-07-14.json",
            "sha256": "b17ef30fd97bc8054a49e42c84d406439c547b97fbd8fb94f0baf59625c55a75",
        },
        "allowed_output_columns": [
            "policy_id",
            "view",
            "decision_time",
            "entry_time",
            "exit_time",
            "action",
            "side",
            "execution_admitted",
        ],
        "execution_order": "only after PIVOT policy and pre-2024 action clock freeze",
        "failure_action": "retire PIVOT before eval",
    }


def _core_manifest() -> dict[str, Any]:
    policy = Policy()
    return {
        "protocol_version": "paired_intrinsic_venue_orderflow_topology_v1",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "source_rows_decoded": False,
        "market_value_rows_decoded": False,
        "funding_value_rows_decoded": False,
        "comparator_rows_decoded": False,
        "post_2023_values_decoded": False,
        "policy": jsonable(asdict(policy)),
        "frozen_documents": {
            "boundary": {
                "path": BOUNDARY_DOCUMENT,
                "sha256": BOUNDARY_DOCUMENT_SHA256,
                "commit": BOUNDARY_COMMIT,
            },
            "mechanism": {
                "path": MECHANISM_DOCUMENT,
                "sha256": MECHANISM_DOCUMENT_SHA256,
                "commit": MECHANISM_COMMIT,
            },
        },
        "research_history_boundary": {
            "repo_wide_btc_history_seen": True,
            "predecessor_cross_venue_outcomes_seen": True,
            "cvicr_source_only_aggregate_seen": True,
            "exact_pivot_source_row_seen": False,
            "exact_pivot_token_or_incidence_seen": False,
            "exact_pivot_post_entry_outcome_seen": False,
            "pivot_comparator_row_seen": False,
            "post_2023_pivot_value_seen": False,
            "claim_scope": "candidate-level frozen eval, not globally pristine history",
            "llm_used_in_this_stage": False,
        },
        "source_contract": {
            "source": SOURCE,
            "source_sha256": SOURCE_SHA256,
            "source_header_sha256": SOURCE_HEADER_SHA256,
            "source_manifest": SOURCE_MANIFEST,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_audit": SOURCE_AUDIT,
            "source_audit_sha256": SOURCE_AUDIT_SHA256,
            "rows": 420_768,
            "interval": [
                "2020-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
            ],
            "bar_interval": "5min",
            "allowlist": list(SOURCE_ALLOWLIST),
            "loader": "pandas.read_csv(usecols=allowlist); no load-and-drop",
            "complete_reference_day": "all 288 source-valid rows",
            "reference_window": "exact prior 28 calendar days; minimum 21 complete",
            "expected_volume": "float64 median of complete daily quote notionals",
            "target": "0.50 * positive finite expected_volume per venue",
            "anchor": "first completed bar reaching venue target; start <=23:50 UTC",
            "tie": "ineligible",
            "current_prefix": "00:00 through buffer row A_late+5m inclusive",
            "later_source_defect": "cannot cancel an entered position",
        },
        "prior_transform_contract": {
            "base_state": (
                "source-valid reference-ready non-tied pair before ordinal "
                "readiness, reservation, split, position, action, or outcome"
            ),
            "history": "immediately previous at most 180 base states",
            "minimum": 90,
            "current_excluded": True,
            "suppressed_states_included": True,
            "quantile_method": "numpy float64 linear",
            "quantiles": [0.25, 0.50, 0.75],
            "bucket_rule": "numpy.searchsorted(thresholds,current,side='right')",
            "duplicate_thresholds": "preserved",
            "future_append_invariant": True,
        },
        "token_contract": {
            "ordered_schema": [
                {"name": name, "values": list(values)}
                for name, values in TOKEN_SCHEMA
            ],
            "count": len(TOKEN_SCHEMA),
            "zero_flow": "ZERO token; not rejected",
            "unknown_downstream_level": "deterministic ABSTAIN",
            "position": "deterministic pre-model guard, not a token",
            "forbidden": [
                "date",
                "year",
                "month",
                "quarter",
                "row_id",
                "event_id",
                "raw_timestamp",
                "raw_numeric",
                "raw_price",
                "return",
                "basis",
                "funding",
                "premium",
                "open_interest",
                "kimchi",
                "dxy",
                "reward",
                "action",
                "pnl",
                "comparator",
                "free_form_rationale",
                "post_2023_value",
            ],
        },
        "execution_contract": {
            "origin": "A_early",
            "state_completion": "A_late+5m",
            "buffer_completion": "A_late+10m",
            "decision_deadline": "A_late+15m",
            "entry": "USD-M BTCUSDT open at A_late+15m",
            "exit": "entry+72*5m scheduled open",
            "action_space": list(ACTION_NAMES),
            "global_action_independent_reservation": True,
            "abstention_releases_reservation": False,
            "crossing_candidate_keeps_reservation": True,
            "split_containment": (
                "origin, anchors, buffer, decision window, entry, held bars, "
                "and exit inside one half-open split"
            ),
            "position_conflict": "deterministic ABSTAIN",
            "stop_take_profit_trailing": None,
            "one_bar_delay": "entry+5m and exit+5m; recompute reservation",
            "one_hour_delay": "entry+60m and exit+60m; recompute reservation",
        },
        "temporal_roles": {
            "train_model_fit": [
                "2020-01-01T00:00:00Z",
                "2022-01-01T00:00:00Z",
            ],
            "selection_not_untouched_test": [
                "2022-01-01T00:00:00Z",
                "2023-01-01T00:00:00Z",
            ],
            "untouched_eval": [
                "2023-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
            ],
            "sealed_start": "2024-01-01T00:00:00Z",
            "adaptation": None,
        },
        "source_support_gates": {
            "global_opportunities_min": 750,
            "train_opportunities_min": 350,
            "each_train_year_min": 150,
            "selection_opportunities_min": 200,
            "eval_opportunities_min": 200,
            "selection_eval_each_half_min": 85,
            "selection_eval_each_quarter_min": 35,
            "train_2020_active_months_min": 7,
            "train_2021_active_months_min": 12,
            "selection_2022_active_months_min": 12,
            "eval_2023_active_months_min": 12,
            "max_month_share": 0.15,
            "max_entry_gap_days": 14.0,
            "each_leader_share_min": 0.20,
            "each_sign_negative_positive_share_min": 0.20,
            "zero_share_min": None,
            "each_quartile_share_range": [0.10, 0.40],
            "session_levels_min": 3,
            "max_session_share": 0.65,
            "gap_narrow_widen_each_share_min": 0.20,
            "leader_switch_share_min": 0.15,
            "max_exact_signature_share": 0.03,
            "downstream_levels_must_exist_in_train": True,
            "counts_basis": "token-ready globally reserved split-contained",
            "prior_history_basis": "all base states including suppressed",
            "required_builder_test_scopes": ["synthetic", "real_prefix"],
            "required_builder_tests": [
                "venue_swap",
                "sign_mirror",
                "future_append",
                "current_value_exclusion_from_every_prior_quartile",
                "suppressed_state_inclusion_in_prior_history",
                "missing_prefix",
                "exact_anchor_tie",
                "exact_zero_sign_preservation",
                "duplicate_quartile_boundaries",
                "option_order_independence_at_serialization",
                "action_independent_reservation",
            ],
            "failure_action": "retire PIVOT-72 unchanged before outcomes",
        },
        "economic_contract": {
            "market": {
                "path": MARKET_SOURCE,
                "sha256": MARKET_SOURCE_SHA256,
                "header_sha256": MARKET_HEADER_SHA256,
                "manifest": MARKET_MANIFEST,
                "manifest_sha256": MARKET_MANIFEST_SHA256,
            },
            "funding": {
                "path": FUNDING_SOURCE,
                "sha256": FUNDING_SOURCE_SHA256,
                "header_sha256": FUNDING_HEADER_SHA256,
                "manifest": FUNDING_MANIFEST,
                "manifest_sha256": FUNDING_MANIFEST_SHA256,
                "interval": "entry_time <= funding_time <= exit_time",
                "cash": "-side*fixed_quantity*settlement_mark*funding_rate",
                "boundary": "retain debits; drop credits at exact entry or exit",
            },
            "quantity": "entry_equity*0.5/entry_open; fixed through trade",
            "base_cost_notional_per_side": 0.0006,
            "stress_cost_notional_per_side": 0.0010,
            "stress_replaces_base": True,
            "strict_mdd": (
                "global HWM; entry cost; favorable held extreme plus credits; "
                "adverse held extreme plus credits/debits and virtual exit "
                "cost; scheduled exit and cost; exclude exit-bar later OHLC"
            ),
            "cagr": "full half-open calendar using 365.25 days",
            "zero_mdd_ratio": {
                "positive_cagr_cap": 1.0e12,
                "otherwise": 0.0,
                "epsilon_pct": 1.0e-12,
            },
            "mean_gross_underlying_move_bp": (
                "mean(side*(exit_open/entry_open-1))*10000"
            ),
            "cluster_signflip": {
                "cluster": "UTC ISO entry week Monday 00:00",
                "draws": policy.cluster_signflip_draws,
                "seed": policy.random_seed,
                "alternative": "one-sided positive mean",
            },
        },
        "reward_contract": {
            "abstain_utility": 0.0,
            "trade_utility": (
                "log(max(account_multiplier,1e-12))"
                " -(1/3)*local_strict_drawdown -0.0010"
            ),
            "hurdle_is_account_level_not_execution_cost": True,
            "oracle_tie_priority": list(ACTION_NAMES),
            "preference_margin": policy.preference_pair_margin,
            "preference_pairs": "all unordered action pairs above margin",
            "outcome_balancing": None,
            "labels_from": "2020-2021 only",
        },
        "baseline_contract": {
            "fit": "2020-2021 only",
            "selection": "2022 only",
            "representation": (
                "train one-hot main levels plus all 66 unordered pair "
                "conjunctions; minimum feature count 3"
            ),
            "nominal_integer_encoding": False,
            "policies": {
                "constants": ["always_abstain", "always_long", "always_short"],
                "signature_memory": {
                    "tie_priority": list(ACTION_NAMES),
                    "unseen": "ABSTAIN",
                },
                "categorical_naive_bayes": {"alpha": 1.0},
                "ridge_contextual_value": {
                    "alpha": 100.0,
                    "unpenalized_intercept": True,
                    "trade_utility_floor": 0.0,
                },
                "extra_trees_contextual_value": {
                    "n_estimators": 512,
                    "criterion": "squared_error",
                    "max_depth": 5,
                    "min_samples_split": 20,
                    "min_samples_leaf": 10,
                    "max_features": "sqrt",
                    "bootstrap": False,
                    "random_state": policy.random_seed,
                    "trade_utility_floor": 0.0,
                },
                "shuffled_oracle_label": {
                    "count": 32,
                    "seeds": [policy.random_seed + index for index in range(32)],
                },
                "shuffled_action_utility": {
                    "count": 32,
                    "seeds": [policy.random_seed + index for index in range(32)],
                },
                "single_token_ridge": 12,
                "leave_one_token_out_ridge": 12,
            },
            "learnability_gate": {
                "absolute_return_positive": True,
                "cagr_to_strict_mdd_min": 1.0,
                "strict_mdd_pct_max": 15.0,
                "both_halves_positive": True,
                "trades_min": 40,
                "each_half_trades_min": 15,
                "each_side_trades_min": 10,
                "max_action_share": 0.90,
                "stress_return_positive": True,
                "one_bar_delay_return_positive": True,
                "weekly_cluster_p_strictly_below": 0.20,
                "beat_strongest_shuffle_return_and_ratio": True,
                "beat_strongest_single_token_return_and_ratio": True,
            },
            "selection_order": [
                "higher_ratio",
                "higher_absolute_return",
                "lower_strict_mdd",
                "lexicographically_smaller_policy_id",
            ],
        },
        "gemma_contract": {
            "model": "google/gemma-4-E2B-it",
            "revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
            "loader": "transformers.AutoModelForCausalLM",
            "tokenizer": "transformers.AutoTokenizer",
            "trust_remote_code": False,
            "text_only": True,
            "quantization": {
                "load_in_4bit": True,
                "type": "nf4",
                "double_quant": True,
                "compute_dtype": "bfloat16",
            },
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.05,
                "bias": "none",
                "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
            "runtime_versions": {
                "torch": "2.9.0",
                "transformers_git": "5d7ff4393ab99aa7cadf4cccd1f814dbb799f2bb",
                "trl": "0.29.0",
                "peft": "0.18.1",
                "bitsandbytes": "0.49.2",
                "numpy": "2.2.6",
                "pandas": "2.3.3",
                "scikit_learn": "1.7.2",
            },
            "memory_gates": {
                "inference_reserved_gib_max": 7.5,
                "training_reserved_gib_max": 24.0,
                "training_allocated_gib_max": 20.0,
                "checkpoint_mib_max": 256,
                "retained_adapters_gib_max": 1.0,
            },
            "prompt": {
                "token_order": list(TOKEN_COLUMNS),
                "option_orders": [list(order) for order in action_option_orders()],
                "max_prompt_completion_tokens": 384,
                "completions": [f"ACTION={action}" for action in ACTION_NAMES],
                "inference": (
                    "mean length-normalized conditional completion log-prob "
                    "over all six option orders"
                ),
                "score_tie_atol": 1.0e-12,
                "score_tie_priority": list(ACTION_NAMES),
                "generation": False,
            },
            "training_symmetry": {
                "views": ["identity", "sign_mirror"],
                "venue_swap": "source-builder equivariance control only",
            },
            "sft": {
                "optimizer": "AdamW",
                "learning_rate": 2.0e-4,
                "betas": [0.9, 0.999],
                "epsilon": 1.0e-8,
                "weight_decay": 0.01,
                "scheduler": "cosine",
                "warmup_steps": 8,
                "max_grad_norm": 1.0,
                "optimizer_steps": 64,
                "per_device_batch_size": 1,
                "gradient_accumulation": 8,
                "packing": False,
                "completion_only_loss": True,
                "bf16": True,
                "seed": policy.random_seed,
            },
            "dpo": {
                "loss": "sigmoid",
                "beta": 0.1,
                "label_smoothing": 0.0,
                "optimizer": "AdamW",
                "learning_rate": 5.0e-6,
                "betas": [0.9, 0.999],
                "epsilon": 1.0e-8,
                "weight_decay": 0.01,
                "scheduler": "cosine",
                "warmup_steps": 8,
                "max_grad_norm": 1.0,
                "optimizer_steps": 96,
                "per_device_batch_size": 1,
                "gradient_accumulation": 8,
                "bf16": True,
                "seed": policy.random_seed,
                "checkpoints": [24, 48, 72, 96],
            },
            "selection_gate_2022": {
                "absolute_return_positive": True,
                "cagr_to_strict_mdd_min": 2.0,
                "strict_mdd_pct_max": 15.0,
                "both_halves_positive": True,
                "trades_min": 50,
                "each_half_trades_min": 20,
                "each_side_trades_min": 15,
                "max_action_share": 0.90,
                "stress_return_positive": True,
                "one_bar_delay_return_positive": True,
                "weekly_cluster_p_strictly_below": 0.15,
                "beat_cheap_absolute_return": True,
                "ratio_margin_over_cheap_min": 0.25,
            },
            "checkpoint_order": [
                "higher_ratio",
                "higher_absolute_return",
                "lower_strict_mdd",
                "earlier_step",
            ],
        },
        "runtime_probe_evidence": {
            "candidate_values_opened": False,
            "model_class": "Gemma4ForConditionalGeneration",
            "parameter_count": 3_936_020_000,
            "text_only_4bit_allocated_bytes": 6_767_540_736,
            "text_only_4bit_reserved_bytes": 6_790_578_176,
            "load_seconds": 5.26,
            "gpu": "NVIDIA GeForce RTX 5090",
        },
        "novelty_contract": {
            "parse_after": (
                "source support, evaluator/model freeze, 2022 checkpoint "
                "selection, and pre-2024 PIVOT action-clock freeze"
            ),
            "static_comparators": _static_comparator_contracts(),
            "carta": _carta_contract(),
            "exact_entry_jaccard_max": 0.10,
            "one_bar_tolerant_jaccard_max": 0.20,
            "twelve_bar_tolerant_jaccard_max": 0.35,
            "six_hour_tolerant_jaccard_intrinsic_family_max": 0.60,
            "absolute_signed_occupancy_pearson_max": 0.40,
            "position_time_jaccard": "report_only",
            "incremental_live_portfolio_occupied_time": "report_only",
            "matching": "maximum-cardinality chronological one-to-one",
            "forbidden_paths": list(FORBIDDEN_COMPARATOR_PATHS),
            "failure_action": "retire PIVOT before 2023 outcomes",
        },
        "eval_2023_gate": {
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_min": 3.0,
            "strict_mdd_pct_max": 15.0,
            "trades_min": 60,
            "each_half_trades_min": 20,
            "each_side_trades_min": 15,
            "both_halves_positive": True,
            "active_execution_months_min": 8,
            "max_execution_month_share": 0.20,
            "max_action_share": 0.90,
            "weekly_clusters_min": 20,
            "weekly_cluster_p_strictly_below": 0.10,
            "mean_gross_underlying_move_bp_min": 20.0,
            "stress_return_positive": True,
            "one_bar_delay_return_positive": True,
            "one_hour_delay": "report_only",
            "each_option_order_audit_return_positive": True,
            "beat_cheap_absolute_return": True,
            "ratio_margin_over_cheap_min": 0.50,
        },
        "sealed_sequence": {
            "open_order": ["2023_eval", "2024", "2025", "2026_ytd_report_only"],
            "each_full_year_uses_eval_gate": True,
            "combined_2024_2025_weekly_cluster_p_strictly_below": 0.05,
            "post_2023_source_extension_requires_prefix_identity": True,
            "leverage_increase_authorized": False,
            "stop_at_first_failure": True,
            "no_parameter_repair": True,
        },
        "strict_sequence": [
            "commit mechanism",
            "commit preregistration manifest and synthetic tests",
            "commit source-only builder",
            "run source-support gate once",
            "retire unchanged on support failure",
            "commit and hash-freeze evaluator, baselines, clocks, model, controls",
            "open only 2020-2022 outcomes",
            "retire before GPU if cheap gate fails",
            "train one SFT and DPO checkpoints 24/48/72/96 if authorized",
            "select on 2022 and freeze pre-2024 actions",
            "run novelty before 2023 outcomes",
            "evaluate 2023 once",
            "open sealed years sequentially only after prior pass",
            "commit every completed unit with hashes and fresh tests",
        ],
    }


def build_manifest() -> dict[str, Any]:
    payload = _core_manifest()
    return {**payload, "manifest_hash": canonical_hash(payload)}


def frozen_dependencies() -> dict[str, str]:
    dependencies = {
        BOUNDARY_DOCUMENT: BOUNDARY_DOCUMENT_SHA256,
        MECHANISM_DOCUMENT: MECHANISM_DOCUMENT_SHA256,
        SOURCE: SOURCE_SHA256,
        SOURCE_MANIFEST: SOURCE_MANIFEST_SHA256,
        SOURCE_AUDIT: SOURCE_AUDIT_SHA256,
        MARKET_SOURCE: MARKET_SOURCE_SHA256,
        MARKET_MANIFEST: MARKET_MANIFEST_SHA256,
        FUNDING_SOURCE: FUNDING_SOURCE_SHA256,
        FUNDING_MANIFEST: FUNDING_MANIFEST_SHA256,
        "results/cchr_live_portfolio_pure_clock_manifest_2026-07-21.json": (
            "6c53ae482cf72bba0f286a47626842bf43070276ff5fe359be718e44864af57d"
        ),
    }
    for comparator in _static_comparator_contracts():
        dependencies[str(comparator["path"])] = str(comparator["sha256"])
    carta = _carta_contract()
    for name in (
        "state_source",
        "policy_source",
        "evaluator_source",
        "support_result",
        "baseline_result",
    ):
        item = carta[name]
        dependencies[str(item["path"])] = str(item["sha256"])
    return dependencies


def validate_frozen_dependencies() -> None:
    for path, expected in frozen_dependencies().items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"PIVOT-72 frozen dependency hash mismatch: {path}: "
                f"{actual} != {expected}"
            )
    for path, expected in (
        (SOURCE, SOURCE_HEADER_SHA256),
        (MARKET_SOURCE, MARKET_HEADER_SHA256),
        (FUNDING_SOURCE, FUNDING_HEADER_SHA256),
    ):
        actual = sha256_csv_header(path)
        if actual != expected:
            raise RuntimeError(
                f"PIVOT-72 frozen header hash mismatch: {path}: "
                f"{actual} != {expected}"
            )
    if not set(SOURCE_ALLOWLIST).issubset(csv_header(SOURCE)):
        raise RuntimeError("PIVOT-72 source allowlist differs from frozen header")
    live_manifest = json.loads(
        Path(
            "results/cchr_live_portfolio_pure_clock_manifest_2026-07-21.json"
        ).read_text(encoding="utf-8")
    )
    if (
        live_manifest.get("clock", {}).get("rows") != 440
        or live_manifest.get("outcome_boundary", {}).get("source_end_exclusive")
        != "2024-01-01T00:00:00Z"
        or live_manifest.get("outcome_boundary", {}).get("post_2023_rows_loaded")
        != 0
    ):
        raise RuntimeError("PIVOT-72 live comparator pre-2024 boundary changed")


def validate_manifest(payload: Mapping[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("PIVOT-72 manifest hash mismatch")
    expected = _core_manifest()
    if core != expected:
        raise RuntimeError("PIVOT-72 manifest core differs from code")


def _canonical_manifest_text(payload: Mapping[str, Any] | None = None) -> str:
    frozen = build_manifest() if payload is None else dict(payload)
    validate_manifest(frozen)
    return (
        json.dumps(
            frozen,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def write_once(path: str | Path, payload: Mapping[str, Any]) -> str:
    validate_manifest(payload)
    validate_frozen_dependencies()
    output = Path(path)
    text = _canonical_manifest_text(payload)
    if output.exists():
        existing = output.read_text(encoding="utf-8")
        if existing != text:
            try:
                stored = json.loads(existing)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"PIVOT-72 existing manifest is invalid JSON: {output}"
                ) from exc
            if stored != dict(payload):
                raise RuntimeError(
                    f"PIVOT-72 existing manifest hash mismatch: {output}"
                )
            raise RuntimeError(
                f"PIVOT-72 noncanonical existing manifest: {output}"
            )
        validate_manifest(json.loads(existing))
        return "verified_existing"
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze PIVOT-72 without decoding source or outcome rows"
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_manifest()
    status = write_once(args.output, payload)
    print(
        json.dumps(
            {
                "status": status,
                "output": args.output,
                "manifest_hash": payload["manifest_hash"],
                "source_rows_decoded": payload["source_rows_decoded"],
                "market_value_rows_decoded": payload["market_value_rows_decoded"],
                "funding_value_rows_decoded": payload["funding_value_rows_decoded"],
                "comparator_rows_decoded": payload["comparator_rows_decoded"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
