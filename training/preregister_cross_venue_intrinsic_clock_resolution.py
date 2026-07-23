"""Freeze CVICR-72 before decoding candidate incidence or market outcomes."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = (
    "results/cross_venue_intrinsic_clock_resolution_"
    "preregistration_2026-07-24.json"
)
BOUNDARY_DOCUMENT = (
    "docs/cross-venue-intrinsic-clock-resolution-boundary-2026-07-24.md"
)
BOUNDARY_DOCUMENT_SHA256 = (
    "9643d3178af71d0e92a0dbe8c3c4f09f7232c1117406609a7467cce793993c3b"
)
MECHANISM_DOCUMENT = (
    "docs/cross-venue-intrinsic-clock-resolution-"
    "mechanism-decision-2026-07-24.md"
)
MECHANISM_DOCUMENT_SHA256 = (
    "25d238e6d5718854ea35ecde3f37720869898e622e2610f46337da2daa315264"
)
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
SOURCE_AUDIT = (
    "results/binance_cross_venue_minute_leadership_audit_2026-07-14.json"
)
SOURCE_AUDIT_SHA256 = (
    "ffe0124ac9c5c0c3f1d1c284b672618cf910dc16cae36e65c1efe79710f039af"
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

CONTROL_ORDER = (
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
)

SCORE_BEARING_CONTROLS = (
    "gap_only",
    "initial_conflict_only",
    "late_alignment_only",
    "no_leader_persistence",
    "no_gap_tail",
    "fixed_expected_time_clocks",
    "stale_laggard_flow_24h",
)


@dataclass(frozen=True)
class Policy:
    policy_id: str = "CVICR-72"
    reference_calendar_days: int = 28
    reference_complete_days_min: int = 21
    intrinsic_volume_fraction: float = 0.50
    latest_anchor_start_minute_utc: int = 17 * 60 + 50
    gap_reference_pairs: int = 180
    gap_reference_pairs_min: int = 90
    gap_quantile: float = 0.60
    signal_availability_bars: int = 1
    computation_buffer_bars: int = 1
    entry_delay_bars_from_late_anchor: int = 2
    hold_bars: int = 72
    leverage: float = 0.50
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010
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


def csv_header_bytes(path: str | Path) -> bytes:
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rb") as handle:
        header = handle.readline()
    if not header.endswith(b"\n") or b"\n" in header[:-1]:
        raise RuntimeError(f"CVICR-72 CSV header is not one LF line: {path}")
    return header


def csv_header(path: str | Path) -> list[str]:
    header = csv_header_bytes(path).decode("utf-8")
    return next(csv.reader([header.rstrip("\n")]))


def sha256_csv_header(path: str | Path) -> str:
    return hashlib.sha256(csv_header_bytes(path)).hexdigest()


def _comparator_contracts() -> list[dict[str, Any]]:
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
            "id": "CATCH-12",
            "path": (
                "results/cash_auction_transfer_catchup_handoff_"
                "clock_2026-07-14.csv"
            ),
            "sha256": (
                "066bf8e08267a043cc191eb436f0aa33105ab948de9f9f1edfde4d9c30de46d1"
            ),
            "header": standard_header,
            "header_sha256": (
                "3211ca0e50e607f39ef0c8ee72329828a267cdab15de029002b0b5beaeb2032c"
            ),
            "entry_column": "entry_date",
            "exit_column": "exit_date",
            "side_column": "side",
            "side_encoding": {"1": 1, "-1": -1},
            "group_column": "branch",
            "selected_groups": ["catch12"],
            "declared_coverage": ["2020-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "six_hour_tolerant_gate": False,
        },
        {
            "id": "CLASP-24",
            "path": (
                "results/cash_late_arrival_spillover_propagation_"
                "clock_2026-07-14.csv"
            ),
            "sha256": (
                "e166f4bd24afd5a2f129bcc26393ad4293ad0bc5792686b3b0fc4a805d53f9d5"
            ),
            "header": standard_header,
            "header_sha256": (
                "3211ca0e50e607f39ef0c8ee72329828a267cdab15de029002b0b5beaeb2032c"
            ),
            "entry_column": "entry_date",
            "exit_column": "exit_date",
            "side_column": "side",
            "side_encoding": {"1": 1, "-1": -1},
            "group_column": "branch",
            "selected_groups": ["clasp24"],
            "declared_coverage": ["2020-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "six_hour_tolerant_gate": False,
        },
        {
            "id": "LURI-48",
            "path": (
                "results/leveraged_um_inventory_release_handoff_"
                "clock_2026-07-14.csv"
            ),
            "sha256": (
                "50765cfed0c3ec6a0d1df18857c4e0a3e574d1aa449538c9b89cfac1fff67095"
            ),
            "header": standard_header,
            "header_sha256": (
                "3211ca0e50e607f39ef0c8ee72329828a267cdab15de029002b0b5beaeb2032c"
            ),
            "entry_column": "entry_date",
            "exit_column": "exit_date",
            "side_column": "side",
            "side_encoding": {"1": 1, "-1": -1},
            "group_column": "branch",
            "selected_groups": ["luri48"],
            "declared_coverage": ["2020-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "six_hour_tolerant_gate": False,
        },
        {
            "id": "CVTT-V01-V04",
            "path": (
                "data/cross_venue_temporal_torsion_v2_"
                "support_clocks_2020_2022.csv.gz"
            ),
            "sha256": (
                "8f933b9d387fbcb764645a7002a5eefa9ee159c9c1ce7e007dca0dc4c16ebe33"
            ),
            "header": [
                "policy_id",
                "route",
                "side",
                "hold_bars",
                "signal_date",
                "signal_row",
                "entry_date",
            ],
            "header_sha256": (
                "a182548526587fa060072ca6fc2ab284167d12e9b56dcc1c6194c5d5f65cc683"
            ),
            "entry_column": "entry_date",
            "exit_column": None,
            "exit_derivation": "entry_date + hold_bars*5m",
            "side_column": "side",
            "side_encoding": {"1": 1, "-1": -1},
            "group_column": "policy_id",
            "selected_groups": ["V01", "V02", "V03", "V04"],
            "declared_coverage": ["2020-01-01T00:00:00Z", "2023-01-01T00:00:00Z"],
            "selection_absence_is_not_failure": True,
            "six_hour_tolerant_gate": False,
        },
        {
            "id": "IVLIR-primary",
            "path": (
                "data/intrinsic_volume_latent_impact_relay_"
                "clocks_2020_2023.csv.gz"
            ),
            "sha256": (
                "523f24a0d955fe99cfb86c62078532c5fc9091234e6669ab9acff2a8f3367788"
            ),
            "header": intrinsic_header,
            "header_sha256": (
                "0ad7d7a39f7d772de30d2c47056efd3c9b7740561eea9a1b69007b4870d5d495"
            ),
            "entry_column": "entry_time",
            "exit_column": "exit_time",
            "side_column": "side",
            "side_encoding": {"LONG": 1, "SHORT": -1},
            "group_column": "clock_name",
            "selected_groups": ["primary"],
            "declared_coverage": ["2020-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "six_hour_tolerant_gate": True,
        },
        {
            "id": "IVFHR-primary-and-any-handoff",
            "path": (
                "data/intrinsic_volume_flow_handoff_relay_"
                "clocks_2020_2023.csv.gz"
            ),
            "sha256": (
                "ab12762dec9a93d41c293766e46dfc80ade81914fb32753a5923faa6437c338e"
            ),
            "header": intrinsic_header,
            "header_sha256": (
                "0ad7d7a39f7d772de30d2c47056efd3c9b7740561eea9a1b69007b4870d5d495"
            ),
            "entry_column": "entry_time",
            "exit_column": "exit_time",
            "side_column": "side",
            "side_encoding": {"LONG": 1, "SHORT": -1},
            "group_column": "clock_name",
            "selected_groups": ["primary", "any_handoff"],
            "compare_groups_separately": True,
            "declared_coverage": ["2020-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "six_hour_tolerant_gate": True,
        },
        {
            "id": "IVPLH-primary",
            "path": (
                "data/intrinsic_volume_price_lag_handoff_"
                "clocks_2020_2023.csv.gz"
            ),
            "sha256": (
                "2efca3b44b0512a9423da90171f43babcadec2316dc6148796f3e61f98138e80"
            ),
            "header": [
                "control",
                "signal_id",
                "source_day",
                "decision_time",
                "entry_time",
                "exit_time",
                "side",
            ],
            "header_sha256": (
                "d5ae2566140aca706f84f916965352daed3aad058e4abcc9614f19f4950f0bbd"
            ),
            "entry_column": "entry_time",
            "exit_column": "exit_time",
            "side_column": "side",
            "side_encoding": {"LONG": 1, "SHORT": -1},
            "group_column": "control",
            "selected_groups": ["primary"],
            "declared_coverage": ["2021-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "six_hour_tolerant_gate": True,
        },
    ]


def _core_manifest() -> dict[str, Any]:
    policy = Policy()
    return {
        "protocol_version": "cross_venue_intrinsic_clock_resolution_v1",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "source_rows_decoded": False,
        "comparator_rows_decoded": False,
        "policy": asdict(policy),
        "frozen_documents": {
            "boundary": {
                "path": BOUNDARY_DOCUMENT,
                "sha256": BOUNDARY_DOCUMENT_SHA256,
                "commit": "dc90f0d9e5577f393e6de488b53483efe7b7972c",
            },
            "mechanism": {
                "path": MECHANISM_DOCUMENT,
                "sha256": MECHANISM_DOCUMENT_SHA256,
                "commit": "cb6aa0544b4fc774e5ee62cb5acf55298eb492ff",
            },
        },
        "research_history_boundary": {
            "repo_wide_btc_history_seen": True,
            "predecessor_cross_venue_and_intrinsic_outcomes_seen": True,
            "exact_cvicr_anchor_or_candidate_incidence_seen": False,
            "exact_cvicr_post_entry_outcomes_seen": False,
            "post_2023_cvicr_source_seen": False,
            "claim_scope": (
                "candidate-level frozen test, not globally pristine history"
            ),
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
            "interval": ["2020-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "bar_interval": "5min",
            "allowlist": list(SOURCE_ALLOWLIST),
            "loader": "pandas.read_csv(usecols=allowlist); no load-and-drop",
            "required_availability": (
                "feature_available_time_utc == trade_earliest_time_utc == date+5m"
            ),
            "reference_day": (
                "all 288 rows source-complete and numerically valid"
            ),
            "current_day": (
                "require only causal prefix through A_late plus complete buffer row; "
                "never inspect later-day completeness to accept/reject a candidate"
            ),
            "missing_policy": (
                "missing reference day stays missing in exact 28-calendar-day "
                "window; current prefix defect cancels current day; never impute"
            ),
        },
        "causal_clock_contract": {
            "expected_volume": (
                "per venue median of complete daily quote notional in exact prior "
                "28 calendar positions, current excluded, minimum 21"
            ),
            "target": "0.50*expected_volume per venue",
            "anchor": (
                "first completed 5m bar reaching own venue target; both anchor "
                "starts <=17:50 UTC; exact tie ineligible"
            ),
            "leader": "venue with earlier anchor; laggard is the other venue",
            "gap": "positive integer (A_late-A_early)/5m",
            "gap_reference": (
                "previous at most 180 valid paired-anchor days, current excluded, "
                "minimum 90; pooled linear q60"
            ),
            "flow": (
                "cumulative signed_quote_notional / cumulative quote_notional "
                "from UTC-day start through requested completed bar"
            ),
            "initial_conflict": (
                "d=sign(leader flow at A_early) is nonzero and laggard early sign=-d"
            ),
            "resolution": (
                "leader and laggard cumulative flow signs at A_late both equal d"
            ),
            "side": "fixed d",
            "price_basis_or_return_input": None,
            "future_bar_used_by_signal": False,
        },
        "execution_contract": {
            "causal_origin": "A_early",
            "resolution_bar_start": "A_late",
            "signal_available_time": "A_late+5m",
            "buffer": "[A_late+5m,A_late+10m) must complete",
            "decision_order_time": "A_late+10m",
            "entry": "USD-M BTCUSDT open at A_late+10m",
            "exit": "entry+72*5m scheduled open",
            "clock_fields": [
                "source_day",
                "causal_origin",
                "resolution_bar_start",
                "signal_available_time",
                "decision_order_time",
                "entry_time",
                "exit_time",
                "side",
                "leader_venue",
            ],
            "comparator_timestamp": "entry_time",
            "global_nonoverlap_before_split": True,
            "crossing_candidate_keeps_global_reservation": True,
            "split_containment": (
                "origin, resolution, availability, buffer, decision, entry, "
                "full held path, and exit in one half-open split"
            ),
            "later_source_defect_cannot_cancel_entered_position": True,
            "stop_or_take_profit": None,
        },
        "source_only_controls": {
            "ordered": list(CONTROL_ORDER),
            "score_bearing": list(SCORE_BEARING_CONTROLS),
            "all_emit_side_in": [-1, 1],
            "zero_or_nonfinite_side": "ineligible",
            "fixed_time_tie": "ineligible",
            "missing_stale_prefix": "ineligible",
            "fixed_expected_time_gap_threshold": (
                "current day's causal primary paired-anchor q60 threshold; "
                "no second fitted gap distribution"
            ),
            "random_side": (
                "SHA256('CVICR-72|'+entry_time_utc), LONG iff first byte<128"
            ),
            "independent_clocks_reserve_independently": True,
        },
        "source_support_gate": {
            "train": ["2020-01-01T00:00:00Z", "2023-01-01T00:00:00Z"],
            "selection": ["2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "train_events_min": 75,
            "each_train_year_events_min": 20,
            "train_active_months_min": 24,
            "train_each_side_share_min": 0.20,
            "train_each_leader_share_min": 0.15,
            "train_max_month_share": 0.15,
            "train_max_quarter_share": 0.30,
            "train_max_entry_gap_days": 90.0,
            "train_max_same_side_run": 10,
            "train_max_same_leader_run": 12,
            "selection_events_min": 24,
            "selection_each_half_events_min": 10,
            "selection_each_quarter_events_min": 4,
            "selection_active_months_min": 8,
            "selection_each_side_share_min": 0.20,
            "selection_each_leader_share_min": 0.15,
            "selection_max_month_share": 0.20,
            "selection_max_entry_gap_days": 75.0,
            "selection_max_same_side_run": 8,
            "selection_max_same_leader_run": 10,
            "mechanism_selectivity": {
                "primary_over_gap_only_max": 0.40,
                "primary_over_initial_conflict_only_max": 0.70,
                "primary_over_late_alignment_only_max": 0.70,
                "primary_over_no_gap_tail_max": 0.70,
                "fixed_expected_time_entry_jaccard_max": 0.10,
                "stale_laggard_flow_entry_jaccard_max": 0.05,
                "clock_basis": (
                    "own globally reserved split-contained accepted entries"
                ),
                "undefined_or_empty_required_control": "one and fail",
            },
            "failure_action": "retire CVICR-72 unchanged before outcomes",
        },
        "novelty_contract": {
            "comparators": _comparator_contracts(),
            "common_coverage": (
                "intersection of declared comparator coverage and CVICR source; "
                "CVTT train only"
            ),
            "exact_entry_jaccard_max": 0.10,
            "one_bar_tolerant_jaccard_max": 0.20,
            "twelve_bar_tolerant_jaccard_max": 0.35,
            "six_hour_tolerant_jaccard_intrinsic_family_max": 0.60,
            "tolerant_matching": (
                "sorted two-pointer maximum-cardinality one-to-one matching; "
                "advance earlier entry outside tolerance, otherwise match and "
                "advance both; matched/(n_candidate+n_comparator-matched)"
            ),
            "absolute_signed_occupancy_pearson_max": 0.40,
            "position_time_jaccard": "report_only",
            "empty_required_extraction_or_nonfinite_correlation": "one and fail",
            "failure_action": "retire CVICR-72 unchanged before outcomes",
        },
        "economic_contract": {
            "instrument": "Binance USD-M BTCUSDT perpetual",
            "leverage": policy.leverage,
            "base_cost_notional_per_side": policy.base_cost_notional_per_side,
            "base_account_cost_per_side": 0.0003,
            "stress_replaces_base": True,
            "stress_cost_notional_per_side": policy.stress_cost_notional_per_side,
            "stress_account_cost_per_side": 0.0005,
            "funding_interval": "entry_time <= funding_time < exit_time",
            "cagr": "full declared calendar including warmup and idle cash",
            "strict_mdd": (
                "global/pre-entry HWM; entry cost; each held-bar favorable then "
                "adverse; funding debit ordering; virtual adverse exit cost; "
                "scheduled-open exit and cost; exclude later exit-bar OHLC"
            ),
            "cluster_signflip": {
                "week": "UTC ISO entry week, Monday 00:00",
                "return": "net account trade return after base costs and funding",
                "statistic": "sum(trade_returns)/N",
                "draws": policy.cluster_signflip_draws,
                "seed": policy.random_seed,
                "rng_reset": "independently per split and control",
                "null": "one Rademacher sign per nonempty week",
                "p": "(1+count(null>=observed))/(draws+1)",
                "empty": 1.0,
            },
        },
        "strict_sequence": {
            "phase_1": "source support, controls, and novelty only",
            "phase_2": "commit and hash-freeze strict evaluator",
            "stages": [
                ["train", "2020-01-01T00:00:00Z", "2023-01-01T00:00:00Z"],
                ["selection", "2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
                ["test_2024", "2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
                ["eval_2025", "2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
                ["final_2026_ytd", "2026-01-01T00:00:00Z", None],
            ],
            "post_2023_source_extension_requires_separate_audit": True,
            "stop_at_first_failure": True,
            "no_parameter_repair": True,
        },
        "economic_gates": {
            "train_and_selection_absolute_return_positive": True,
            "train_and_selection_cagr_to_strict_mdd_min": 3.0,
            "train_and_selection_strict_mdd_pct_max": 15.0,
            "mean_gross_underlying_bp_min": 30.0,
            "stress_absolute_return_positive": True,
            "one_bar_delay_absolute_return_positive": True,
            "weekly_cluster_signflip_p_max": 0.10,
            "long_and_short_sleeve_return_positive": True,
            "train_each_year_return_positive": True,
            "selection_each_half_return_positive": True,
            "selection_trades_min": 24,
            "selection_each_half_trades_min": 10,
            "score_bearing_controls": list(SCORE_BEARING_CONTROLS),
            "primary_ratio_margin_over_controls_min": 0.50,
            "placebos_may_not_fully_qualify": [
                "exact_direction_flip",
                "deterministic_random_side",
                "one_bar_execution_delay",
                "one_hour_execution_delay",
            ],
            "test_and_eval": {
                "absolute_return_positive": True,
                "cagr_to_strict_mdd_min": 3.0,
                "strict_mdd_pct_max": 15.0,
                "trades_min": 20,
                "stress_absolute_return_positive": True,
                "one_bar_delay_absolute_return_positive": True,
                "weekly_cluster_signflip_p_max": 0.10,
            },
            "combined_2024_2025_cluster_p_max": 0.05,
        },
        "live_parity": {
            "sources": [
                "official Binance Spot BTCUSDT one-minute klines",
                "official Binance USD-M BTCUSDT one-minute klines",
            ],
            "persist_raw_minute_rows": True,
            "required_minute_checks": [
                "exact UTC minute boundary",
                "unique monotonic minute identity",
                "positive quote activity",
                "finite taker-buy quote within total quote activity",
            ],
            "five_minute_finalization_before_accumulation": True,
            "persist_receipt_and_finalization_timestamps": True,
            "maintain_separate_venue_quote_and_signed_flow_sums": True,
            "reference_window": (
                "exact prior 28 calendar days with at least 21 complete days"
            ),
            "current_prefix_defect_action": "cancel current day and remain flat",
            "computation_buffer_bars": policy.computation_buffer_bars,
            "fail_flat_on": [
                "missing bar",
                "reordered bar",
                "duplicate bar",
                "clock drift",
                "late data",
                "source divergence",
                "anchor computed before finalization",
            ],
            "rest_repair": (
                "may restore storage for later reference windows; may not backdate "
                "a missed decision or resurrect a canceled historical trade"
            ),
        },
        "llm_boundary": {
            "activation_requires_unchanged_train_selection_and_oos_pass": True,
            "action_space": ["TRADE_FIXED_SIDE", "ABSTAIN"],
            "allowed_tokens": [
                "leader_venue",
                "gap_rank_bucket",
                "early_conflict_relation",
                "leader_persistence_relation",
                "laggard_resolution_relation",
                "causal_flow_strength_buckets",
                "source_validity_state",
                "current_position",
            ],
            "forbidden": [
                "timestamp",
                "row_identifier",
                "raw_price",
                "future_label",
                "sealed_reward",
                "side_choice",
                "hold_choice",
                "leverage_increase",
            ],
        },
        "stopping_rule": (
            "first failed source, support, selectivity, novelty, or economic gate "
            "retires CVICR-72 unchanged"
        ),
    }


def build_manifest() -> dict[str, Any]:
    core = _core_manifest()
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_manifest(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError("CVICR-72 preregistration hash mismatch")
    if core != _core_manifest():
        raise RuntimeError("CVICR-72 preregistration core differs from code")


def frozen_dependencies() -> dict[str, str]:
    dependencies = {
        BOUNDARY_DOCUMENT: BOUNDARY_DOCUMENT_SHA256,
        MECHANISM_DOCUMENT: MECHANISM_DOCUMENT_SHA256,
        SOURCE: SOURCE_SHA256,
        SOURCE_MANIFEST: SOURCE_MANIFEST_SHA256,
        SOURCE_AUDIT: SOURCE_AUDIT_SHA256,
    }
    for item in _comparator_contracts():
        dependencies[item["path"]] = item["sha256"]
    return dependencies


def validate_frozen_dependencies() -> None:
    for path, expected in frozen_dependencies().items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"CVICR-72 frozen dependency changed: {path}")
    if sha256_csv_header(SOURCE) != SOURCE_HEADER_SHA256:
        raise RuntimeError("CVICR-72 source header changed")
    header = csv_header(SOURCE)
    missing = sorted(set(SOURCE_ALLOWLIST).difference(header))
    if missing:
        raise RuntimeError(f"CVICR-72 source allowlist missing columns: {missing}")
    for item in _comparator_contracts():
        if sha256_csv_header(item["path"]) != item["header_sha256"]:
            raise RuntimeError(f"CVICR-72 comparator header changed: {item['id']}")


def _canonical_manifest_text() -> str:
    return json.dumps(
        build_manifest(),
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"


def write_once(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path)
    validate_frozen_dependencies()
    validate_manifest(payload)
    canonical_text = _canonical_manifest_text()
    output.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            handle.write(canonical_text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o644)
        try:
            os.link(temporary, output)
        except FileExistsError:
            if (
                output.is_symlink()
                or not output.is_file()
                or output.read_text(encoding="utf-8") != canonical_text
            ):
                raise RuntimeError(
                    "refusing noncanonical existing CVICR-72 preregistration"
                )
            return "verified_existing"
        return "created"
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_manifest()
    status = write_once(args.output, payload)
    print(
        json.dumps(
            {
                "status": status,
                "output": str(args.output),
                "manifest_hash": payload["manifest_hash"],
                "outcomes_opened": payload["outcomes_opened"],
                "source_incidence_opened": payload["source_incidence_opened"],
                "source_rows_decoded": payload["source_rows_decoded"],
                "comparator_rows_decoded": payload["comparator_rows_decoded"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
