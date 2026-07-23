"""Freeze RQHR-72 before synthetic nulls, source incidence, or BTC outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


POLICY_ID = "RQHR-72"
PROTOCOL_VERSION = "radial_quote_handoff_relay_preregistration_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path("training/preregister_radial_quote_handoff_relay.py")
MECHANISM_DECISION = Path(
    "docs/radial-quote-handoff-relay-mechanism-decision-2026-07-23.md"
)
MECHANISM_DECISION_SHA256 = (
    "0b42b726df121b265c5e6780db24098b44d8d6ff11cd3967cf05ca8ed0b38ea6"
)
COMMON_WINDOW_POLICY = Path(
    "docs/novelty-comparator-common-window-policy-2026-07-23.md"
)
COMMON_WINDOW_POLICY_SHA256 = (
    "928bce6e04fb34001478b4b4ea84156580b661c88a0f0338065a891c009bd580"
)
DEFAULT_OUTPUT = Path(
    "results/radial_quote_handoff_relay_preregistration_2026-07-23.json"
)

SOURCE_PANEL = Path(
    "data/binance_um_book_centroid_btcusdt_2023/"
    "BTCUSDT_um_book_centroid_skew_5m_2023.csv.gz"
)
SOURCE_PANEL_SHA256 = (
    "c4053ce27d28bebda4137349192b1a940360231469f63edc32bacabb2ce54131"
)
SOURCE_MANIFEST = Path(
    "results/binance_um_book_centroid_btcusdt_2023_manifest.json"
)
SOURCE_MANIFEST_SHA256 = (
    "d8237c4562d33c12eff162776f723cc5fc94649b69d26a6230e16fc38c52bba1"
)
SOURCE_BUILDER = Path("training/build_binance_um_book_centroid_2023.py")
SOURCE_BUILDER_SHA256 = (
    "6021a1ee140500350e8b6bc0e8dae5ca32a84db39039c21d809ca798909a5c24"
)
RNCM_PREREGISTRATION = Path(
    "training/preregister_residual_notional_centroid_migration.py"
)
RNCM_PREREGISTRATION_SHA256 = (
    "733ef4c3aaa823f19c8fe9303d3405def0c86f593c35bb2556a69edc3f67ad6f"
)

RQHR_COLUMNS = (
    "date",
    "skew_2_net",
    "skew_2_path",
    "skew_2_efficiency",
    "skew_3_net",
    "skew_3_path",
    "skew_3_efficiency",
    "skew_4_net",
    "skew_4_path",
    "skew_4_efficiency",
    "skew_5_net",
    "skew_5_path",
    "skew_5_efficiency",
    "source_complete",
    "source_available_at",
)

HISTORY_BINDINGS: tuple[Mapping[str, Any], ...] = (
    {
        "name": "source_axis_decision",
        "path": Path("docs/btc-alpha-source-axis-decision-2026-07-20.md"),
        "sha256": (
            "9a46c534c932efc4c38fa0a0ad168e40f33803527de12b80ac7a49f550c7dadd"
        ),
    },
    {
        "name": "source_build_audit",
        "path": Path("docs/rncm-2023-source-build-audit-2026-07-20.md"),
        "sha256": (
            "bf19382d550bfa1c4bcb6dfec080f4f0a57c64cd256c920dce8dc8158aee4ddb"
        ),
    },
    {
        "name": "rncm_preregistration",
        "path": RNCM_PREREGISTRATION,
        "sha256": RNCM_PREREGISTRATION_SHA256,
    },
    {
        "name": "rncm_support_result",
        "path": Path(
            "results/residual_notional_centroid_migration_support_2026-07-20.json"
        ),
        "sha256": (
            "887c532eb3163cfac47eb9fc2956326f02491b2890e4c0231e084807978577dc"
        ),
    },
    {
        "name": "rncm_support_rejection",
        "path": Path(
            "docs/residual-notional-centroid-migration-support-rejection-2026-07-20.md"
        ),
        "sha256": (
            "34764817293d8914d4e4aa3d12d26d998abbfba23f3506d5c26dcbbd85e9c343"
        ),
    },
)

COMPARATOR_SPECS: tuple[Mapping[str, Any], ...] = (
    {
        "group": "ccbvfr:primary",
        "path": Path(
            "results/cross_collateral_book_validated_flow_rejection_"
            "event_clock_2026-07-18.json"
        ),
        "sha256": (
            "79b4838ae634efcff705e028a0ddff8b75d28d79180e3ac89f54b9cab7e5005f"
        ),
        "expected_raw_rows": 144,
        "canonical_clock_sha256": (
            "d2cdcad8f57867722c220e32029d0ccbf1f1aa511e5ae590cf43411a588af4bd"
        ),
        "protocol": "CBFR-72 canonical outcome-blind event-clock freeze",
        "closed_flags": {
            "post_entry_outcomes_opened": False,
            "entry_or_later_ohlc_loaded": False,
        },
        "selection_end_exclusive": "2024-01-01 00:00:00",
        "artifact_canonical_fields": None,
        "producer": Path(
            "training/preregister_cross_collateral_book_validated_flow_rejection.py"
        ),
        "producer_sha256": (
            "004fa71b1951eff58eca592863cf7ad09e0e36e4749a3e611ce299e1ac3d601f"
        ),
        "parser": "all embedded event positions, displayed dates, interval, side",
        "canonical_projection": "full ordered embedded event dictionaries",
    },
    {
        "group": "pdf10:primary",
        "path": Path(
            "results/cross_collateral_liquidity_credibility_fracture_"
            "event_clock_2026-07-14.json"
        ),
        "sha256": (
            "ab8209308619b97880277b95fcc1a2f825b050a603e24b3e2125ddd5bfb226f8"
        ),
        "expected_raw_rows": 591,
        "canonical_clock_sha256": (
            "ce1c6ec42434874d97c6b6034f51a73771b27e314da6d37a4f44b0563e6972e2"
        ),
        "protocol": "PDF-10 canonical event-clock freeze",
        "closed_flags": {
            "outcomes_opened_for_pdf10": False,
            "price_or_return_loaded": False,
        },
        "selection_end_exclusive": None,
        "artifact_canonical_fields": [
            "signal_position",
            "entry_position",
            "exit_position",
            "side",
            "branch",
            "hold_bars",
        ],
        "producer": Path(
            "training/preregister_cross_collateral_liquidity_credibility_fracture.py"
        ),
        "producer_sha256": (
            "8947050c990b5638f6d8b2e952f252289ddef6c92f85fb13f75001fe721e6e28"
        ),
        "dependency": Path(
            "results/cross_collateral_liquidity_credibility_fracture_"
            "support_2026-07-14.json"
        ),
        "dependency_sha256": (
            "9a3001db640ec8041d885645d33f11dd6075276685eb22f8ae3c618363d3099a"
        ),
        "parser": "replay exact PDF-10 support clock and validate every row",
        "canonical_projection": "ordered signal_position and numeric side",
    },
    {
        "group": "crrc:primary",
        "path": Path(
            "results/cross_venue_radial_refill_compression_"
            "event_clock_2026-07-17.json"
        ),
        "sha256": (
            "09d2ca954c5c4d06b981575c6b0f0e4dc6b49d8a693da418f3f26e5cc454c835"
        ),
        "expected_raw_rows": 156,
        "canonical_clock_sha256": (
            "81e09e3d1d5592f12ce1994077efa279ebf1de4c29a6f5a144060d16ee6b2e9f"
        ),
        "protocol": "CRRC-72 canonical outcome-blind event-clock freeze",
        "closed_flags": {
            "outcomes_opened": False,
            "price_funding_return_or_equity_loaded": False,
        },
        "selection_end_exclusive": "2024-01-01 00:00:00",
        "artifact_canonical_fields": [
            "signal_position",
            "entry_position",
            "exit_position",
            "side",
            "hold_bars",
        ],
        "producer": Path(
            "training/qualify_cross_venue_radial_refill_compression.py"
        ),
        "producer_sha256": (
            "96372733a597ca486b52292480ceacde631056054b2d914aa9180024218fa0e7"
        ),
        "parser": "all embedded event positions, displayed dates, interval, side",
        "canonical_projection": (
            "ordered signal_position, entry_position, exit_position, side, hold_bars"
        ),
    },
)

PRIOR_RESEARCH_DISCLOSURE: Mapping[str, Any] = {
    "source_family_previously_tested": True,
    "source_family_pristine_claim": False,
    "rncm_median_and_quote_center_values_previously_opened": True,
    "rncm_support_incidence_previously_opened": True,
    "rncm_nonoverlap_counts_by_quantile": {
        "0.995": 5,
        "0.990": 16,
        "0.985": 31,
        "0.975": 39,
    },
    "rqhr_net_path_efficiency_values_opened": False,
    "rqhr_features_arms_confirmations_or_events_opened": False,
    "rqhr_comparator_overlap_opened": False,
    "average_quote_source_market_outcomes_opened": False,
    "prior_comparator_timing_rows_partially_opened_for_validation": True,
}

EXPECTED_OUTCOME_BOUNDARY: Mapping[str, Any] = {
    "mechanism_bytes_hashed": True,
    "common_window_policy_bytes_hashed": True,
    "source_panel_bytes_hashed": True,
    "source_manifest_bytes_hashed": True,
    "source_manifest_metadata_parsed": False,
    "source_builder_bytes_hashed": True,
    "source_value_rows_read": 0,
    "rqhr_columns_read": 0,
    "rqhr_features_computed": 0,
    "synthetic_nulls_run": 0,
    "rqhr_arms_or_terminals_derived": 0,
    "rqhr_events_derived": 0,
    "history_artifact_bytes_hashed": True,
    "history_value_rows_read": 0,
    "comparator_and_producer_bytes_hashed": True,
    "comparator_value_rows_read": 0,
    "btc_market_rows_read": 0,
    "funding_rows_read": 0,
    "future_return_rows_read": 0,
    "pnl_cagr_mdd_opened": False,
    "network_calls": 0,
    "subprocess_calls": 0,
}
STATIC_TEST_OUTCOME_BOUNDARY: Mapping[str, Any] = {
    **EXPECTED_OUTCOME_BOUNDARY,
    "mechanism_bytes_hashed": False,
    "common_window_policy_bytes_hashed": False,
    "source_panel_bytes_hashed": False,
    "source_manifest_bytes_hashed": False,
    "source_builder_bytes_hashed": False,
    "history_artifact_bytes_hashed": False,
    "comparator_and_producer_bytes_hashed": False,
}


@dataclass(frozen=True)
class Config:
    output: str = str(DEFAULT_OUTPUT)


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        raise RuntimeError("RQHR path must be repository-relative")
    resolved = (REPOSITORY_ROOT / candidate).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("RQHR path must remain repository-relative") from exc
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
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def policy_payload() -> dict[str, Any]:
    return {
        "candidate": POLICY_ID,
        "singleton": True,
        "research_status": "source_family_successor_exact_columns_incidence_blind",
        "economic_hypothesis": {
            "mechanism": "near-radii average-quote impulse handed to far radii",
            "positive_relay": "ask-side outward repricing; LONG",
            "negative_relay": "bid-side outward repricing; SHORT",
            "rncm_in_place_repair": False,
            "independent_source_family_claim": False,
        },
        "contamination": dict(PRIOR_RESEARCH_DISCLOSURE),
        "source": {
            "panel": str(SOURCE_PANEL),
            "manifest": str(SOURCE_MANIFEST),
            "builder": str(SOURCE_BUILDER),
            "exact_columns": list(RQHR_COLUMNS),
            "forbidden_columns": [
                "center_quote_median",
                "skew_2_median",
                "skew_3_median",
                "skew_4_median",
                "skew_5_median",
            ],
            "grid_start": "2023-01-01T00:00:00Z",
            "grid_end_exclusive": "2024-01-01T00:00:00Z",
            "grid_rows": 105_120,
            "bar_minutes": 5,
            "availability": "date+5 elapsed minutes",
            "incomplete_row_action": "no value; break race continuity",
            "imputation_forward_fill_backward_fill": False,
            "post_2023_rows_allowed_during_support": False,
            "live_parity_is_separate_production_gate": True,
        },
        "source_algebra": {
            "absolute_tolerance": "5e-12",
            "path_gte_abs_net": True,
            "zero_path_implies_zero_net_and_efficiency": True,
            "positive_path_efficiency": "abs(net)/path",
            "efficiency_closed_interval": ["0", "1"],
            "repair_clipping_winsorization_or_residualization": False,
        },
        "arithmetic": {
            "real_panel": "exact decimal/rational after algebra tolerance check",
            "nearest_rank": "sort exact values; index ceil(q*N)-1",
            "binary_float_real_panel_ranking": False,
            "binary_float_only_for_hash_bound_synthetic_replay": True,
        },
        "features": {
            "near_sign": "common nonzero sign(skew_2_net,skew_3_net), else 0",
            "far_sign": "common nonzero sign(skew_4_net,skew_5_net), else 0",
            "near_intensity": "(abs(skew_2_net)+abs(skew_3_net))/2",
            "far_intensity": "(abs(skew_4_net)+abs(skew_5_net))/2",
            "near_efficiency": "min(skew_2_efficiency,skew_3_efficiency)",
            "far_efficiency": "min(skew_4_efficiency,skew_5_efficiency)",
        },
        "thresholds": {
            "strict_prior_grid_rows": 8_640,
            "minimum_valid_prior_values": 4_032,
            "current_excluded": True,
            "missing_rows_remain_in_calendar_window": True,
            "near_quantile": "39/40",
            "far_quantile": "9/10",
            "interpolation": False,
            "fallback_or_sweep": False,
        },
        "race": {
            "arm": {
                "current_row_complete": True,
                "near_sign_allowed": ["+1", "-1"],
                "near_efficiency_minimum": "3/5",
                "near_intensity_relation": ">= current near_threshold",
                "previous_grid_row_complete": True,
                "previous_near_threshold_available": True,
                "previous_near_intensity_relation": (
                    "< previous row's own near_threshold"
                ),
                "same_sign_far_blocks_when": {
                    "far_sign": "near_sign",
                    "far_efficiency_minimum": "1/2",
                    "far_intensity_relation": ">= current far_threshold",
                },
                "no_active_race": True,
                "queued_or_replaced_by_later_arm": False,
            },
            "state": {
                "stored_fields": ["arm_sign", "arm_grid_position"],
                "eligible_elapsed_grid_bars": [1, 2, 3, 4, 5, 6],
            },
            "confirmation": {
                "current_row_complete": True,
                "far_sign": "stored arm_sign",
                "far_efficiency_minimum": "1/2",
                "far_intensity_relation": ">= current far_threshold",
                "cumulative_interval": "arm bar through current bar inclusive",
                "cumulative_value": "exact sum of skew_2_net+skew_3_net",
                "cumulative_sign": "strictly stored arm_sign",
                "creates_candidate": True,
            },
            "cancellation": {
                "near_sign": "opposite stored arm_sign",
                "near_efficiency_minimum": "1/2",
                "creates_candidate": False,
            },
            "terminal_rules": {
                "simultaneous_confirmation_and_cancellation": (
                    "ambiguous; no event; race immediately retired"
                ),
                "incomplete_grid_row": "cancel immediately; no event",
                "no_terminal_by_age_6": "timeout; no event",
                "terminal_bar_consumed": True,
                "rearm_earliest": "next elapsed grid row",
            },
            "candidate": {
                "side": "stored arm_sign",
                "confirmation_age_inclusive": [1, 6],
                "persistence_cancellation_ambiguity_timeout_trade": False,
            },
        },
        "execution": {
            "signal_time": "confirming source_available_at",
            "entry_time": "signal_time+5 elapsed minutes",
            "processing_latency_bars": 1,
            "hold_bars_5m": 72,
            "hold_elapsed_hours": 6,
            "notional_exposure": 0.5,
            "reservation_interval": "[entry,exit)",
            "chronological_reservation": True,
            "accept_rule": "entry >= previous accepted exit",
            "reservation_reset": "only at known UTC quarter boundary",
            "global_nonoverlap": True,
            "adjacent_exit_entry_allowed": True,
            "signal_entry_exit_same_utc_quarter": True,
            "suppressed_candidate_queueing": False,
            "stop_take_profit_trailing_dynamic_size_or_regime_gate": False,
        },
        "source_support_gates": {
            "total_minimum": 120,
            "each_half_year_minimum": 45,
            "each_quarter_minimum": 20,
            "each_side_share_minimum": "7/20",
            "maximum_quarter_share": "2/5",
            "maximum_month_share": "3/20",
            "maximum_entry_gap_elapsed_days": 21,
            "age_ge_2_total_share_minimum": "1/5",
            "age_ge_2_each_half_share_minimum": "1/10",
            "exact_timing_unique_quarter_contained_nonoverlap": True,
            "post_2023_source_rows": 0,
            "synthetic_raw_confirmations": 0,
            "synthetic_accepted_events": 0,
            "failure_action": "retire before comparator rows and outcomes",
        },
        "source_controls": {
            "common": {
                "same_exact_source_values": True,
                "same_radius_specific_strict_prior_thresholds": True,
                "same_source_availability_and_processing_latency": True,
                "same_chronological_scheduler_and_nonoverlap": True,
                "same_72_bar_hold_and_quarter_containment": True,
                "immediate_controls_have_race_state": False,
                "relay_controls_have_independent_race_state": True,
            },
            "simultaneous_near_far": {
                "requires_primary_arm_conditions": [1, 2, 3, 4],
                "replaces_primary_arm_condition_5": True,
                "far_sign": "same as near_sign on current bar",
                "far_efficiency_minimum": "1/2",
                "far_intensity_relation": ">= current far_threshold",
                "candidate_side": "shared sign",
                "race": False,
            },
            "far_to_near_reverse_relay": {
                "arm_far_sign_allowed": ["+1", "-1"],
                "arm_far_efficiency_minimum": "3/5",
                "arm_far_intensity_relation": ">= current far_threshold",
                "previous_grid_row_complete": True,
                "previous_far_threshold_available": True,
                "previous_far_intensity_relation": (
                    "< previous row's own far_threshold"
                ),
                "near_already_qualified_blocks_when": {
                    "near_sign": "arm_far_sign",
                    "near_efficiency_minimum": "1/2",
                    "near_intensity_relation": ">= current near_threshold",
                },
                "race_elapsed_grid_bars": 6,
                "confirmation": {
                    "near_sign": "stored far arm sign",
                    "near_efficiency_minimum": "1/2",
                    "near_intensity_relation": ">= current near_threshold",
                    "cumulative_value": "exact sum of skew_4_net+skew_5_net",
                    "cumulative_interval": "arm bar through current bar inclusive",
                    "cumulative_sign": "strictly stored far arm sign",
                },
                "cancellation": (
                    "opposite common far sign with far_efficiency>=1/2"
                ),
                "terminal_rules_identical_to_primary": True,
            },
            "no_efficiency_relay": {
                "base": "primary race",
                "deleted_predicates": "every efficiency predicate",
                "sign_crossing_threshold_cumulative_and_timing_unchanged": True,
                "cancellation": "opposite common near sign alone",
            },
            "near_only": {
                "requires_primary_arm_conditions": [1, 2, 3, 4],
                "primary_arm_condition_5_used": False,
                "candidate_side": "near_sign",
                "race": False,
            },
            "far_only": {
                "far_sign_allowed": ["+1", "-1"],
                "far_efficiency_minimum": "1/2",
                "far_intensity_relation": ">= current far_threshold",
                "previous_grid_row_complete": True,
                "previous_far_threshold_available": True,
                "previous_far_intensity_relation": (
                    "< previous row's own far_threshold"
                ),
                "candidate_side": "far_sign",
                "race": False,
            },
            "stale_final_signals": {
                "source": "already-built primary confirmation clock",
                "shift_elapsed_grid_rows": {
                    "one_bar_stale": 1,
                    "five_bar_stale": 5,
                },
                "preserve_primary_side": True,
                "destination_row_must_be_complete": True,
                "destination_source_available_at_used": True,
                "rerun_processing_latency_quarter_and_nonoverlap": True,
                "outside_2023_or_incomplete_destination": "drop; never search forward",
            },
            "quarter_far_triple_permutation": {
                "tuple": [
                    "skew_4_net",
                    "skew_4_path",
                    "skew_4_efficiency",
                    "skew_5_net",
                    "skew_5_path",
                    "skew_5_efficiency",
                ],
                "donor_pool": "complete rows within same UTC quarter",
                "donor_sort_key": (
                    "SHA256('RQHR-72|quarter_far_triple_permutation|"
                    "<quarter>|<donor-date>')"
                ),
                "recipient_order": "complete rows chronological",
                "mapping": "zip sorted donors to chronological recipients",
                "incomplete_rows_remain_incomplete": True,
                "recipient_availability_preserved": True,
                "recompute": [
                    "all far features",
                    "strict-prior far thresholds",
                    "full relay and scheduler",
                ],
            },
            "side_controls": {
                "reuse_exact_primary_entries_and_exits": True,
                "deterministic_random_side": (
                    "LONG iff first byte of SHA256('RQHR-72|"
                    "deterministic_random_side|<entry-UTC-ISO>') is even"
                ),
                "exact_direction_flip": "negate primary side",
                "constant_long": "all intervals LONG",
                "constant_short": "all intervals SHORT",
            },
            "primary_must_beat_every_control": True,
        },
        "mechanical_nulls": {
            "source": str(RNCM_PREREGISTRATION),
            "source_sha256": RNCM_PREREGISTRATION_SHA256,
            "scenarios": [
                "smooth_symmetric",
                "tick_rounded_anchor",
                "stepped_asymmetric",
                "missing_rows",
                "discrete_asymmetric_ladder",
            ],
            "grid_bars": 105_120,
            "scheduled_snapshot_slots_per_bar": 10,
            "snapshot_position": "bar_index+snapshot_index/10",
            "snapshot_time": "bar_open+snapshot_index*30 seconds",
            "missing_predicate": "bar_index%1009<3 suppresses all ten slots",
            "aggregation": {
                "net": "last-first",
                "path": "sum(abs(current-previous))",
                "efficiency": "0 if path==0 else abs(net)/path",
            },
            "must_run_before_real_rqhr_column_read": True,
            "maximum_raw_confirmations_each_scenario": 0,
            "maximum_accepted_events_each_scenario": 0,
        },
        "common_window_policy": {
            "path": str(COMMON_WINDOW_POLICY),
            "sha256": COMMON_WINDOW_POLICY_SHA256,
            "comparison_window": [
                "2023-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
            ],
            "raw_group_validation_before_filter": True,
            "fully_contained_intervals_only": True,
            "boundary_intervals_clipped": False,
            "all_window_counts_reported": True,
        },
        "novelty": {
            "groups": [spec["group"] for spec in COMPARATOR_SPECS],
            "minimum_fully_contained_rows_each_group": 10,
            "parse_all_raw_rows_before_window_filter": True,
            "raw_group_checks": [
                "artifact protocol and closed flags",
                "declared event count and frozen canonical hash",
                "valid side and positive interval",
                "unique entry and chronological global nonoverlap",
                "quarter containment and position/date/hold consistency",
            ],
            "legacy_display_time_zone_trusted": False,
            "aware_time_reconstruction": (
                "2023-01-01T00:00:00Z+position*5 elapsed minutes"
            ),
            "display_date_must_equal_reconstructed_naive_utc": True,
            "maximum_exact_entry_jaccard": "1/10",
            "one_to_one_tolerance_5m_bars": 12,
            "maximum_candidate_containment": "7/20",
            "maximum_absolute_signed_exposure_correlation": "7/20",
            "undefined_comparison_action": "fail closed",
            "outcomes_allowed": False,
        },
        "economic_sequence": [
            "mechanical nulls",
            "2023 source support and controls",
            "2023 comparator novelty",
            "freeze strict evaluator",
            "train calendar 2023",
            "immutable source extension then test calendar 2024",
            "eval calendar 2025",
            "recent exact partial 2026 window",
        ],
        "strict_economic_gates": {
            "train": ["2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "recent": ["2026-01-01T00:00:00Z", "2026-07-19T00:00:00Z"],
            "full_year_minimum_trades": 100,
            "full_year_minimum_each_side": 25,
            "recent_minimum_trades": 40,
            "recent_minimum_each_side": 10,
            "signed_net_return_after_costs_and_funding_positive": True,
            "cagr_to_strict_mdd_minimum": 3.0,
            "strict_mdd_pct_maximum": 15.0,
            "strict_intratrade_high_water_mdd": True,
            "base_cost_notional_per_side": 0.0006,
            "stress_cost_notional_per_side": 0.0010,
            "stress_signed_net_return_positive": True,
            "every_contained_calendar_quarter_positive": True,
            "calendar_week_cluster_sign_flip_p_maximum": 0.10,
            "full_year_cagr_for_train_test_eval": True,
            "annualized_exact_window_cagr_for_recent": True,
            "inactive_time_in_cagr": True,
            "primary_control_ratio_margin_minimum": 0.25,
        },
        "portfolio_gate": {
            "stage": "after eval pass and before portfolio promotion",
            "maximum_absolute_signed_occupied_exposure_correlation": "7/20",
            "comparison_targets": "every frozen live portfolio sleeve",
            "window": "each exact common OOS window",
            "return_to_mdd_frontier_must_improve": True,
            "allowed_weight_rule": (
                "unchanged sleeve weights or separately preregistered allocation"
            ),
            "may_rescue_failed_standalone_split": False,
        },
        "production_gate": {
            "historical_live_feature_parity_required": True,
            "research_outcome_pass_alone_authorizes_live": False,
        },
        "rllm_boundary": {
            "authorized_before_source_novelty_train_and_test_pass": False,
            "later_actions": ["TRADE_FIXED_SIDE", "ABSTAIN"],
            "may_create_event_reverse_side_change_size_or_hold": False,
        },
        "mutable_parameters": [],
        "stopping_rule": (
            "any provenance, source-algebra, synthetic-null, incidence, novelty, "
            "train, or later sequential failure retires RQHR-72 unchanged"
        ),
    }


def _verify_hash(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"RQHR {label} hash mismatch: {path}")
    return observed


def _source_binding(*, verify: bool) -> dict[str, Any]:
    if verify:
        panel_hash = _verify_hash(SOURCE_PANEL, SOURCE_PANEL_SHA256, "source panel")
        manifest_hash = _verify_hash(
            SOURCE_MANIFEST, SOURCE_MANIFEST_SHA256, "source manifest"
        )
        builder_hash = _verify_hash(
            SOURCE_BUILDER, SOURCE_BUILDER_SHA256, "source builder"
        )
        mode = "raw bytes for SHA-256 only; no CSV or JSON parsing"
    else:
        panel_hash = SOURCE_PANEL_SHA256
        manifest_hash = SOURCE_MANIFEST_SHA256
        builder_hash = SOURCE_BUILDER_SHA256
        mode = "declared static fixture binding; no file read"
    return {
        "panel": str(SOURCE_PANEL),
        "panel_sha256": panel_hash,
        "manifest": str(SOURCE_MANIFEST),
        "manifest_sha256": manifest_hash,
        "builder": str(SOURCE_BUILDER),
        "builder_sha256": builder_hash,
        "read_mode": mode,
        "manifest_metadata_parsed": False,
        "source_value_rows_read": 0,
        "rqhr_columns_read": 0,
    }


def _history_bindings(*, verify: bool) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for spec in HISTORY_BINDINGS:
        observed = (
            _verify_hash(spec["path"], spec["sha256"], spec["name"])
            if verify
            else spec["sha256"]
        )
        output.append(
            {
                "name": spec["name"],
                "path": str(spec["path"]),
                "sha256": observed,
                "read_mode": (
                    "raw bytes for SHA-256 only"
                    if verify
                    else "declared static fixture binding; no file read"
                ),
                "historical_values_previously_opened": True,
                "value_rows_read_during_rqhr_preregistration": 0,
            }
        )
    return output


def _comparator_bindings(*, verify: bool) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for spec in COMPARATOR_SPECS:
        artifact_hash = (
            _verify_hash(spec["path"], spec["sha256"], spec["group"])
            if verify
            else spec["sha256"]
        )
        producer_hash = (
            _verify_hash(
                spec["producer"],
                spec["producer_sha256"],
                f"{spec['group']} producer",
            )
            if verify
            else spec["producer_sha256"]
        )
        row = {
            "group": spec["group"],
            "path": str(spec["path"]),
            "sha256": artifact_hash,
            "expected_raw_rows": spec["expected_raw_rows"],
            "canonical_clock_sha256": spec["canonical_clock_sha256"],
            "expected_protocol": spec["protocol"],
            "expected_closed_flags": dict(spec["closed_flags"]),
            "expected_selection_end_exclusive": spec[
                "selection_end_exclusive"
            ],
            "artifact_canonical_fields": spec["artifact_canonical_fields"],
            "producer": str(spec["producer"]),
            "producer_sha256": producer_hash,
            "parser": spec["parser"],
            "canonical_projection": spec["canonical_projection"],
            "canonical_serialization": "sorted-key compact JSON UTF-8",
            "common_window_policy_sha256": COMMON_WINDOW_POLICY_SHA256,
            "value_rows_read_during_preregistration": 0,
            "read_mode": (
                "artifact and producer raw bytes for SHA-256 only"
                if verify
                else "declared static fixture binding; no file read"
            ),
        }
        if "dependency" in spec:
            dependency_hash = (
                _verify_hash(
                    spec["dependency"],
                    spec["dependency_sha256"],
                    f"{spec['group']} dependency",
                )
                if verify
                else spec["dependency_sha256"]
            )
            row["dependency"] = str(spec["dependency"])
            row["dependency_sha256"] = dependency_hash
        output.append(row)
    return output


def build_preregistration(*, verify_sources: bool = True) -> dict[str, Any]:
    if verify_sources:
        _verify_hash(
            MECHANISM_DECISION,
            MECHANISM_DECISION_SHA256,
            "mechanism decision",
        )
        _verify_hash(
            COMMON_WINDOW_POLICY,
            COMMON_WINDOW_POLICY_SHA256,
            "common-window policy",
        )
    policy = policy_payload()
    boundary = (
        EXPECTED_OUTCOME_BOUNDARY
        if verify_sources
        else STATIC_TEST_OUTCOME_BOUNDARY
    )
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": POLICY_ID,
        "config": asdict(Config()),
        "policy": policy,
        "policy_hash": canonical_hash(policy),
        "mechanism_decision": {
            "path": str(MECHANISM_DECISION),
            "sha256": MECHANISM_DECISION_SHA256,
        },
        "common_window_policy": {
            "path": str(COMMON_WINDOW_POLICY),
            "sha256": COMMON_WINDOW_POLICY_SHA256,
        },
        "source_binding": _source_binding(verify=verify_sources),
        "history_bindings": _history_bindings(verify=verify_sources),
        "comparator_bindings": _comparator_bindings(verify=verify_sources),
        "verification_mode": (
            "verified_hashes_without_value_parsing"
            if verify_sources
            else "static_test_fixture"
        ),
        "artifact_eligible": verify_sources,
        "source_family_values_previously_opened": True,
        "rqhr_net_path_efficiency_values_opened": False,
        "rqhr_features_arms_confirmations_or_events_opened": False,
        "synthetic_nulls_run": False,
        "comparator_rows_opened_during_preregistration": False,
        "outcomes_opened": False,
        "performance_values_opened": False,
        "prior_research_disclosure": dict(PRIOR_RESEARCH_DISCLOSURE),
        "outcome_boundary": dict(boundary),
        "preregistration_source": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "next_action": (
            "freeze support builder, then run hash-bound synthetic nulls before "
            "any real RQHR column"
        ),
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def validate_preregistration(
    payload: Mapping[str, Any], *, verify_sources: bool = True
) -> None:
    if payload.get("candidate") != POLICY_ID:
        raise RuntimeError("RQHR candidate identity drift")
    if payload.get("policy") != policy_payload():
        raise RuntimeError("RQHR frozen policy drift")
    if payload.get("policy_hash") != canonical_hash(policy_payload()):
        raise RuntimeError("RQHR policy hash mismatch")
    if payload.get("prior_research_disclosure") != PRIOR_RESEARCH_DISCLOSURE:
        raise RuntimeError("RQHR prior-research disclosure drift")
    expected_boundary = (
        EXPECTED_OUTCOME_BOUNDARY
        if verify_sources
        else STATIC_TEST_OUTCOME_BOUNDARY
    )
    if payload.get("outcome_boundary") != expected_boundary:
        raise RuntimeError("RQHR outcome boundary drift")
    if payload.get("artifact_eligible") is not verify_sources:
        raise RuntimeError("RQHR artifact eligibility drift")
    expected_mode = (
        "verified_hashes_without_value_parsing"
        if verify_sources
        else "static_test_fixture"
    )
    if payload.get("verification_mode") != expected_mode:
        raise RuntimeError("RQHR verification mode drift")
    if payload.get("common_window_policy") != {
        "path": str(COMMON_WINDOW_POLICY),
        "sha256": COMMON_WINDOW_POLICY_SHA256,
    }:
        raise RuntimeError("RQHR common-window policy binding drift")
    for field in (
        "rqhr_net_path_efficiency_values_opened",
        "rqhr_features_arms_confirmations_or_events_opened",
        "synthetic_nulls_run",
        "comparator_rows_opened_during_preregistration",
        "outcomes_opened",
        "performance_values_opened",
    ):
        if payload.get(field) is not False:
            raise RuntimeError(f"RQHR boundary opened: {field}")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("RQHR canonical hash mismatch")
    expected = build_preregistration(verify_sources=verify_sources)
    expected["config"] = dict(payload.get("config", {}))
    expected_core = {
        key: value for key, value in expected.items() if key != "manifest_hash"
    }
    expected["manifest_hash"] = canonical_hash(expected_core)
    if payload != expected:
        raise RuntimeError("RQHR preregistration differs from frozen build")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        path.resolve().relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("RQHR output must remain inside repository") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_preregistration(cfg: Config = Config()) -> tuple[dict[str, Any], str]:
    output = _repository_path(cfg.output)
    payload = build_preregistration()
    payload["config"] = asdict(cfg)
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    payload["manifest_hash"] = canonical_hash(core)
    validate_preregistration(payload)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        validate_preregistration(existing)
        if existing != payload:
            raise RuntimeError("existing RQHR preregistration differs; refusing overwrite")
        return payload, "verified_existing"
    try:
        _atomic_write(output, payload)
        return payload, "created"
    except FileExistsError:
        existing = json.loads(output.read_text(encoding="utf-8"))
        validate_preregistration(existing)
        if existing != payload:
            raise RuntimeError("concurrent RQHR preregistration differs")
        return payload, "verified_existing"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload, status = write_preregistration(Config(output=args.output))
    print(
        json.dumps(
            {
                "status": status,
                "candidate": payload["candidate"],
                "output": args.output,
                "policy_hash": payload["policy_hash"],
                "manifest_hash": payload["manifest_hash"],
                "rqhr_values_opened": payload[
                    "rqhr_net_path_efficiency_values_opened"
                ],
                "comparator_rows_opened": payload[
                    "comparator_rows_opened_during_preregistration"
                ],
                "outcomes_opened": payload["outcomes_opened"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
