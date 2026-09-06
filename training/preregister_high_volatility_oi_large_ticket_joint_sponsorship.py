"""Outcome-blind preregistration for HVOILTJ-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from training import preregister_high_volatility_cross_structure_action_vote as hvcav


POLICY_ID = "HVOILTJ-8"
DEFAULT_OUTPUT = Path("results/high_volatility_oi_large_ticket_joint_sponsorship_preregistration_2026-08-16.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    contract = hvcav.build()
    core = {
        "protocol_version": "high_volatility_oi_large_ticket_joint_sponsorship_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-16",
        "singleton": True,
        "exploratory_discovery": True,
        "fresh_confirmatory_evidence": False,
        "source_incidence_opened": False,
        "outcomes_opened": False,
        "gross9_rows_opened": False,
        "candidate_family": [POLICY_ID],
        "candidate_family_size": 1,
        "mechanism": {
            "claim": (
                "A volatile BTC move is more likely to persist when two independent sponsorship "
                "channels agree: five-minute absolute OI changes co-occur with absolute price "
                "changes and gross OI churn is elevated, while six-hour quote turnover is more "
                "temporally concentrated than execution counts, indicating unusually large "
                "tickets. At the fresh joint onset, follow the common three-hour, six-hour, and "
                "final-hour direction for eight hours."
            ),
            "side": "common strict sign of completed three-hour, six-hour, and final-hour BTC returns",
            "why_distinct": (
                "HVOILTJ requires simultaneous leverage coactivity and large-ticket temporal "
                "concentration. It uses neither OI direction/endpoints, funding, premium, spot, "
                "alts, FX, fitted outcomes, prior event clocks, nor promoted controls."
            ),
            "why_suited_to_volatile_regimes": "both independent three-hour and six-hour variation ranks must be at least 0.65",
            "why_low_gross9_overlap_is_plausible": "fresh joint 3-hour OI-coactivity and 6-hour ticket-clustering onsets are absent from Gross9 primitives",
        },
        "features": {
            "decision_grid": "exact UTC boundaries divisible by three hours",
            "oi_price_block": "36 exact coherent BTCUSDT five-minute bars and 37 exact positive 5m OI observations over [D-3h,D]",
            "oi_price_coactivity": "Pearson correlation(abs(dlog OI),abs(five-minute return)), finite with nonzero vector variances",
            "gross_oi_activity": "sum(abs(dlog OI)) over 36 changes, finite strict positive",
            "oi_ranks": "strict-prior midranks over 720/504 earlier source-valid 3h blocks; coactivity>=0.75, gross activity>=0.60, variation>=0.65",
            "ticket_block": "360 exact coherent BTCUSDT one-minute rows [D-6h,D) with positive execution counts and nonnegative quote turnover",
            "large_ticket_clustering": "turnover-share HHI minus execution-count-share HHI, finite strict positive",
            "ticket_ranks": "strict-prior hourly midranks over 2160/1440 earlier source-valid 6h windows; clustering>=0.80 and variation>=0.65",
            "direction_confirmation": "three-hour, six-hour, and final-hour returns have one strict sign",
            "joint_eligible": "all OI, ticket, variation, and direction gates pass at D",
            "joint_onset": "joint eligible now and the immediately prior exact 3h source-valid opportunity was joint ineligible; missing prior cannot trigger",
            "side": "strict sign of completed three-hour return",
            "no_imputation": True,
        },
        "clock": {
            "decision": "completed exact three-hour boundary D",
            "entry": "exact BTCUSDT perpetual D+5m open",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal entry",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact held settlements only after source and Gross9 pass",
        },
        "policy": {
            "oi_block_hours": 3,
            "oi_history_blocks": 720,
            "oi_minimum_history_blocks": 504,
            "coactivity_rank_min": 0.75,
            "gross_oi_activity_rank_min": 0.60,
            "oi_variation_rank_min": 0.65,
            "ticket_window_minutes": 360,
            "ticket_history_hours": 2160,
            "ticket_minimum_history_hours": 1440,
            "clustering_rank_min": 0.80,
            "ticket_variation_rank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 8,
            "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.0010,
        },
        "stages": contract["stages"],
        "source_support_gates": contract["source_support_gates"],
        "gross9_novelty_gates": contract["gross9_novelty_gates"],
        "economic_gates": contract["economic_gates"],
        "source_plan": {
            "bars": "Postgres bars_binance BTCUSDT exact coherent 1m OHLC, quote_asset_volume, and number_of_trades",
            "open_interest": "Postgres open_interest_binance BTCUSDT period=5m source=open_interest_hist exact observations",
            "window": ["2023-04-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "read_after_preregistration": True,
            "execution_prices": "sealed until source support and Gross9 pass",
        },
        "research_boundary": {
            "prior_HVOIPCSR_source_novelty_and_train_failure_known": True,
            "prior_HVLTTC_source_novelty_and_train_failure_known": True,
            "prior_component_event_rows_reused": False,
            "exact_joint_state_incidence_or_outcomes_known": False,
            "user_authorized_multi_condition_composition": True,
            "prior_outcomes_informed_candidate_family": True,
            "prior_outcomes_used_to_set_component_thresholds_side_hold_or_clock": False,
            "component_thresholds_copied_without_search": True,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "composition_is_exploratory_not_confirmatory": True,
            "promoted_prior_control": False,
            "classification": "exploratory discovery; not fresh confirmatory evidence",
        },
        "stopping_rule": "terminal first failure; no source, component, rank, onset, side, clock, hold, subset, threshold, comparator, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: Mapping[str, Any]) -> None:
    if value != build():
        raise RuntimeError("HVOILTJ-8 preregistration drift")
    contract = hvcav.build()
    for key in ("stages", "source_support_gates", "gross9_novelty_gates", "economic_gates"):
        if value[key] != contract[key]:
            raise RuntimeError(f"HVOILTJ-8 {key} drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = build()
    validate(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
