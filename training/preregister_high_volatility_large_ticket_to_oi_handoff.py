"""Outcome-blind preregistration for HVLTOIH-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from training import preregister_high_volatility_cross_structure_action_vote as hvcav


POLICY_ID = "HVLTOIH-8"
DEFAULT_OUTPUT = Path("results/high_volatility_large_ticket_to_oi_handoff_preregistration_2026-08-16.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    contract = hvcav.build()
    core = {
        "protocol_version": "high_volatility_large_ticket_to_oi_handoff_v1",
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
                "In a volatile market, unusually large-ticket temporal clustering at D-3h can "
                "initiate price discovery before leverage participates. If the next completed "
                "three-hour block at D exhibits extreme absolute OI-price coactivity and gross OI "
                "churn in the same direction as the earlier six-hour and final-hour large-ticket "
                "move, follow the confirmed large-ticket-to-leverage handoff for eight hours."
            ),
            "side": "common strict sign of lagged six-hour/final-hour returns and current three-hour return",
            "why_distinct": (
                "HVLTOIH is an ordered two-stage handoff: lagged ticket concentration followed by "
                "current OI-price coactivity. It is neither the simultaneous HVOILTJ conjunction "
                "nor either component clock and uses no funding, premium, spot, alt, or fitted outcome."
            ),
            "why_suited_to_volatile_regimes": "lagged ticket and current OI-price variation ranks must each be at least 0.65",
            "why_low_gross9_overlap_is_plausible": "an exact three-hour ordered ticket-to-leverage handoff is absent from Gross9 primitives",
        },
        "features": {
            "decision_grid": "exact UTC boundaries divisible by three hours",
            "lagged_ticket_state": "at D-3h, a source-valid 6h large-ticket state with clustering rank>=0.80 and variation rank>=0.65",
            "large_ticket_clustering": "turnover-share HHI minus execution-count-share HHI over 360 exact one-minute bars, finite strict positive",
            "current_oi_state": "at D, a source-valid 3h OI-price state with coactivity rank>=0.75, gross OI activity rank>=0.60, and variation rank>=0.65",
            "oi_price_coactivity": "Pearson correlation(abs(dlog OI),abs(five-minute return)) over 36 exact pairs",
            "gross_oi_activity": "sum(abs(dlog OI)) over the current 36 changes",
            "direction_confirmation": "lagged six-hour return, lagged final-hour return, and current three-hour return have one strict sign",
            "ordered_handoff": "lagged ticket state and current OI state both pass with exact D-(D-3h) spacing and direction confirmation",
            "handoff_onset": "ordered handoff now and immediately prior exact 3h source-valid opportunity was inactive; missing prior cannot trigger",
            "side": "strict sign of current completed three-hour return",
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
            "handoff_lag_hours": 3,
            "oi_history_blocks": 720,
            "oi_minimum_history_blocks": 504,
            "coactivity_rank_min": 0.75,
            "gross_oi_activity_rank_min": 0.60,
            "oi_variation_rank_min": 0.65,
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
            "prior_HVOIPCSR_and_HVLTTC_outcomes_known": True,
            "prior_simultaneous_HVOILTJ_train_failure_known": True,
            "same_mechanism_as_HVOILTJ": False,
            "prior_component_event_rows_reused": False,
            "exact_ordered_handoff_incidence_or_outcomes_known": False,
            "user_authorized_multi_condition_composition": True,
            "component_thresholds_copied_without_search": True,
            "prior_outcomes_used_to_set_lag_thresholds_side_hold_or_clock": False,
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
        "stopping_rule": "terminal first failure; no source, component, lag, rank, onset, side, clock, hold, subset, threshold, comparator, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: Mapping[str, Any]) -> None:
    if value != build():
        raise RuntimeError("HVLTOIH-8 preregistration drift")
    contract = hvcav.build()
    for key in ("stages", "source_support_gates", "gross9_novelty_gates", "economic_gates"):
        if value[key] != contract[key]:
            raise RuntimeError(f"HVLTOIH-8 {key} drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = build()
    validate(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
