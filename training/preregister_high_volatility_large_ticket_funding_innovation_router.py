"""Outcome-blind preregistration for HVLTFIR-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from training import preregister_high_volatility_cross_structure_action_vote as hvcav


POLICY_ID = "HVLTFIR-8"
DEFAULT_OUTPUT = Path("results/high_volatility_large_ticket_funding_innovation_router_preregistration_2026-08-16.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    contract = hvcav.build()
    core = {
        "protocol_version": "high_volatility_large_ticket_funding_innovation_router_v1",
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
                "At a fresh volatile large-ticket clustering onset, the latest actual funding "
                "innovation distinguishes crowding from under-positioning. If the funding change "
                "has the completed price direction, fade the move because leverage is crowding "
                "into concentrated execution; if funding changes against price, continue because "
                "large-ticket demand is overcoming adverse carry."
            ),
            "side": "negative funding-change sign, equivalently price continuation under adverse innovation and reversal under aligned innovation",
            "why_distinct": (
                "HVLTFIR preserves the independently defined large-ticket onset but routes its side "
                "with the sign relation of two causally observed actual funding settlements. It uses "
                "no funding level/rank, OI, premium, spot, alt, fitted outcome, or promoted control."
            ),
            "why_suited_to_volatile_regimes": "large-ticket clustering rank>=0.80 and six-hour realized-variation rank>=0.65",
            "why_low_gross9_overlap_is_plausible": "hourly large-ticket onsets with an actual funding-innovation side router are absent from Gross9 primitives",
        },
        "features": {
            "decision_grid": "every exact UTC hour D",
            "large_ticket_state": "360 exact coherent BTCUSDT one-minute bars [D-6h,D)",
            "large_ticket_clustering": "turnover-share HHI minus execution-count-share HHI, finite strict positive",
            "ticket_ranks": "strict-prior hourly midranks over 2160/1440 source-valid windows; clustering>=0.80 and variation>=0.65",
            "direction_confirmation": "six-hour and final-hour returns have one strict sign",
            "ticket_onset": "ticket state eligible now and immediately prior exact source-valid hourly state ineligible",
            "latest_funding": "latest unique actual BTCUSDT settlement S1<=D and its immediately preceding actual settlement S0",
            "funding_freshness": "0<=D-S1<8h and 7h59m<=S1-S0<=8h1m; no timestamp snapping or imputation",
            "funding_innovation": "funding_rate[S1]-funding_rate[S0], finite strict nonzero",
            "routing": "aligned funding innovation reverses the confirmed price direction; opposite innovation continues it",
            "side": "negative strict sign of funding innovation",
            "no_imputation": True,
        },
        "clock": {
            "decision": "completed exact hourly boundary D",
            "entry": "exact BTCUSDT perpetual D+5m open",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal entry",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "signal settlements precede entry; exact held settlements included only after source and Gross9 pass",
        },
        "policy": {
            "ticket_window_minutes": 360,
            "ticket_history_hours": 2160,
            "ticket_minimum_history_hours": 1440,
            "clustering_rank_min": 0.80,
            "variation_rank_min": 0.65,
            "funding_max_age_hours": 8,
            "funding_gap_min_minutes": 479,
            "funding_gap_max_minutes": 481,
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
            "funding": "Postgres funding_rates_binance BTCUSDT actual timestamps and rates",
            "window": ["2023-04-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "read_after_preregistration": True,
            "execution_prices": "sealed until source support and Gross9 pass",
        },
        "research_boundary": {
            "prior_HVLTTC_source_novelty_and_train_failure_known": True,
            "prior_funding_level_and_adverse_flow_source_failures_known": True,
            "exact_large_ticket_funding_innovation_routing_incidence_or_outcomes_known": False,
            "component_ticket_thresholds_copied_without_search": True,
            "user_authorized_multi_condition_composition": True,
            "prior_outcomes_informed_candidate_family": True,
            "prior_outcomes_used_to_set_funding_gap_side_hold_or_clock": False,
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
        "stopping_rule": "terminal first failure; no source, ticket, funding chronology, gap, route, side, clock, hold, subset, threshold, comparator, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: Mapping[str, Any]) -> None:
    if value != build():
        raise RuntimeError("HVLTFIR-8 preregistration drift")
    contract = hvcav.build()
    for key in ("stages", "source_support_gates", "gross9_novelty_gates", "economic_gates"):
        if value[key] != contract[key]:
            raise RuntimeError(f"HVLTFIR-8 {key} drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    value = build(); validate(value); args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
