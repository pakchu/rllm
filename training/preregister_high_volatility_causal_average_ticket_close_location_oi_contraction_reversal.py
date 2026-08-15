"""Outcome-blind preregistration for HVCATCLOICR-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from training import preregister_high_volatility_cross_structure_action_vote as hvcav


POLICY_ID = "HVCATCLOICR-8"
DEFAULT_OUTPUT = Path("results/high_volatility_causal_average_ticket_close_location_oi_contraction_reversal_preregistration_2026-08-16.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    contract = hvcav.build()
    core = {
        "protocol_version": "high_volatility_causal_average_ticket_close_location_oi_contraction_reversal_v1",
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
            "claim": "When BTC perpetual average quote ticket size accelerates unusually from the first to the second four-hour half, the completed cycle closes in the directional outer 35 percent of its exact high-low range, realized variation is elevated, and open interest contracts, large transactions absorb a liquidation-driven directional close and the move should reverse over eight hours.",
            "side": "opposite strict sign of the completed eight-hour BTC perpetual return",
            "why_distinct": "HVCATCLOICR requires within-cycle acceleration in quote turnover per transaction and independently confirms the completed direction with an exact completed-range close location, plus exact OI contraction and variation. No prior composite event set or control is reused; it uses no spot, ETH, funding, premium-index, fitted outcome, or promoted control.",
            "why_suited_to_volatile_regimes": "the completed eight-hour BTC realized variation must be in its causal upper 40 percent",
            "why_low_gross9_overlap_is_plausible": "fixed eight-hour accelerating-average-ticket plus OI-contraction absorption states are absent from Gross9 primitives",
        },
        "features": {
            "decision": "exact 00:00, 08:00, or 16:00 UTC boundary D",
            "open_interest": "finite positive BTCUSDT period=5m sum_open_interest observations at D-8h and D, each exchange-timestamped no later than D",
            "oi_contraction": "strict negative log(OI[D]/OI[D-8h])",
            "completed_cycle": "480 exact coherent bars_binance BTCUSDT one-minute rows with nonnegative quote_asset_volume and positive integer number_of_trades over [D-8h,D)",
            "half_average_tickets": "sum quote_asset_volume divided by sum number_of_trades separately over first and second 240-minute halves; both quote sums and trade-count sums strict positive",
            "average_ticket_acceleration": "log(second_half_average_ticket/first_half_average_ticket), finite strict positive",
            "average_ticket_acceleration_rank": "strict-prior midrank over at most 270 source-valid cycles; current excluded; minimum 180; rank>=0.60",
            "close_location": "(final completed one-minute close-cycle_low)/(cycle_high-cycle_low), finite with strict positive exact completed-cycle range",
            "close_location_confirmation": "close_location>0.65 for positive completed_return or close_location<0.35 for negative completed_return",
            "completed_return": "sum of 480 one-minute log(close/open) returns, finite strict nonzero",
            "realized_variation": "sqrt(sum squared one-minute log(close/open) returns over exactly 480 coherent BTCUSDT bars [D-8h,D))",
            "variation_rank": "strict-prior midrank over at most 270 source-valid cycles; current excluded; minimum 180; rank>=0.60",
            "eligible": "exact coherent completed perpetual cycle, positive average-ticket acceleration, nonzero completed return, directional outer-35-percent completed-range close location, two exact OI endpoints with strict contraction, and both average-ticket-acceleration and variation ranks pass at D; no taker-flow sign, spot source, funding, premium-index, alt, or post-decision condition",
            "no_imputation": True,
        },
        "clock": {
            "decisions": "exact 00/08/16 UTC boundaries",
            "entry": "exact BTCUSDT D+5m open",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal entry",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact held settlements only after source and Gross9 pass",
        },
        "policy": {
            "history_cycles": 270,
            "minimum_history_cycles": 180,
            "path_minutes": 480,
            "half_minutes": 240,
            "average_ticket_acceleration_rank_min": 0.60,
            "variation_rank_min": 0.60,
            "close_location_outer_fraction": 0.35,
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
            "open_interest": "Postgres open_interest_binance BTCUSDT period=5m exact endpoint observations",
            "bars": "Postgres bars_binance BTCUSDT exact coherent 1m OHLC, quote_asset_volume, and number_of_trades",
            "window": ["2020-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "read_after_preregistration": True,
            "execution_prices": "sealed until source support and Gross9 pass",
        },
        "research_boundary": {
            "prior_HVCATCLIR_source_novelty_and_train_failure_known": True,
            "prior_HVCCLOICR_source_novelty_and_train_failure_known": True,
            "prior_HVCATAIR_source_novelty_and_train_failure_known": True,
            "user_authorized_multi_condition_composition": True,
            "exact_composite_incidence_or_outcomes_known": False,
            "same_mechanism_as_prior_average_ticket_candidates": False,
            "prior_HVCVAIR_source_novelty_and_train_failure_known": True,
            "same_mechanism_as_HVCVAIR": False,
            "prior_HVCNAIR_source_novelty_and_train_failure_known": True,
            "same_mechanism_as_HVCNAIR": False,
            "prior_outcomes_informed_candidate_family": True,
            "prior_outcomes_used_to_set_rank_side_hold_or_clock": False,
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
        "stopping_rule": "terminal first failure; no OI endpoint or contraction rule, average-ticket acceleration formula, close-location confirmation, acceleration or variation history, side, clock, hold, subset, threshold, comparator, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: Mapping[str, Any]) -> None:
    if value != build():
        raise RuntimeError("HVCATCLOICR-8 preregistration drift")
    contract = hvcav.build()
    for key in ("stages", "source_support_gates", "gross9_novelty_gates", "economic_gates"):
        if value[key] != contract[key]:
            raise RuntimeError(f"HVCATCLOICR-8 {key} drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = build()
    validate(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
