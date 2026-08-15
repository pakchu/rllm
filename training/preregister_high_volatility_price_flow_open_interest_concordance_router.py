"""Outcome-blind preregistration for HVPFOICR-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from training import preregister_high_volatility_cross_structure_action_vote as hvcav


POLICY_ID = "HVPFOICR-8"
DEFAULT_OUTPUT = Path("results/high_volatility_price_flow_open_interest_concordance_router_preregistration_2026-08-16.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    contract = hvcav.build()
    core = {
        "protocol_version": "high_volatility_price_flow_open_interest_concordance_router_v1",
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
            "claim": "When a completed high-variation BTC eight-hour cycle has price/aggregate aggressive-flow sign agreement under open-interest expansion, or price/flow sign disagreement under open-interest contraction, the aggressive-flow direction identifies respectively sponsored continuation or liquidation absorption; trade that flow direction for eight hours.",
            "side": "strict sign of completed aggregate aggressive quote flow",
            "why_distinct": "HVPFOICR requires an exact completed-range close location with exact OI contraction and variation. No prior composite event set or control is reused; it uses no spot, ETH, funding, premium-index, fitted outcome, or promoted control.",
            "why_suited_to_volatile_regimes": "the completed eight-hour BTC realized variation must be in its causal upper 40 percent",
            "why_low_gross9_overlap_is_plausible": "fixed eight-hour price-flow-OI sign concordance states are absent from Gross9 primitives",
        },
        "features": {
            "decision": "exact 00:00, 08:00, or 16:00 UTC boundary D",
            "open_interest": "finite positive BTCUSDT period=5m sum_open_interest observations at D-8h and D, exact and available by D",
            "oi_change": "log(OI[D]/OI[D-8h]), finite strict nonzero",
            "cycle": "480 exact coherent BTCUSDT bars_binance interval=1m rows [D-8h,D) with positive quote turnover and 0<=taker_buy_quote<=quote turnover",
            "price_return": "sum of 480 minute ln(close/open) returns, finite strict nonzero",
            "flow_share": "sum(2*taker_buy_quote-quote_asset_volume)/sum(quote_asset_volume), finite strict nonzero",
            "state_identity": "sign(price_return)*sign(flow_share)==sign(oi_change); expansion requires price-flow agreement and contraction requires disagreement",
            "side": "sign(flow_share) for both sponsored-continuation and liquidation-absorption states",
            "realized_variation": "sqrt(sum squared one-minute price returns), strict positive",
            "variation_rank": "strict-prior midrank over at most 270 source-valid cycles; current excluded; minimum 180; rank>=0.60",
            "eligible": "exact source-valid cycle, strict nonzero price/flow/OI signs, state_identity true, and variation rank passes",
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
            "variation_rank_min": 0.60,
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
            "bars": "Postgres bars_binance BTCUSDT exact coherent 1m OHLC, quote_asset_volume, and taker_buy_quote",
            "window": ["2020-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "read_after_preregistration": True,
            "execution_prices": "sealed until source support and Gross9 pass",
        },
        "research_boundary": {
            "prior_flow_OI_source_and_router_failures_known": True,
            "exact_symmetric_price_flow_OI_identity_incidence_or_outcomes_known": False,
            
            "user_authorized_multi_condition_composition": True,
            "prior_composite_incidence_or_outcomes_known": False,
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
        "stopping_rule": "terminal first failure; no OI endpoint, price-flow-OI identity, variation history, side, clock, hold, subset, threshold, comparator, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: Mapping[str, Any]) -> None:
    if value != build():
        raise RuntimeError("HVPFOICR-8 preregistration drift")
    contract = hvcav.build()
    for key in ("stages", "source_support_gates", "gross9_novelty_gates", "economic_gates"):
        if value[key] != contract[key]:
            raise RuntimeError(f"HVPFOICR-8 {key} drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = build()
    validate(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
