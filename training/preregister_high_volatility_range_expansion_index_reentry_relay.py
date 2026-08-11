"""Outcome-blind preregistration for HVREI-24."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVREI-24"
DEFAULT_OUTPUT = Path("results/high_volatility_range_expansion_index_reentry_relay_preregistration_2026-08-11.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    contract = copy.deepcopy(template.build())
    contract.pop("manifest_hash")
    contract.update(
        protocol_version="high_volatility_range_expansion_index_reentry_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-11",
        mechanism={
            "claim": "During elevated realized variation, the canonical TD Range Expansion Index returning inside its -60/+60 envelope on completed four-hour BTC auctions identifies filtered directional range expansion beginning to reverse; follow the published inward re-entry direction for twenty-four hours.",
            "side": "long when TD REI crosses strictly above -60 after being at or below -60; short when it crosses strictly below +60 after being at or above +60",
            "why_distinct": "TD REI sums two-bar high and low displacement only when its separate five-to-eight-bar structural filters admit the move. DeMarker compares one-bar positive high/low expansions; DMI uses true-range directional movement; Mass Index ignores direction; ordinary oscillators use closes or candle bodies. HVREI uses no volume, flow, OI, funding, fitted outcome, reused event, repair, or promoted control.",
            "why_suited_to_volatile_regimes": "completed trailing twenty-four-hour realized variation must rank in its causal upper 35%, focusing canonical range-exhaustion re-entry on July-like auctions",
            "why_low_gross9_overlap_is_plausible": "sparse four-hour structurally filtered two-bar range-expansion re-entry clocks are absent from Gross9 primitives",
        },
        external_basis={
            "origin": "Thomas R. DeMark, TD Range Expansion Index; Technical Analysis of Stocks & Commodities, August 1997",
            "definition_source": "https://demark.com/wp-content/uploads/2025/07/1997-Stocks-Commodities-August.pdf",
            "fixed_definition": "eight-bar rolling sums of conditionally admitted [(high-high[-2])+(low-low[-2])] divided by rolling sums of the corresponding absolute differences, multiplied by 100; structural conditions use lags 2,5,6,7,8",
            "fixed_interpretation": "published weakness signal is downward re-entry through +60 and strength signal is upward re-entry through -60",
            "selection_use": "published formula, eight-bar period, +/-60 levels, and inward direction only; no incidence or outcomes",
        },
        features={
            "decision_grid": "every exact four-hour UTC boundary",
            "source_bar": "exact aggregation of 240 coherent BTCUSDT one-minute rows [T-4h,T)",
            "range_difference": "(high[t]-high[t-2])+(low[t]-low[t-2])",
            "denominator_term": "abs(high[t]-high[t-2])+abs(low[t]-low[t-2])",
            "high_filter_zero": "high[t-2]<close[t-7] and high[t-2]<close[t-8] and high[t]<high[t-5] and high[t]<high[t-6]",
            "low_filter_zero": "low[t-2]>close[t-7] and low[t-2]>close[t-8] and low[t]>low[t-5] and low[t]>low[t-6]",
            "admitted_numerator_term": "zero only when either zero-filter is true; otherwise range_difference",
            "rei": "100 times rolling eight-bar admitted numerator sum divided by rolling eight-bar denominator sum; invalid at nonpositive denominator",
            "reentry": "current valid REI strictly crosses inward through -60 or +60 from the immediately prior valid REI",
            "variation": "sqrt(sum squared completed five-minute open-to-close log returns over [T-24h,T)), finite strict positive",
            "variation_rank": "strict-prior midrank over at most 180 earlier valid decisions, minimum 120, current excluded; rank>=0.65",
            "no_imputation": True,
        },
        clock={
            "feature_available": "four-hour boundary after completed source bar",
            "entry": "exact BTCUSDT open five elapsed minutes later",
            "hold": "24 elapsed hours",
            "reservation": "global half-open first-eligible reservation; exit first on equal open",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        policy={
            "rei_periods": 8,
            "lower_level": -60.0,
            "upper_level": 60.0,
            "variation_hours": 24,
            "variation_history_decisions": 180,
            "minimum_variation_history_decisions": 120,
            "variation_rank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 24,
            "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        diagnostic_controls={
            "names": ["no_variation_gate", "unfiltered_rei", "one_bar_stale_reentry", "direction_flip", "forced_long"],
            "cannot_be_promoted": True,
        },
        source_plan={
            "bars": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close"],
                "window": ["2023-05-01T00:00:00Z", "2026-08-01T00:00:00Z"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        research_boundary={
            "canonical_td_rei_definition_read": True,
            "repository_td_rei_candidate_found": False,
            "adjacent_demarker_dmi_mass_index_outcomes_known": True,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "canonical TD REI(8) +/-60 inward re-entry under the requested high-variation regime",
        },
        stopping_rule="Terminal first failure; no REI period, structural filter, formula, level, crossing, variation, side, hold, clock, subset, threshold, or control repair.",
    )
    return {**contract, "manifest_hash": canonical_hash(contract)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVREI preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build()
    validate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
