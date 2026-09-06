"""Outcome-blind preregistration for HVCTIIR-8."""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
from typing import Any, Mapping
from training import preregister_high_volatility_cross_structure_action_vote as hvcav

POLICY_ID = "HVCTIIR-8"
DEFAULT_OUTPUT = Path("results/high_volatility_causal_ticket_impact_inventory_relay_preregistration_2026-08-16.json")

def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def build() -> dict[str, Any]:
    contract = hvcav.build()
    core = {
        "protocol_version": "high_volatility_causal_ticket_impact_inventory_relay_v1", "policy_id": POLICY_ID,
        "as_of_date": "2026-08-16", "singleton": True, "exploratory_discovery": True,
        "fresh_confirmatory_evidence": False, "source_incidence_opened": False, "outcomes_opened": False,
        "gross9_rows_opened": False, "candidate_family": [POLICY_ID], "candidate_family_size": 1,
        "mechanism": {
            "claim": "When larger BTC perpetual transactions are unusually aligned with larger absolute one-minute repricing inside a completed high-variation eight-hour cycle and open interest expands, fresh leveraged large-ticket impact sponsors continuation of the completed direction for eight hours.",
            "side": "strict sign of the completed eight-hour BTC perpetual return",
            "why_distinct": "HVCTIIR measures within-cycle co-movement between log average quote ticket and absolute minute return. It is neither average-ticket level or acceleration nor aggregate turnover or trade-count acceleration, and uses no taker sign, spot, alt, funding, premium, fitted outcome, prior event set, or promoted control.",
            "why_suited_to_volatile_regimes": "completed BTC realized variation must be in its causal upper 40 percent",
            "why_low_gross9_overlap_is_plausible": "large-ticket price-impact alignment plus exact OI sponsorship is absent from Gross9 primitives",
        },
        "features": {
            "decision": "exact 00:00, 08:00, or 16:00 UTC boundary D",
            "open_interest": "finite positive BTCUSDT period=5m sum_open_interest at D-8h and D",
            "oi_expansion": "strict positive log(OI[D]/OI[D-8h])",
            "completed_cycle": "480 exact coherent bars_binance BTCUSDT one-minute rows [D-8h,D), each with nonnegative quote_asset_volume and positive integer number_of_trades",
            "minute_ticket": "quote_asset_volume/number_of_trades per completed minute; finite nonnegative",
            "ticket_impact_alignment": "Pearson correlation across all 480 rows between log1p(minute_ticket) and abs(log(close/open)); both vectors must have strict positive population standard deviation; finite strict positive correlation",
            "ticket_impact_rank": "strict-prior midrank over at most 270 source-valid cycles; current excluded; minimum 180; rank>=0.60",
            "completed_return": "sum of 480 one-minute log(close/open) returns, finite strict nonzero",
            "realized_variation": "sqrt(sum squared one-minute log(close/open) returns over the exact completed cycle), finite strict positive",
            "variation_rank": "strict-prior 270/180 midrank; current excluded; rank>=0.60",
            "eligible": "positive ticket-impact alignment and rank, high variation rank, strict OI expansion, and nonzero completed return; no post-decision condition",
            "no_imputation": True,
        },
        "clock": {"decisions": "exact 00/08/16 UTC boundaries", "entry": "exact BTCUSDT D+5m open", "hold": "8 elapsed hours", "reservation": "global half-open; exit first on equal entry", "split_crossing_action": "skip", "gross_exposure": .5, "funding": "not a signal input; exact held settlements only after source and Gross9 pass"},
        "policy": {"history_cycles": 270, "minimum_history_cycles": 180, "path_minutes": 480, "ticket_impact_rank_min": .60, "variation_rank_min": .60, "entry_delay_minutes": 5, "hold_hours": 8, "leverage": .5, "base_cost_per_notional_side": .0006, "stress_cost_per_notional_side": .001},
        "stages": contract["stages"], "source_support_gates": contract["source_support_gates"],
        "gross9_novelty_gates": contract["gross9_novelty_gates"], "economic_gates": contract["economic_gates"],
        "source_plan": {"open_interest": "Postgres open_interest_binance BTCUSDT period=5m exact endpoints", "bars": "Postgres bars_binance BTCUSDT coherent 1m OHLC, quote_asset_volume, number_of_trades", "window": ["2020-01-01T00:00:00Z", "2026-08-01T00:00:00Z"], "read_after_preregistration": True, "execution_prices": "sealed until source and Gross9 pass"},
        "research_boundary": {"prior_ticket_candidate_outcomes_known": True, "prior_HVCATAIR_source_novelty_and_train_failure_known": True, "same_mechanism_as_HVCATAIR": False, "prior_outcomes_used_to_set_formula_rank_side_hold_or_clock": False, "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False, "classification": "exploratory discovery; not fresh confirmatory evidence"},
        "stopping_rule": "terminal first failure; no ticket functional, impact statistic, OI rule, history, side, clock, hold, subset, threshold, comparator, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}

def validate(value: Mapping[str, Any]) -> None:
    if value != build(): raise RuntimeError("HVCTIIR-8 preregistration drift")
    contract = hvcav.build()
    for key in ("stages", "source_support_gates", "gross9_novelty_gates", "economic_gates"):
        if value[key] != contract[key]: raise RuntimeError(f"HVCTIIR-8 {key} drift")

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); args=parser.parse_args()
    value=build(); validate(value); args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n"); print(args.output)
