"""Outcome-blind preregistration for HVCFDR-8."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any, Mapping
from training import preregister_high_volatility_cross_structure_action_vote as hvcav

POLICY_ID = "HVCFDR-8"
DEFAULT_OUTPUT = Path("results/high_volatility_causal_funding_delta_reversal_preregistration_2026-08-16.json")

def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def build() -> dict[str, Any]:
    contract = hvcav.build()
    core = {
        "protocol_version": "high_volatility_causal_funding_delta_reversal_v1",
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
            "claim": "In a causally high-variation funding cycle, an unusually large one-settlement funding-rate change is a leverage-demand shock that mean-reverts over the next settlement cycle.",
            "side": "negative strict sign of funding_rate[S]-funding_rate[P]",
            "why_distinct": "Unlike HVFADR, this policy does not require or follow price/funding divergence, has no onset rule, and interprets the funding first difference itself as a crowded shock to fade.",
            "why_suited_to_volatile_regimes": "the completed intervening eight-hour BTC realized variation must be in its causal upper 40 percent",
            "why_low_gross9_overlap_is_plausible": "actual funding-settlement first-difference shocks are absent from Gross9 primitives",
        },
        "features": {
            "decision": "actual BTCUSDT funding settlement S",
            "previous_settlement": "immediately prior actual settlement P with S-P exactly 8 elapsed hours",
            "funding_change": "finite nonzero funding_rate[S]-funding_rate[P]",
            "change_rank": "strict-prior midrank of absolute funding_change over at most 270 source-valid cycles; current excluded; minimum 180; rank>=0.70",
            "realized_variation": "sqrt(sum squared one-minute log(close/open) returns over exactly 480 coherent BTCUSDT bars [P,S))",
            "variation_rank": "strict-prior midrank over at most 270 source-valid cycles; current excluded; minimum 180; rank>=0.60",
            "eligible": "both ranks pass at S; no divergence, onset, funding-level, OI, premium, alt, or post-settlement condition",
            "no_imputation": True,
        },
        "clock": {
            "decisions": "actual settlements, normally 00/08/16 UTC",
            "entry": "exact BTCUSDT S+5m open",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal entry",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "signal settlement precedes entry; exact held settlements only after source and Gross9 pass",
        },
        "policy": {
            "required_settlement_gap_hours": 8,
            "history_cycles": 270,
            "minimum_history_cycles": 180,
            "absolute_change_rank_min": 0.70,
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
            "funding": "Postgres funding_rates_binance BTCUSDT actual timestamps/rates",
            "bars": "Postgres bars_binance BTCUSDT exact coherent 1m OHLC",
            "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "read_after_preregistration": True,
            "execution_prices": "sealed until source support and Gross9 pass",
        },
        "research_boundary": {
            "prior_HVFADR_source_incidence_and_failure_known": True,
            "prior_HVFADR_outcomes_opened": False,
            "same_mechanism_as_HVFADR": False,
            "prior_outcomes_used_to_set_formula_rank_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "classification": "exploratory discovery; not fresh confirmatory evidence",
        },
        "stopping_rule": "terminal first failure; no settlement gap, history, rank, side, variation, clock, hold, subset, threshold, comparator, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}

def validate(value: Mapping[str, Any]) -> None:
    if value != build():
        raise RuntimeError("HVCFDR-8 preregistration drift")
    contract = hvcav.build()
    for key in ("stages", "source_support_gates", "gross9_novelty_gates", "economic_gates"):
        if value[key] != contract[key]:
            raise RuntimeError(f"HVCFDR-8 {key} drift")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    value = build(); validate(value); args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
