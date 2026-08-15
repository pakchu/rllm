"""Outcome-blind preregistration for HVCEDOIR-24."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any, Mapping
from training import preregister_high_volatility_cross_structure_action_vote as hvcav

POLICY_ID = "HVCEDOIR-24"
DEFAULT_OUTPUT = Path("results/high_volatility_causal_extreme_displacement_open_interest_reversal_preregistration_2026-08-16.json")

def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def build() -> dict[str, Any]:
    contract = hvcav.build()
    core = {
        "protocol_version": "high_volatility_causal_extreme_displacement_open_interest_reversal_v1",
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
            "claim": "During causally elevated twenty-four-hour variation, an extreme current four-hour high or low displacement beyond the trailing close envelope accompanied by exact open-interest expansion identifies a newly levered intrabar excursion unsupported by prior accepted closes and should reverse over twenty-four hours.",
            "side": "long when normalized downside displacement dominates; short when normalized upside displacement dominates",
            "why_distinct": "HVCEDOIR combines current-auction extrema versus the prior 22 accepted closes with exact same-auction OI expansion. The original HVCEDR source incidence and Gross9 failure are known but its economic outcomes were never opened. It uses no funding, flow, ticket, fitted channel, reused composite event set, or promoted control.",
            "why_suited_to_volatile_regimes": "the completed intervening eight-hour BTC realized variation must be in its causal upper 40 percent",
            "why_low_gross9_overlap_is_plausible": "four-hour close-envelope versus current-extreme displacement clocks are absent from Gross9 primitives",
        },
        "features": {
            "decision": "every exact four-hour UTC boundary D",
            "current_auction": "240 exact coherent BTCUSDT one-minute OHLC rows over [D-4h,D)",
            "open_interest": "finite positive BTCUSDT period=5m sum_open_interest observations at D-4h and D with strict positive log(OI[D]/OI[D-4h])",
            "prior_close_envelope": "highest and lowest last close from the 22 uninterrupted completed four-hour auctions ending at D-4h; current auction excluded",
            "downside_displacement": "(highest_prior_close-current_low)/highest_prior_close, strict positive candidate",
            "upside_displacement": "(current_high-lowest_prior_close)/lowest_prior_close, strict positive candidate",
            "dominant_displacement": "strictly larger positive displacement; ties or two nonpositive values invalid",
            "displacement_rank": "strict-prior midrank of dominant displacement over at most 540 source-valid decisions; current excluded; minimum 360; rank>=0.70",
            "realized_variation": "sqrt(sum squared one-minute log(close/open) returns over exactly 1,440 coherent BTCUSDT rows [D-24h,D))",
            "variation_rank": "strict-prior midrank over at most 540 source-valid decisions; current excluded; minimum 360; rank>=0.60",
            "eligible": "23 uninterrupted coherent four-hour auctions including current, one strict dominant displacement, with exact OI expansion and both ranks passing at D; no funding, premium, flow, ticket, alt, or post-decision condition",
            "no_imputation": True,
        },
        "clock": {
            "decisions": "exact 00/04/08/12/16/20 UTC boundaries",
            "entry": "exact BTCUSDT D+5m open",
            "hold": "24 elapsed hours",
            "reservation": "global half-open; exit first on equal entry",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact held settlements only after source and Gross9 pass",
        },
        "policy": {
            "history_decisions": 540,
            "minimum_history_decisions": 360,
            "prior_close_periods": 22,
            "current_auction_minutes": 240,
            "variation_minutes": 1440,
            "displacement_rank_min": 0.70,
            "variation_rank_min": 0.60,
            "entry_delay_minutes": 5,
            "hold_hours": 24,
            "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.0010,
        },
        "stages": contract["stages"],
        "source_support_gates": contract["source_support_gates"],
        "gross9_novelty_gates": contract["gross9_novelty_gates"],
        "economic_gates": contract["economic_gates"],
        "source_plan": {
            "bars": "Postgres bars_binance BTCUSDT exact coherent 1m OHLC",
            "open_interest": "Postgres open_interest_binance BTCUSDT period=5m exact endpoints",
            "window": ["2020-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "read_after_preregistration": True,
            "execution_prices": "sealed until source support and Gross9 pass",
        },
        "research_boundary": {
            "prior_HVFADR_source_incidence_and_failure_known": True,
            "prior_HVFADR_outcomes_opened": False,
            "same_mechanism_as_HVFADR": False,
            "prior_HVCFDR_source_incidence_and_failure_known": True,
            "prior_HVCFDR_outcomes_opened": False,
            "same_mechanism_as_HVCFDR": False,
            "prior_HVCFIR_source_incidence_and_failure_known": True,
            "prior_HVCFIR_outcomes_opened": False,
            "same_mechanism_as_HVCFIR": False,
            "prior_HVCFZR_source_incidence_and_failure_known": True,
            "prior_HVCFZR_outcomes_opened": False,
            "same_mechanism_as_HVCFZR": False,
            "prior_HVCFLR_source_incidence_and_failure_known": True,
            "prior_HVCFLR_outcomes_opened": False,
            "same_mechanism_as_HVCFLR": False,
            "prior_HVCFMR_source_novelty_and_train_failure_known": True,
            "same_mechanism_as_HVCFMR": False,
            "prior_HVCFBR_source_incidence_and_failure_known": True,
            "prior_HVCFBR_outcomes_opened": False,
            "same_mechanism_as_HVCFBR": False,
            "prior_HVCFRR_source_incidence_and_failure_known": True,
            "prior_HVCFRR_outcomes_opened": False,
            "same_mechanism_as_HVCFRR": False,
            "prior_HVCORR_source_novelty_and_train_failure_known": True,
            "same_mechanism_as_HVCORR": False,
            "prior_HVDPR_source_incidence_known": True,
            "prior_HVDPR_outcomes_opened": False,
            "same_mechanism_as_HVDPR": False,
            "prior_HVCOPR_source_novelty_and_train_failure_known": True,
            "same_mechanism_as_HVCOPR": False,
            "prior_HVDUDAR_outcomes_known": True,
            "same_mechanism_as_HVDUDAR": False,
            "original_HVCEDR_source_incidence_and_novelty_failure_known": True,
            "original_HVCEDR_economic_outcomes_opened": False,
            "exact_composite_incidence_or_outcomes_known": False,
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
        "stopping_rule": "terminal first failure; no auction, prior-close envelope, displacement formula, OI endpoint or expansion rule, rank history, variation, side, clock, hold, subset, threshold, comparator, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}

def validate(value: Mapping[str, Any]) -> None:
    if value != build():
        raise RuntimeError("HVCEDOIR-24 preregistration drift")
    contract = hvcav.build()
    for key in ("stages", "source_support_gates", "gross9_novelty_gates", "economic_gates"):
        if value[key] != contract[key]:
            raise RuntimeError(f"HVCEDOIR-24 {key} drift")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    value = build(); validate(value); args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
