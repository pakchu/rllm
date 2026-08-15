"""Outcome-blind preregistration for HVCOSR-24."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any, Mapping
from training import preregister_high_volatility_cross_structure_action_vote as hvcav

POLICY_ID = "HVCOSR-24"
DEFAULT_OUTPUT = Path("results/high_volatility_causal_oi_semivariance_sponsorship_relay_preregistration_2026-08-16.json")

def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def build() -> dict[str, Any]:
    contract = hvcav.build()
    core = {
        "protocol_version": "high_volatility_causal_oi_semivariance_sponsorship_relay_v1",
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
            "claim": "When a completed UTC-day BTC path has a large signed upside-versus-downside realized-semivariance imbalance, total realized variation is causally elevated, and open interest expands over the same day, newly built leverage sponsors the dominant intraday shock direction for twenty-four hours.",
            "side": "strict sign of upside_semivariance-downside_semivariance",
            "why_distinct": "HVCOSR uses signed one-minute realized-semivariance imbalance plus exact daily endpoint OI expansion. HVSIR used BTC semivariance reversal without OI and failed Gross9; HVCASI used cross-alt six-hour consensus without OI and failed train. HVCOSR reuses neither event set nor control and does not use daily body, GK variance, funding, premium, fitted outcome, or promoted control.",
            "why_suited_to_volatile_regimes": "the completed twenty-four-hour BTC realized variation must be in its causal upper 40 percent",
            "why_low_gross9_overlap_is_plausible": "daily OI-sponsored signed semivariance states are absent from Gross9 primitives",
        },
        "features": {
            "decision": "every exact 00:00 UTC boundary D",
            "open_interest": "finite positive BTCUSDT period=5m sum_open_interest observations at D-24h and D, each exchange-timestamped no later than D",
            "oi_expansion": "strict positive log(OI[D]/OI[D-24h])",
            "completed_day": "1,440 exact coherent BTCUSDT one-minute OHLC rows over [D-24h,D)",
            "upside_semivariance": "sum squared positive one-minute log(close/open) returns over exactly 1,440 coherent rows",
            "downside_semivariance": "sum squared negative one-minute log(close/open) returns over exactly 1,440 coherent rows",
            "semivariance_imbalance": "(upside_semivariance-downside_semivariance)/(upside_semivariance+downside_semivariance), finite strict nonzero",
            "absolute_imbalance_rank": "strict-prior midrank of absolute semivariance imbalance over at most 270 source-valid days; current excluded; minimum 180; rank>=0.60",
            "realized_variation": "sqrt(sum squared one-minute log(close/open) returns over exactly 1,440 coherent BTCUSDT rows [D-24h,D))",
            "variation_rank": "strict-prior midrank over at most 270 source-valid days; current excluded; minimum 180; rank>=0.60",
            "eligible": "exact coherent completed day, two exact OI endpoints with strict expansion, nonzero semivariance imbalance, and both absolute-imbalance and variation ranks pass at D; no funding, range, midpoint, premium, alt, or post-decision condition",
            "no_imputation": True,
        },
        "clock": {
            "decisions": "exact 00:00 UTC boundaries",
            "entry": "exact BTCUSDT D+5m open",
            "hold": "24 elapsed hours",
            "reservation": "global half-open; exit first on equal entry",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact held settlements only after source and Gross9 pass",
        },
        "policy": {
            "history_cycles": 270,
            "minimum_history_cycles": 180,
            "day_minutes": 1440,
            "absolute_imbalance_rank_min": 0.60,
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
            "open_interest": "Postgres open_interest_binance BTCUSDT period=5m exact endpoint observations",
            "bars": "Postgres bars_binance BTCUSDT exact coherent 1m OHLC",
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
            "prior_HVSIR_source_and_gross9_failure_known": True,
            "same_mechanism_as_HVSIR": False,
            "prior_HVCASI_source_novelty_and_train_failure_known": True,
            "same_mechanism_as_HVCASI": False,
            "same_mechanism_as_HVCOPR": False,
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
        "stopping_rule": "terminal first failure; no OI endpoint or expansion rule, semivariance formula, imbalance or variation history, side, clock, hold, subset, threshold, comparator, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}

def validate(value: Mapping[str, Any]) -> None:
    if value != build():
        raise RuntimeError("HVCOSR-24 preregistration drift")
    contract = hvcav.build()
    for key in ("stages", "source_support_gates", "gross9_novelty_gates", "economic_gates"):
        if value[key] != contract[key]:
            raise RuntimeError(f"HVCOSR-24 {key} drift")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    value = build(); validate(value); args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
