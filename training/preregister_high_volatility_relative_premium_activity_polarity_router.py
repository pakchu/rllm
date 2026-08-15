"""Outcome-blind preregistration for HVPRAPR-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from training import preregister_high_volatility_cross_structure_action_vote as hvcav


POLICY_ID = "HVPRAPR-12"
DEFAULT_OUTPUT = Path("results/high_volatility_relative_premium_activity_polarity_router_preregistration_2026-08-16.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    contract = hvcav.build()
    core = {
        "protocol_version": "high_volatility_relative_premium_activity_polarity_router_v1",
        "policy_id": POLICY_ID, "as_of_date": "2026-08-16", "singleton": True,
        "exploratory_discovery": True, "fresh_confirmatory_evidence": False,
        "source_incidence_opened": False, "outcomes_opened": False, "gross9_rows_opened": False,
        "candidate_family": [POLICY_ID], "candidate_family_size": 1,
        "mechanism": {
            "claim": (
                "Relative premium activity separates two high-volatility mechanisms. When premium "
                "activity is low relative to BTC variation, follow the spot-led move; when it is "
                "high, fade derivatives-heavy overreaction. Route the completed-day direction for twelve hours."
            ),
            "side": "completed BTC direction in the lower relative-activity tail and its negative in the upper tail",
            "why_distinct": (
                "HVPRAPR requires the lower tail of unsigned premium path activity jointly with the "
                "upper tail of BTC variation. It uses no premium sign/displacement, funding, OI, "
                "spot table, alt, fitted outcome, prior event clock, or promoted control."
            ),
            "why_suited_to_volatile_regimes": "completed BTC daily realized variation must rank at least 0.65",
            "why_low_gross9_overlap_is_plausible": "one daily two-sided relative-premium-activity routing decision is absent from Gross9 primitives",
        },
        "features": {
            "source_day": "exact prior UTC day [D-24h,D)",
            "premium_path": "1,440 exact unique bars_binance_premium BTCUSDT one-minute closes, finite signed values",
            "premium_total_variation": "sum(abs(close[i]-close[i-1])) over 1,439 within-day pairs, strict positive",
            "btc_path": "288 exact coherent BTCUSDT five-minute aggregates from 1,440 one-minute rows over the same day",
            "btc_return": "log(final five-minute close/first five-minute open), strict nonzero",
            "btc_realized_variation": "sqrt(sum squared five-minute open-to-close log returns), strict positive",
            "relative_premium_activity": "premium_total_variation/btc_realized_variation, finite strict positive",
            "causal_ranks": "strict-prior midranks of relative premium activity and BTC variation over at most 270 earlier source-valid days, minimum 180; current excluded",
            "relative_activity_tails": "lower rank<=0.25 or upper rank>=0.75; middle ineligible",
            "routing": "lower tail follows btc_return; upper tail fades btc_return",
            "high_variation": "BTC realized-variation rank>=0.65",
            "eligible": "source valid, either relative-activity tail, high BTC variation, and strict nonzero BTC return",
            "availability": "D 00:00 UTC after both source paths complete",
            "side": "sign(btc_return) in lower tail and -sign(btc_return) in upper tail", "no_imputation": True,
        },
        "clock": {
            "decision": "exact 00:00 UTC D after the completed source day",
            "entry": "exact BTCUSDT perpetual D+5m open", "hold": "12 elapsed hours",
            "reservation": "daily opportunities do not overlap", "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact held settlements only after source and Gross9 pass",
        },
        "policy": {
            "premium_minutes": 1440, "btc_bars_5m": 288,
            "history_days": 270, "minimum_history_days": 180,
            "lower_relative_activity_rank_max": 0.25, "upper_relative_activity_rank_min": 0.75, "btc_variation_rank_min": 0.65,
            "entry_delay_minutes": 5, "hold_hours": 12, "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006, "stress_cost_per_notional_side": 0.0010,
        },
        "stages": contract["stages"], "source_support_gates": contract["source_support_gates"],
        "gross9_novelty_gates": contract["gross9_novelty_gates"], "economic_gates": contract["economic_gates"],
        "source_plan": {
            "premium": "Postgres bars_binance_premium BTCUSDT exact 1m closes",
            "btc": "Postgres bars_binance BTCUSDT exact coherent 1m OHLC aggregated to 5m",
            "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "read_after_preregistration": True, "execution_prices": "sealed until source support and Gross9 pass",
        },
        "research_boundary": {
            "prior_high_premium_activity_HVPASR_train_failure_known": True,
            "prior_absolute_HVPQC_source_failure_known": True,
            "prior_relative_HVPRQC_source_failure_known": True,
            "two_sided_relative_router_control_existed": False,
            "exact_two_sided_router_incidence_or_outcomes_known": False,
            "prior_outcomes_used_to_set_rank_side_hold_or_clock": False,
            "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False, "candidate_count": 1, "grid": False,
            "repair_of_prior_candidate": False, "promoted_prior_control": False,
            "classification": "exploratory discovery; not fresh confirmatory evidence",
        },
        "stopping_rule": "terminal first failure; no source, relative-activity tails, variation rank, side, clock, hold, subset, threshold, comparator, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: Mapping[str, Any]) -> None:
    if value != build(): raise RuntimeError("HVPRAPR-12 preregistration drift")
    contract = hvcav.build()
    for key in ("stages", "source_support_gates", "gross9_novelty_gates", "economic_gates"):
        if value[key] != contract[key]: raise RuntimeError(f"HVPRAPR-12 {key} drift")


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);args=parser.parse_args()
    value=build();validate(value);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n");print(args.output)
