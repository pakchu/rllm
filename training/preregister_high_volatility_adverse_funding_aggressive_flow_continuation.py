"""Outcome-blind preregistration for HVAFGF-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from training import preregister_high_volatility_cross_structure_action_vote as hvcav


POLICY_ID = "HVAFGF-8"
DEFAULT_OUTPUT = Path("results/high_volatility_adverse_funding_aggressive_flow_continuation_preregistration_2026-08-16.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    contract = hvcav.build()
    core = {
        "protocol_version": "high_volatility_adverse_funding_aggressive_flow_continuation_v1",
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
                "At an actual funding settlement after a high-variation eight-hour cycle, "
                "same-direction BTC perpetual price and aggregate aggressive quote flow identify "
                "active directional demand. If the settled funding sign is opposite that demand, "
                "the move is not yet supported by crowded same-direction carry; continue in the "
                "price-flow direction for eight hours."
            ),
            "side": "common strict sign of completed perpetual return and aggregate aggressive quote flow",
            "why_distinct": (
                "HVAFGF combines actual adverse settled funding with completed price-flow "
                "concordance. It uses no spot, OI, alt, FX, post-settlement confirmation, "
                "funding-rank tail, fitted outcome, or promoted control."
            ),
            "why_suited_to_volatile_regimes": "completed BTC perpetual variation must be in its causal upper 40 percent",
            "why_low_gross9_overlap_is_plausible": "actual-settlement adverse carry joined to eight-hour aggressive-flow concordance is absent from Gross9 primitives",
        },
        "features": {
            "decision": "actual BTCUSDT funding settlement S",
            "funding_rate": "finite strict nonzero actual funding_rate[S], available at S",
            "cycle": "480 exact coherent BTCUSDT perpetual one-minute rows [S-8h,S) with valid quote turnover and taker-buy quote",
            "price_return": "sum of 480 minute ln(close/open) returns, finite strict nonzero",
            "flow_share": "sum(2*taker_buy_quote-quote_asset_volume)/sum(quote_asset_volume), finite strict nonzero",
            "price_flow_concordance": "price_return and flow_share have one strict sign",
            "funding_adversity": "funding_rate[S] has the strict opposite sign to price_return and flow_share",
            "realized_variation": "sqrt(sum squared one-minute price returns), strict positive",
            "variation_rank": "strict-prior midrank over at most 270 source-valid settlement cycles; current excluded; minimum 180; rank>=0.60",
            "eligible": "exact source-valid cycle, price-flow concordance, adverse settled funding, and variation rank passes",
            "side": "strict sign of flow_share",
            "no_imputation": True,
        },
        "clock": {
            "decisions": "actual BTCUSDT funding settlements, normally 00/08/16 UTC",
            "entry": "exact BTCUSDT S+5m open",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal entry",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "trigger settlement precedes entry; exact held settlements only after source and Gross9 pass",
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
            "funding": "Postgres funding_rates_binance BTCUSDT actual timestamps/rates",
            "bars": "Postgres bars_binance BTCUSDT exact coherent 1m OHLC, quote_asset_volume, and taker_buy_quote",
            "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "read_after_preregistration": True,
            "execution_prices": "sealed until source support and Gross9 pass",
        },
        "research_boundary": {
            "prior_price_flow_OI_router_gross9_failure_known": True,
            "prior_taker_flow_decay_train_failure_known": True,
            "prior_funding_family_source_failures_known": True,
            "exact_adverse_funding_price_flow_conjunction_incidence_or_outcomes_known": False,
            "user_authorized_multi_condition_composition": True,
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
        "stopping_rule": "terminal first failure; no settlement, source, price-flow, funding, variation, side, clock, hold, subset, threshold, comparator, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: Mapping[str, Any]) -> None:
    if value != build():
        raise RuntimeError("HVAFGF-8 preregistration drift")
    contract = hvcav.build()
    for key in ("stages", "source_support_gates", "gross9_novelty_gates", "economic_gates"):
        if value[key] != contract[key]:
            raise RuntimeError(f"HVAFGF-8 {key} drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = build()
    validate(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
