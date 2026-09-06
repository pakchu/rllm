"""Outcome-blind preregistration for HVTFDAC-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from training import preregister_high_volatility_cross_structure_action_vote as contract


POLICY_ID = "HVTFDAC-8"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_taker_flow_decay_acceptance_continuation_preregistration_2026-08-16.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    frozen = contract.build()
    core = {
        "protocol_version": "high_volatility_taker_flow_decay_acceptance_continuation_v1",
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
                "After a causally high-variation completed BTC eight-hour cycle, lower quadratic variation "
                "in the second four hours together with unchanged return direction and higher directional "
                "efficiency indicates volatility decay into price acceptance; follow that accepted direction for eight hours."
            ),
            "side": "common strict sign of first-half and second-half completed returns",
            "why_distinct": (
                "HVTFDAC uses within-cycle variance decay and an efficiency ordering. Prior temporal-variance "
                "ignition concentrated variance late over a 24-hour daily clock; this candidate requires the "
                "opposite variance ordering on fixed eight-hour cycles and reuses no event set or control."
            ),
            "why_suited_to_volatile_regimes": "the completed eight-hour variation must rank in its causal upper 35 percent",
            "why_low_gross9_overlap_is_plausible": "taker-pressure decay plus price-direction acceptance is absent from Gross9 primitives",
        },
        "features": {
            "decision": "exact 00:00, 08:00, or 16:00 UTC boundary D",
            "cycle": "480 exact coherent BTCUSDT bars_binance interval=1m rows [D-8h,D)",
            "minute_return": "ln(close/open) for each completed minute; finite",
            "first_half_return": "sum minute_return over [D-8h,D-4h), strict nonzero",
            "second_half_return": "sum minute_return over [D-4h,D), strict nonzero",
            "direction_agreement": "first-half and second-half returns have one strict sign",
            "first_half_flow_imbalance": "sum(2*taker_buy_quote-quote_asset_volume)/sum(quote_asset_volume) over the first 240 bars, finite",
            "second_half_flow_imbalance": "the same ratio over the final 240 bars, finite",
            "directional_flow_decay": "with side=sign(first_half_return), 0 < side*second_half_flow_imbalance < side*first_half_flow_imbalance; strict with no fitted threshold",
            "realized_variation": "sqrt(first_half_qv+second_half_qv), strict positive",
            "variation_rank": "strict-prior midrank over at most 270 source-valid cycles; current excluded; minimum 180; rank>=0.65",
            "onset": "eligible now and immediately prior exact source-valid cycle ineligible; missing prior opportunity cannot trigger",
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
            "variation_rank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 8,
            "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        "stages": frozen["stages"],
        "source_support_gates": frozen["source_support_gates"],
        "gross9_novelty_gates": frozen["gross9_novelty_gates"],
        "economic_gates": frozen["economic_gates"],
        "source_plan": {
            "bars": "Postgres bars_binance BTCUSDT exact coherent interval=1m OHLC, quote_asset_volume, and taker_buy_quote",
            "window": ["2023-04-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "read_after_preregistration": True,
            "execution_prices": "sealed until source support and Gross9 pass",
        },
        "research_boundary": {
            "prior_taker_persistence_concentration_seasonality_failures_known": True,
            "prior_flow_microstructure_family_outcomes_known": True,
            "exact_taker_flow_decay_acceptance_incidence_or_outcomes_known": False,
            "prior_event_sets_reused": False,
            "prior_outcomes_used_to_set_rank_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "classification": "exploratory discovery; not fresh confirmatory evidence",
        },
        "stopping_rule": (
            "terminal first failure; no taker-flow decay ordering, variation history, side, "
            "clock, hold, subset, threshold, comparator, or control repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: Mapping[str, Any]) -> None:
    if value != build():
        raise RuntimeError("HVTFDAC-8 preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = build()
    validate(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
