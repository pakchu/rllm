"""Outcome-blind preregistration for HVDO-Z14-5-3-9-24."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVDO-Z14-5-3-9-24"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_derivative_oscillator_zero_cross_relay_preregistration_2026-08-11.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    candidate = copy.deepcopy(template.build())
    candidate.pop("manifest_hash")
    candidate.update(
        protocol_version="high_volatility_derivative_oscillator_zero_cross_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-11",
        mechanism={
            "claim": (
                "During elevated realized variation, a completed four-hour zero crossing of the "
                "canonical Derivative Oscillator identifies a new acceleration regime in "
                "double-smoothed RSI relative to its signal line; follow the crossing side for "
                "twenty-four hours."
            ),
            "side": "long when the Derivative Oscillator newly crosses above zero; short when it newly crosses below zero",
            "why_distinct": (
                "The Derivative Oscillator applies Wilder RSI(14), sequential EMA(5) and EMA(3), "
                "then subtracts a nine-bar SMA of the double-smoothed RSI. It is not raw RSI "
                "extremes, MACD on price, stochastic, typical-price WaveTrend, close-only moving "
                "average crossover, volume, flow, OI, funding, fitted outcome, reused event set, "
                "or promoted control."
            ),
            "why_suited_to_volatile_regimes": (
                "the oscillator detects momentum acceleration while completed trailing "
                "twenty-four-hour realized variation must rank in its causal upper 35%"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "four-hour crossings of an RSI-derivative MACD-style histogram are absent from "
                "Gross9 primitives"
            ),
        },
        external_basis={
            "origin": "Constance Brown Derivative Oscillator; QuantConnect LEAN canonical implementation",
            "definition_source": "https://github.com/QuantConnect/Lean/blob/master/Indicators/DerivativeOscillator.cs",
            "documentation_source": (
                "https://www.quantconnect.com/docs/v2/writing-algorithms/indicators/"
                "supported-indicators/derivative-oscillator"
            ),
            "fixed_definition": (
                "documented parameters RSI=14, first EMA=5, second EMA=3, signal SMA=9; "
                "Wilder RSI; oscillator=double_smoothed_RSI-SMA(double_smoothed_RSI,9); "
                "warmup 29 completed bars"
            ),
            "selection_use": "official formula, documented parameters, and histogram zero-cross direction only; no incidence or outcomes",
        },
        features={
            "decision_grid": "every exact four-hour UTC boundary",
            "source_bar": "exact aggregation of 240 coherent BTCUSDT one-minute rows [T-4h,T)",
            "rsi": "Wilder RSI of completed four-hour closes with period 14",
            "smoothed_rsi": "LEAN EMA of ready RSI values with period 5",
            "double_smoothed_rsi": "LEAN EMA of ready first-smoothed RSI values with period 3",
            "signal": "simple mean of the latest nine ready double-smoothed RSI values",
            "derivative_oscillator": "double_smoothed_rsi-signal",
            "event": (
                "oscillator newly strictly above zero from not above gives side +1; oscillator "
                "newly strictly below zero from not below gives side -1"
            ),
            "variation": (
                "sqrt(sum squared completed five-minute open-to-close log returns over [T-24h,T)), "
                "finite strict positive"
            ),
            "variation_rank": (
                "strict-prior midrank over at most 180 earlier valid decisions, minimum 120, "
                "current excluded; rank>=0.65"
            ),
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
            "rsi_periods": 14,
            "first_ema_periods": 5,
            "second_ema_periods": 3,
            "signal_periods": 9,
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
            "names": ["no_variation_gate", "raw_rsi_midline", "one_bar_stale_cross", "direction_flip"],
            "cannot_be_promoted": True,
        },
        source_plan={
            "bars": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close"],
                "window": ["2020-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        research_boundary={
            "canonical_derivative_oscillator_definition_read": True,
            "repository_derivative_oscillator_candidate_found": False,
            "prior_rsi_and_oscillator_candidate_outcomes_known": True,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "official Derivative Oscillator(14,5,3,9) zero crossing under the requested high-variation regime",
        },
        stopping_rule=(
            "Terminal first failure; no RSI, EMA, signal period, smoothing, crossing, variation, "
            "side, hold, clock, subset, control, or other repair."
        ),
    )
    return {**candidate, "manifest_hash": canonical_hash(candidate)}


def validate(candidate: dict[str, Any]) -> None:
    core = {key: value for key, value in candidate.items() if key != "manifest_hash"}
    if candidate.get("manifest_hash") != canonical_hash(core) or candidate != build():
        raise RuntimeError("HVDO-Z14-5-3-9-24 preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    registration = build()
    validate(registration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(registration, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
