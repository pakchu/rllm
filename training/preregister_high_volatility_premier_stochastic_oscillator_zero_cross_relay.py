"""Outcome-blind preregistration for HVPSO-Z14-3-24."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVPSO-Z14-3-24"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_premier_stochastic_oscillator_zero_cross_relay_preregistration_2026-08-11.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    candidate = copy.deepcopy(template.build())
    candidate.pop("manifest_hash")
    candidate.update(
        protocol_version="high_volatility_premier_stochastic_oscillator_zero_cross_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-11",
        mechanism={
            "claim": (
                "During elevated realized variation, a completed four-hour zero crossing of the "
                "canonical Premier Stochastic Oscillator identifies a new bounded range-position "
                "trend after double exponential smoothing; follow the crossing for twenty-four hours."
            ),
            "side": "long when PSO newly crosses above zero; short when PSO newly crosses below zero",
            "why_distinct": (
                "Premier Stochastic transforms fourteen-bar Fast %K around 50, applies two "
                "sequential EMA(3) smoothers, then maps the result through a bounded exponential "
                "ratio. It is not ordinary stochastic K/D crossing, Stochastic Momentum Index, "
                "Stochastic RSI, raw range breakout, WaveTrend, volume, flow, OI, funding, fitted "
                "outcome, reused event set, or promoted control."
            ),
            "why_suited_to_volatile_regimes": (
                "the input is completed-bar position within a causal high-low envelope and "
                "trailing twenty-four-hour realized variation must rank in its causal upper 35%"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "four-hour zero crossings of a double-smoothed bounded stochastic transform are "
                "absent from Gross9 primitives"
            ),
        },
        external_basis={
            "origin": "Premier Stochastic Oscillator; QuantConnect LEAN canonical implementation",
            "definition_source": "https://github.com/QuantConnect/Lean/blob/master/Indicators/PremierStochasticOscillator.cs",
            "documentation_source": (
                "https://www.quantconnect.com/docs/v2/writing-algorithms/indicators/"
                "supported-indicators/premier-stochastic-oscillator"
            ),
            "fixed_definition": (
                "documented parameters period=14 and EMA period=3; FastK=100*(close-lowestLow14)/"
                "(highestHigh14-lowestLow14); NSK=0.1*(FastK-50); SS=EMA(EMA(NSK,3),3); "
                "PSO=(exp(SS)-1)/(exp(SS)+1); warmup 18 completed bars"
            ),
            "selection_use": "official formula, documented parameters, and signed zero crossing only; no incidence or outcomes",
        },
        features={
            "decision_grid": "every exact four-hour UTC boundary",
            "source_bar": "exact aggregation of 240 coherent BTCUSDT one-minute rows [T-4h,T)",
            "fast_k": "100*(close-lowest low over 14)/(highest high over 14-lowest low over 14), finite positive range",
            "normalized_stochastic": "0.1*(fast_k-50)",
            "first_smoothing": "LEAN EMA of normalized stochastic with period 3",
            "double_smoothing": "LEAN EMA of ready first smoothing with period 3",
            "pso": "(exp(double_smoothing)-1)/(exp(double_smoothing)+1)",
            "event": "PSO newly strictly above zero from not above gives +1; newly strictly below zero from not below gives -1",
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
            "stochastic_periods": 14,
            "ema_periods": 3,
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
            "names": ["no_variation_gate", "raw_fast_k_midline", "one_bar_stale_cross", "direction_flip"],
            "cannot_be_promoted": True,
        },
        source_plan={
            "bars": {
                "table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close"],
                "window": ["2020-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        research_boundary={
            "canonical_premier_stochastic_definition_read": True,
            "repository_premier_stochastic_candidate_found": False,
            "prior_stochastic_family_candidate_outcomes_known": True,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "official Premier Stochastic Oscillator(14,3) zero crossing under the requested high-variation regime",
        },
        stopping_rule=(
            "Terminal first failure; no stochastic or EMA period, transform, crossing, variation, "
            "side, hold, clock, subset, control, or other repair."
        ),
    )
    return {**candidate, "manifest_hash": canonical_hash(candidate)}


def validate(candidate: dict[str, Any]) -> None:
    core = {key: value for key, value in candidate.items() if key != "manifest_hash"}
    if candidate.get("manifest_hash") != canonical_hash(core) or candidate != build():
        raise RuntimeError("HVPSO-Z14-3-24 preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(); registration = build(); validate(registration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(registration, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
