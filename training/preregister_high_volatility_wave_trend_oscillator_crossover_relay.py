"""Outcome-blind preregistration for HVWTO-C10-21-4-24."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVWTO-C10-21-4-24"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_wave_trend_oscillator_crossover_relay_preregistration_2026-08-11.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    candidate = copy.deepcopy(template.build())
    candidate.pop("manifest_hash")
    candidate.update(
        protocol_version="high_volatility_wave_trend_oscillator_crossover_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-11",
        mechanism={
            "claim": (
                "During elevated realized variation, a completed four-hour bullish or bearish "
                "crossover of the canonical WaveTrend main line and signal line identifies a "
                "new volatility-normalized typical-price momentum wave; follow the crossover "
                "for twenty-four hours."
            ),
            "side": "long on a new WT1 cross above WT2; short on a new WT1 cross below WT2",
            "why_distinct": (
                "WaveTrend normalizes typical-price displacement by an exponentially smoothed "
                "mean absolute deviation, then applies a second EMA and a four-bar SMA signal. "
                "It is not MACD, RSI, stochastic, close-only moving-average crossover, fixed "
                "channel, volume, flow, OI, funding, fitted outcome, reused event set, or "
                "promoted control."
            ),
            "why_suited_to_volatile_regimes": (
                "the channel index scales price displacement by causal local absolute deviation "
                "and completed trailing twenty-four-hour realized variation must rank in its "
                "causal upper 35%"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "four-hour crossovers of a chained typical-price deviation-normalized oscillator "
                "are absent from Gross9 primitives"
            ),
        },
        external_basis={
            "origin": "WaveTrend Oscillator; QuantConnect LEAN canonical implementation and documentation",
            "definition_source": "https://github.com/QuantConnect/Lean/blob/master/Indicators/WaveTrendOscillator.cs",
            "documentation_source": (
                "https://www.quantconnect.com/docs/v2/writing-algorithms/indicators/"
                "supported-indicators/wave-trend-oscillator"
            ),
            "fixed_definition": (
                "documented example parameters channel=10, average=21, signal=4; "
                "HLC3=(high+low+close)/3; ESA=EMA(HLC3,10); D=EMA(abs(HLC3-ESA),10); "
                "CI=(HLC3-ESA)/(0.015*D); WT1=EMA(CI,21); WT2=SMA(WT1,4); "
                "warmup 42 completed bars"
            ),
            "selection_use": (
                "official formula, documented parameters, and documented WT1/WT2 crossover "
                "interpretation only; no incidence or outcomes"
            ),
        },
        features={
            "decision_grid": "every exact four-hour UTC boundary",
            "source_bar": "exact aggregation of 240 coherent BTCUSDT one-minute rows [T-4h,T)",
            "typical_price": "(high+low+close)/3 on a valid completed four-hour bar",
            "channel_average": "LEAN EMA of typical price with period 10",
            "channel_deviation": "LEAN EMA of abs(typical_price-channel_average) with period 10",
            "channel_index": "(typical_price-channel_average)/(0.015*channel_deviation), finite positive denominator",
            "wt1": "LEAN EMA of channel_index with period 21",
            "wt2": "simple mean of the latest four ready WT1 observations",
            "event": (
                "WT1 newly strictly above WT2 from not above gives side +1; WT1 newly strictly "
                "below WT2 from not below gives side -1"
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
            "channel_periods": 10,
            "average_periods": 21,
            "signal_periods": 4,
            "normalization_constant": 0.015,
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
            "names": ["no_variation_gate", "zero_cross", "one_bar_stale_cross", "direction_flip"],
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
            "canonical_wave_trend_definition_read": True,
            "repository_wave_trend_candidate_found": False,
            "prior_oscillator_candidate_outcomes_known": True,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": (
                "official WaveTrend(10,21,4) line crossover under the requested high-variation regime"
            ),
        },
        stopping_rule=(
            "Terminal first failure; no channel, average, signal period, normalization, smoothing, "
            "crossing, variation, side, hold, clock, subset, control, or other repair."
        ),
    )
    return {**candidate, "manifest_hash": canonical_hash(candidate)}


def validate(candidate: dict[str, Any]) -> None:
    core = {key: value for key, value in candidate.items() if key != "manifest_hash"}
    if candidate.get("manifest_hash") != canonical_hash(core) or candidate != build():
        raise RuntimeError("HVWTO-C10-21-4-24 preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    registration = build()
    validate(registration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(registration, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
