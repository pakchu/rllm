"""Outcome-blind preregistration for HVKAMA-24."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVKAMA-24"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_kaufman_adaptive_average_crossover_relay_preregistration_2026-08-11.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    contract = copy.deepcopy(template.build())
    contract.pop("manifest_hash")
    contract.update(
        protocol_version="high_volatility_kaufman_adaptive_average_crossover_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-11",
        mechanism={
            "claim": "During elevated realized variation, a strict completed four-hour BTC close crossover of the canonical Kaufman Adaptive Moving Average marks a trend change whose smoothing speed is determined only by causal path efficiency; follow the new side for twenty-four hours.",
            "side": "long when completed close-minus-KAMA crosses strictly from negative to positive; short on the strict reverse crossover",
            "why_distinct": "MACD uses two fixed exponential speeds, time-price fit estimates a regression slope, and CHOP measures range-path efficiency only as an event gate. HVKAMA recursively changes its smoothing constant from the canonical ten-period efficiency ratio before testing price versus the adaptive state; it uses no volume, flow, OI, funding, fitted outcome, reused event set, or promoted control.",
            "why_suited_to_volatile_regimes": "completed trailing twenty-four-hour realized variation must rank in its causal upper 35%",
            "why_low_gross9_overlap_is_plausible": "sparse four-hour adaptive-state crossovers are absent from Gross9 primitives",
        },
        external_basis={
            "origin": "Perry Kaufman canonical Kaufman Adaptive Moving Average convention",
            "fixed_definition": "10-period efficiency ratio, fast length 2, slow length 30; smoothing constant is the squared interpolation between fast and slow constants; recursive KAMA crossover",
            "selection_use": "published 10/2/30 parameters and price-crossover trend interpretation only; no incidence or outcomes",
        },
        features={
            "decision_grid": "every exact four-hour UTC boundary",
            "source_bar": "exact aggregation of 240 coherent BTCUSDT one-minute rows [T-4h,T)",
            "efficiency_ratio": "abs(close-close ten completed bars earlier) divided by sum of ten absolute completed close changes, strict positive denominator",
            "smoothing_constant": "(ER*(2/3-2/31)+2/31)^2",
            "kama": "causal recursive prior KAMA plus smoothing_constant*(completed close-prior KAMA), initialized at the first source-valid completed close with ten-change history and reset across source gaps",
            "crossover": "current completed close-minus-KAMA strict nonzero sign differs from immediately prior finite nonzero sign",
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
            "efficiency_periods": 10,
            "fast_periods": 2,
            "slow_periods": 30,
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
            "names": ["no_variation_gate", "fixed_slow_ema_crossover", "one_bar_stale_crossover", "direction_flip"],
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
            "canonical_kaufman_definition_read": True,
            "repository_kaufman_candidate_found": False,
            "prior_macd_time_fit_choppiness_outcomes_known": True,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "canonical KAMA price crossover under the requested high-variation regime",
        },
        stopping_rule="Terminal first failure; no efficiency, fast/slow period, initialization, crossover, variation, side, hold, clock, subset, control, or other repair.",
    )
    return {**contract, "manifest_hash": canonical_hash(contract)}


def validate(contract: dict[str, Any]) -> None:
    core = {key: value for key, value in contract.items() if key != "manifest_hash"}
    if contract.get("manifest_hash") != canonical_hash(core) or contract != build():
        raise RuntimeError("HVKAMA preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = build()
    validate(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(args.output)
