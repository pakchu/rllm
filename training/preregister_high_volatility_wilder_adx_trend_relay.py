"""Outcome-blind preregistration for HVWADX-24."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVWADX-24"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_wilder_adx_trend_relay_preregistration_2026-08-11.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    registration = copy.deepcopy(template.build())
    registration.pop("manifest_hash")
    registration.update(
        protocol_version="high_volatility_wilder_adx_trend_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-11",
        mechanism={
            "claim": (
                "A completed four-hour BTC auction with canonical fourteen-period Wilder "
                "ADX at or above 25 identifies directional movement strong enough to persist "
                "for the next twenty-four hours when realized variation is elevated."
            ),
            "side": "long when +DI exceeds -DI; short when -DI exceeds +DI",
            "why_distinct": (
                "HVWRSI is a daily 30/70 oscillator reversal and is terminal. Weekly momentum "
                "uses endpoint return magnitude. HVWADX uses Wilder smoothed true range and "
                "mutually exclusive directional movement to require trend strength before "
                "taking the DI direction, with no return-tail threshold, OI, flow, volume, "
                "funding, fitted outcome, prior event, or promoted control."
            ),
            "why_suited_to_volatile_regimes": (
                "the completed trailing twenty-four-hour realized variation must rank in its "
                "causal upper 35%"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "four-hour Wilder directional-movement trend-strength clocks are absent from "
                "Gross9 primitives"
            ),
        },
        external_basis={
            "origin": "J. Welles Wilder Jr., New Concepts in Technical Trading Systems (1978)",
            "fixed_definition": (
                "fourteen periods; Wilder recursive true range, +DM, -DM, DX, and ADX; "
                "conventional ADX 25 trend-strength level"
            ),
            "selection_use": "canonical published formula, period, and level only; no candidate incidence or stage outcomes",
        },
        features={
            "decision_grid": "every exact four-hour UTC boundary",
            "source_bar": "exact aggregation of 240 coherent BTCUSDT one-minute rows [T-4h,T)",
            "true_range": "max(high-low, abs(high-prior_close), abs(low-prior_close))",
            "plus_dm": "high-prior_high when strictly greater than prior_low-low and positive, else zero",
            "minus_dm": "prior_low-low when strictly greater than high-prior_high and positive, else zero",
            "wilder_seed": "arithmetic sums of TR, +DM, and -DM over the first fourteen exact four-hour transitions",
            "wilder_update": "smoothed_sum=prior_smoothed_sum-prior_smoothed_sum/14+current_component, recursively without reset",
            "directional_indices": "+DI=100*smoothed_plus_dm/smoothed_tr; -DI analogously",
            "dx": "100*abs(+DI-minus_DI)/(+DI+minus_DI), invalid when denominator is zero",
            "adx": "arithmetic mean of first fourteen finite DX values, then Wilder recursive average without reset",
            "variation": "sqrt(sum squared completed five-minute open-to-close log returns over [T-24h,T)), finite strict positive",
            "variation_rank": "strict-prior midrank over at most 180 earlier valid decisions, minimum 120, current excluded; rank>=0.65",
            "eligibility": "ADX>=25, unequal finite +DI/-DI, and variation gate",
            "no_imputation": True,
        },
        clock={
            "feature_available": "four-hour UTC boundary after the completed source bar",
            "entry": "exact BTCUSDT open five elapsed minutes later",
            "hold": "24 elapsed hours",
            "reservation": "global half-open first-eligible reservation; exit first on equal open",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        policy={
            "adx_periods": 14,
            "adx_min": 25.0,
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
            "names": ["no_adx_gate", "no_variation_gate", "one_bar_stale_state", "direction_flip"],
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
            "canonical_wilder_definition_read": True,
            "repository_plain_wilder_adx_candidate_found": False,
            "prior_wilder_rsi_outcomes_known": True,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "canonical Wilder directional trend-strength continuation in the requested high-variation regime",
        },
        stopping_rule=(
            "Terminal first failure; no ADX period, formula, 25 level, variation, side, hold, "
            "clock, subset, threshold, or control repair."
        ),
    )
    return {**registration, "manifest_hash": canonical_hash(registration)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVWADX preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build()
    validate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(args.output)
