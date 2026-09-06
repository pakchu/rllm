"""Outcome-blind preregistration for HVMIB-24."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVMIB-24"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_mass_index_bulge_reversal_preregistration_2026-08-11.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    contract = copy.deepcopy(template.build())
    contract.pop("manifest_hash")
    contract.update(
        protocol_version="high_volatility_mass_index_bulge_reversal_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-11",
        mechanism={
            "claim": "During elevated realized variation, completion of the canonical Mass Index reversal bulge on completed four-hour BTC auctions marks exhaustion of range expansion; trade opposite the sign of the completed nine-bar displacement for twenty-four hours.",
            "side": "short when the completed close is strictly above its close nine bars earlier; long when strictly below",
            "why_distinct": "HVCIR uses path efficiency, breakout candidates use price boundaries, and ATR-style candidates use range magnitude. HVMIB uses the fixed sum of a single-EMA/double-EMA high-low range ratio and only emits after its canonical 27-above then 26.5-below bulge completion; it uses no CHOP direction flip, OI, flow, funding, fitted outcome, reused event set, or promoted control.",
            "why_suited_to_volatile_regimes": "completed trailing twenty-four-hour realized variation must rank in its causal upper 35%",
            "why_low_gross9_overlap_is_plausible": "sparse four-hour range-EMA bulge completions are absent from Gross9 primitives",
        },
        external_basis={
            "origin": "Donald Dorsey canonical Mass Index reversal-bulge convention",
            "fixed_definition": "single 9-period EMA of high-low divided by its 9-period EMA, summed over 25 completed periods; arm above 27 and fire on later strict crossing below 26.5",
            "selection_use": "published 9/25 parameters, 27/26.5 bulge levels, and trend-reversal interpretation only; no incidence or outcomes",
        },
        features={
            "decision_grid": "every exact four-hour UTC boundary",
            "source_bar": "exact aggregation of 240 coherent BTCUSDT one-minute rows [T-4h,T)",
            "range": "completed-bar high minus low, strict positive",
            "ema1": "causal adjusted-false 9-period EMA of completed high-low range, minimum 9 valid consecutive bars",
            "ema2": "causal adjusted-false 9-period EMA of ema1, minimum 9 additional valid consecutive bars",
            "mass_index": "rolling sum over 25 completed bars of ema1/ema2, all components finite strict positive",
            "bulge_state": "arm only when Mass Index is strictly above 27; remain armed until the first later completed bar strictly below 26.5, which emits and clears the arm",
            "direction": "negative sign of current completed close minus completed close nine bars earlier, strict nonzero",
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
            "ema_periods": 9,
            "mass_sum_periods": 25,
            "bulge_arm_level": 27.0,
            "bulge_release_level": 26.5,
            "direction_lag_bars": 9,
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
            "names": ["no_variation_gate", "one_bar_stale_bulge", "direction_flip", "same_direction"],
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
            "canonical_mass_index_definition_read": True,
            "repository_mass_index_candidate_found": False,
            "prior_range_breakout_choppiness_outcomes_known": True,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "canonical Mass Index reversal bulge under the requested high-variation regime",
        },
        stopping_rule="Terminal first failure; no EMA period, sum period, bulge levels, state reset, direction, variation, side, hold, clock, subset, control, or other repair.",
    )
    return {**contract, "manifest_hash": canonical_hash(contract)}


def validate(contract: dict[str, Any]) -> None:
    core = {key: value for key, value in contract.items() if key != "manifest_hash"}
    if contract.get("manifest_hash") != canonical_hash(core) or contract != build():
        raise RuntimeError("HVMIB preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = build()
    validate(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(args.output)
