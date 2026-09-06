"""Outcome-blind preregistration for HVCE-24."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVCE-24"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_chandelier_exit_state_flip_relay_preregistration_2026-08-11.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    contract = copy.deepcopy(template.build())
    contract.pop("manifest_hash")
    contract.update(
        protocol_version="high_volatility_chandelier_exit_state_flip_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-11",
        mechanism={
            "claim": "During elevated realized variation, a completed four-hour BTC flip through the active canonical Chandelier Exit stop marks a volatility-scaled trend regime change anchored to the latest twenty-two-bar extreme; follow the new state for twenty-four hours.",
            "side": "long when a short state closes strictly above its trailing short stop; short when a long state closes strictly below its trailing long stop",
            "why_distinct": "Supertrend anchors symmetric bands to current hl2, Donchian trades raw boundary breakouts, and Parabolic SAR accelerates from successive extreme points. HVCE anchors one-sided stops to fixed twenty-two-bar extremes minus/plus three Wilder ATR and trails only the active stop; it uses no prior candidate parameter repair, volume, flow, OI, funding, fitted outcome, reused event set, or promoted control.",
            "why_suited_to_volatile_regimes": "completed trailing twenty-four-hour realized variation must rank in its causal upper 35%",
            "why_low_gross9_overlap_is_plausible": "sparse four-hour extreme-anchored ATR trailing-state flips are absent from Gross9 primitives",
        },
        external_basis={
            "origin": "Chuck LeBeau canonical Chandelier Exit convention",
            "fixed_definition": "22-period Wilder ATR and 22-period highest-high/lowest-low with 3 ATR offset; active long stop trails upward and active short stop trails downward; completed close breach flips state",
            "selection_use": "published 22/3 parameters and trailing-stop trend direction only; no incidence or outcomes",
        },
        features={
            "decision_grid": "every exact four-hour UTC boundary",
            "source_bar": "exact aggregation of 240 coherent BTCUSDT one-minute rows [T-4h,T)",
            "true_range": "max(high-low, abs(high-prior completed close), abs(low-prior completed close))",
            "atr": "Wilder recursive 22-period ATR initialized by the first mean of 22 consecutive valid true ranges and reset across source gaps",
            "raw_stops": "long=latest 22-bar highest high-3*ATR; short=latest 22-bar lowest low+3*ATR",
            "trailing_state": "initialize short at first finite stops; while short carry min(prior short stop,current raw short) and flip long only on close>stop; while long carry max(prior long stop,current raw long) and flip short only on close<stop; reset the newly active stop to its current raw value",
            "event": "current state differs from immediately prior finite state",
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
            "lookback_periods": 22,
            "atr_periods": 22,
            "atr_multiplier": 3.0,
            "initial_state": -1,
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
            "names": ["no_variation_gate", "untrailed_raw_stop_flip", "one_bar_stale_flip", "direction_flip"],
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
            "canonical_chandelier_definition_read": True,
            "repository_chandelier_candidate_found": False,
            "prior_supertrend_donchian_psar_outcomes_known": True,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "canonical Chandelier Exit 22/3 state flip under the requested high-variation regime",
        },
        stopping_rule="Terminal first failure; no lookback, ATR period, multiplier, initialization, trailing recursion, state, variation, side, hold, clock, subset, control, or other repair.",
    )
    return {**contract, "manifest_hash": canonical_hash(contract)}


def validate(contract: dict[str, Any]) -> None:
    core = {key: value for key, value in contract.items() if key != "manifest_hash"}
    if contract.get("manifest_hash") != canonical_hash(core) or contract != build():
        raise RuntimeError("HVCE preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = build()
    validate(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(args.output)
