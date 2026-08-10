"""Outcome-blind preregistration for HVHA-24."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVHA-24"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_heikin_ashi_color_flip_relay_preregistration_2026-08-11.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    contract = copy.deepcopy(template.build())
    contract.pop("manifest_hash")
    contract.update(
        protocol_version="high_volatility_heikin_ashi_color_flip_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-11",
        mechanism={
            "claim": "During elevated realized variation, a strict color flip in completed four-hour causal Heikin-Ashi BTC candles marks a smoothed auction-direction transition; follow the new synthetic-candle color for twenty-four hours.",
            "side": "long when Heikin-Ashi close-minus-open flips strictly from negative to positive; short on the strict reverse flip",
            "why_distinct": "Ordinary candle candidates use raw body/wick morphology, moving-average candidates smooth closes, and Supertrend carries ATR bands. HVHA recursively smooths the prior synthetic open and completed OHLC mean into a transformed auction candle before testing color; it uses no prior candidate parameter repair, volume, flow, OI, funding, fitted outcome, reused event set, or promoted control.",
            "why_suited_to_volatile_regimes": "completed trailing twenty-four-hour realized variation must rank in its causal upper 35%",
            "why_low_gross9_overlap_is_plausible": "sparse four-hour recursive synthetic-candle color flips are absent from Gross9 primitives",
        },
        external_basis={
            "origin": "canonical Heikin-Ashi candle construction",
            "fixed_definition": "HA close=(open+high+low+close)/4; HA open=(prior HA open+prior HA close)/2, initialized as (open+close)/2 after a source gap; trade strict HA body color flip",
            "selection_use": "published recursive construction and color trend interpretation only; no incidence or outcomes",
        },
        features={
            "decision_grid": "every exact four-hour UTC boundary",
            "source_bar": "exact aggregation of 240 coherent BTCUSDT one-minute rows [T-4h,T)",
            "ha_close": "mean of completed raw open, high, low, close",
            "ha_open": "mean of immediately prior causal HA open and HA close; initialize at mean of raw open and close after each source gap",
            "color": "strict sign of HA close-minus-HA open; zero bodies are invalid",
            "event": "current finite nonzero color differs from immediately prior finite nonzero color",
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
            "initial_open_rule": "raw_open_close_midpoint",
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
            "names": ["no_variation_gate", "raw_candle_color_flip", "one_bar_stale_flip", "direction_flip"],
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
            "canonical_heikin_ashi_definition_read": True,
            "repository_heikin_ashi_candidate_found": False,
            "prior_raw_candle_moving_average_supertrend_outcomes_known": True,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "canonical Heikin-Ashi color flip under the requested high-variation regime",
        },
        stopping_rule="Terminal first failure; no HA initialization, recursion, color, variation, side, hold, clock, subset, control, or other repair.",
    )
    return {**contract, "manifest_hash": canonical_hash(contract)}


def validate(contract: dict[str, Any]) -> None:
    core = {key: value for key, value in contract.items() if key != "manifest_hash"}
    if contract.get("manifest_hash") != canonical_hash(core) or contract != build():
        raise RuntimeError("HVHA preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = build()
    validate(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(args.output)
