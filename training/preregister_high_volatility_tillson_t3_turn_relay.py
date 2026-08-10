"""Outcome-blind preregistration for HVT3-24."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVT3-24"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_tillson_t3_turn_relay_preregistration_2026-08-11.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    contract = copy.deepcopy(template.build())
    contract.pop("manifest_hash")
    contract.update(
        protocol_version="high_volatility_tillson_t3_turn_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-11",
        mechanism={
            "claim": "During elevated realized variation, a completed four-hour canonical Tillson T3 slope turn identifies a smoothed trend reversal with less lag than a conventional multi-pass EMA; follow the new slope direction for twenty-four hours.",
            "side": "long when the T3 first difference crosses strictly above zero; short when it crosses strictly below zero",
            "why_distinct": "T3 combines six causal EMAs with Tim Tillson's fixed volume-factor coefficients. The event is a sign change in the completed T3 slope, not a price/EMA crossover, oscillator threshold, prior candidate repair, volume, flow, OI, funding, fitted outcome, reused event set, or promoted control.",
            "why_suited_to_volatile_regimes": "completed trailing twenty-four-hour realized variation must rank in its causal upper 35%",
            "why_low_gross9_overlap_is_plausible": "sparse four-hour six-stage generalized-DEMA slope turns are absent from Gross9 primitives",
        },
        external_basis={
            "origin": "Tim Tillson canonical T3 moving-average convention",
            "fixed_definition": "six recursive EMAs of period 5 and volume factor 0.7, combined as T3=c1*E6+c2*E5+c3*E4+c4*E3 with the published coefficient identities",
            "selection_use": "common canonical period 5 and volume factor 0.7 plus slope-turn interpretation only; no incidence or outcomes",
        },
        features={
            "decision_grid": "every exact four-hour UTC boundary",
            "source_bar": "exact aggregation of 240 coherent BTCUSDT one-minute rows [T-4h,T)",
            "ema_chain": "six consecutive causal EMA(5) transforms of completed four-hour closes, reset across source gaps and each emitted only after its own full five-observation warmup",
            "t3": "c1*E6+c2*E5+c3*E4+c4*E3 where c1=-a^3, c2=3a^2+3a^3, c3=-6a^2-3a-3a^3, c4=1+3a+a^3+3a^2, a=0.7",
            "event": "current completed T3 first difference strictly crosses above zero from <=0 or below zero from >=0",
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
            "ema_periods": 5,
            "volume_factor": 0.7,
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
            "names": ["no_variation_gate", "price_ema_cross", "one_bar_stale_turn", "direction_flip"],
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
            "canonical_t3_definition_read": True,
            "repository_t3_candidate_found": False,
            "prior_candidate_outcomes_known": True,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "canonical Tillson T3(5,0.7) slope turn under the requested high-variation regime",
        },
        stopping_rule="Terminal first failure; no EMA period, volume factor, slope definition, variation, side, hold, clock, subset, control, or other repair.",
    )
    return {**contract, "manifest_hash": canonical_hash(contract)}


def validate(contract: dict[str, Any]) -> None:
    core = {key: value for key, value in contract.items() if key != "manifest_hash"}
    if contract.get("manifest_hash") != canonical_hash(core) or contract != build():
        raise RuntimeError("HVT3 preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = build()
    validate(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(args.output)
