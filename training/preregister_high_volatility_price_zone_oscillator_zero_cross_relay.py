"""Outcome-blind preregistration for HVPZO-6."""
from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVPZO-6"
DEFAULT_OUTPUT = Path("results/high_volatility_price_zone_oscillator_zero_cross_relay_preregistration_2026-08-11.json")

def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def build() -> dict[str, Any]:
    core = copy.deepcopy(template.build())
    core.pop("manifest_hash")
    core.update(
        protocol_version="high_volatility_price_zone_oscillator_zero_cross_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-11",
        mechanism={
            "claim": "During elevated realized variation, a completed fifteen-minute canonical Price Zone Oscillator zero crossing identifies a change in the balance of rising versus falling price levels; follow the new sign for six hours.",
            "side": "long on a strict upward PZO zero crossing and short on a strict downward PZO zero crossing",
            "why_distinct": "PZO exponentially smooths signed completed closing-price levels and normalizes by the identically smoothed unsigned closing-price level. Unlike VZO it uses no volume; unlike price moving-average crosses it classifies each completed close by its one-period direction before smoothing; it uses no OI, funding, external daily clock, prior event set, repair, or promoted control.",
            "why_suited_to_volatile_regimes": "the completed trailing twenty-four-hour BTC realized variation must rank in its causal upper 35%, restricting PZO transitions to July-like volatile states",
            "why_low_gross9_overlap_is_plausible": "state-dependent fifteen-minute zero crossings create irregular intraday clocks absent from Gross9 primitives",
        },
        external_basis={
            "origin": "Price Zone Oscillator convention",
            "fixed_definition": "signed close is +close after an up close, -close after a down close, and zero after an unchanged close; PZO=100*EMA(14,signed close)/EMA(14,close); strict zero crossing",
            "selection_use": "published 14-period formula, zero-line interpretation, and direction only; no source incidence or outcomes",
        },
        features={
            "decision_grid": "every exact fifteen-minute UTC boundary",
            "source_bar": "exact aggregation of 15 coherent BTCUSDT one-minute rows [T-15m,T)",
            "signed_close": "+completed close when completed close exceeds the immediately prior contiguous completed close, -completed close when below, zero when equal",
            "ema": "causal alpha=2/(14+1), seeded by the simple average of the first 14 contiguous valid inputs and reset across gaps",
            "pzo": "100 times EMA(14,signed_close) divided by EMA(14,completed_close), requiring finite strict-positive denominator",
            "event": "current finite PZO strictly crosses above zero from <=0 or below zero from >=0 at the immediately prior valid decision",
            "variation": "sqrt(sum squared completed one-minute open-to-close log returns over [T-24h,T)), requiring 1,440 exact coherent finite positive rows",
            "variation_rank": "strict-prior midrank over at most 672 earlier source-valid decisions, minimum 480, current excluded; rank>=0.65",
            "no_imputation": True,
        },
        clock={
            "feature_available": "fifteen-minute boundary after all source rows complete",
            "entry": "exact BTCUSDT open five elapsed minutes later",
            "hold": "six elapsed hours",
            "reservation": "chronological global half-open first-eligible reservation; exit first on equal open",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        policy={
            "ema_periods": 14,
            "variation_hours": 24,
            "variation_history_decisions": 672,
            "minimum_variation_history_decisions": 480,
            "variation_rank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 6,
            "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        diagnostic_controls={
            "names": ["no_variation_gate", "unsmoothed_signed_close_cross", "one_bar_stale_cross", "direction_flip", "forced_long"],
            "cannot_be_promoted": True,
        },
        source_plan={
            "bars": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "high", "low", "close"], "window": ["2023-06-20T00:00:00Z", "2026-08-01T00:00:00Z"], "read_after_preregistration": True},
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        research_boundary={
            "canonical_pzo_definition_read": True,
            "repository_pzo_candidate_found": False,
            "prior_volume_zone_and_price_indicator_outcomes_known": True,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent canonical price-direction zone oscillator plus user-required high-variation regime and irregular intraday event clock",
        },
        stopping_rule="Terminal first failure; no oscillator formula, EMA period, crossing, variation, side, hold, clock, subset, threshold, or control repair.",
    )
    return {**core, "manifest_hash": canonical_hash(core)}

def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVPZO preregistration drift")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build()
    validate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
