"""Outcome-blind preregistration for HVSTC-24."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVSTC-24"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_schaff_trend_cycle_reentry_relay_preregistration_2026-08-11.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    contract = copy.deepcopy(template.build())
    contract.pop("manifest_hash")
    contract.update(
        protocol_version="high_volatility_schaff_trend_cycle_reentry_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-11",
        mechanism={
            "claim": "During elevated realized variation, a completed four-hour canonical Schaff Trend Cycle reentry from below 25 or above 75 identifies an early cyclical trend turn after double stochastic normalization of MACD; follow the reentry direction for twenty-four hours.",
            "side": "long when STC crosses strictly above 25; short when STC crosses strictly below 75",
            "why_distinct": "MACD trades a raw fixed-EMA difference crossover and stochastic trades price range location. HVSTC transforms MACD(23,50) through two causal 10-period stochastic normalizations with fixed 0.5 recursive smoothing before threshold reentry; it uses no prior candidate threshold repair, volume, flow, OI, funding, fitted outcome, reused event set, or promoted control.",
            "why_suited_to_volatile_regimes": "completed trailing twenty-four-hour realized variation must rank in its causal upper 35%",
            "why_low_gross9_overlap_is_plausible": "sparse four-hour double-normalized trend-cycle reentries are absent from Gross9 primitives",
        },
        external_basis={
            "origin": "Doug Schaff canonical Schaff Trend Cycle convention",
            "fixed_definition": "EMA fast 23 minus EMA slow 50, two successive 10-period stochastic normalizations, each recursively smoothed with factor 0.5; long cross above 25 and short cross below 75",
            "selection_use": "published 23/50/10 settings, 0.5 smoothing, and 25/75 reentry interpretation only; no incidence or outcomes",
        },
        features={
            "decision_grid": "every exact four-hour UTC boundary",
            "source_bar": "exact aggregation of 240 coherent BTCUSDT one-minute rows [T-4h,T)",
            "macd": "causal EMA(23) close minus causal EMA(50) close, each reset across source gaps and emitted after its full warmup",
            "first_stochastic": "100*(MACD-min latest 10 MACD)/(max-min), strict positive range, followed by recursive factor-0.5 smoothing",
            "second_stochastic": "100*(first smoothed value-min latest 10)/(max-min), strict positive range, followed by recursive factor-0.5 smoothing to STC",
            "event": "current STC strictly crosses above 25 from <=25 or strictly below 75 from >=75",
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
            "fast_ema_periods": 23,
            "slow_ema_periods": 50,
            "cycle_periods": 10,
            "smoothing_factor": 0.5,
            "lower_reentry": 25.0,
            "upper_reentry": 75.0,
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
            "names": ["no_variation_gate", "raw_macd_zero_cross", "one_bar_stale_reentry", "direction_flip"],
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
            "canonical_schaff_definition_read": True,
            "repository_schaff_candidate_found": False,
            "prior_macd_stochastic_fisher_outcomes_known": True,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "canonical Schaff Trend Cycle 23/50/10 reentry under the requested high-variation regime",
        },
        stopping_rule="Terminal first failure; no EMA period, cycle period, smoothing, threshold, normalization, variation, side, hold, clock, subset, control, or other repair.",
    )
    return {**contract, "manifest_hash": canonical_hash(contract)}


def validate(contract: dict[str, Any]) -> None:
    core = {key: value for key, value in contract.items() if key != "manifest_hash"}
    if contract.get("manifest_hash") != canonical_hash(core) or contract != build():
        raise RuntimeError("HVSTC preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = build()
    validate(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(args.output)
