"""Outcome-blind preregistration for HVCRSI-24."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVCRSI-24"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_connors_rsi_extreme_reversal_preregistration_2026-08-11.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    contract = copy.deepcopy(template.build())
    contract.pop("manifest_hash")
    contract.update(
        protocol_version="high_volatility_connors_rsi_extreme_reversal_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-11",
        mechanism={
            "claim": "During elevated realized variation, a completed four-hour canonical Connors RSI(3,2,100) entering an extreme identifies jointly stretched price momentum, directional streak, and one-bar return rank; fade the extreme for twenty-four hours.",
            "side": "long when Connors RSI enters strictly below 10 from at least 10; short when it enters strictly above 90 from at most 90",
            "why_distinct": "Wilder RSI uses one smoothed price-change ratio, stochastic uses range location, and DeMarker uses high/low comparisons. HVCRSI equally combines 3-period price RSI, 2-period RSI of signed consecutive-close streak length, and the current one-bar return percentile against 100 strictly prior returns; it uses no prior candidate threshold repair, volume, flow, OI, funding, fitted outcome, reused event set, or promoted control.",
            "why_suited_to_volatile_regimes": "completed trailing twenty-four-hour realized variation must rank in its causal upper 35%",
            "why_low_gross9_overlap_is_plausible": "sparse four-hour three-component exhaustion entries are absent from Gross9 primitives",
        },
        external_basis={
            "origin": "Larry Connors canonical Connors RSI convention",
            "fixed_definition": "mean of RSI(close,3), RSI(signed consecutive-close streak,2), and 100-period percent rank of current one-bar return; fade entries below 10 and above 90",
            "selection_use": "published 3/2/100 components and 10/90 extreme-reversal interpretation only; no incidence or outcomes",
        },
        features={
            "decision_grid": "every exact four-hour UTC boundary",
            "source_bar": "exact aggregation of 240 coherent BTCUSDT one-minute rows [T-4h,T)",
            "price_rsi": "Wilder RSI(3) of completed close changes, initialized by mean gains/losses and reset across source gaps",
            "streak": "positive count of consecutive strictly rising completed closes, negative count of consecutive strictly falling closes, zero on unchanged close or reset",
            "streak_rsi": "Wilder RSI(2) of completed streak changes, reset across source gaps",
            "percent_rank": "100 times count of strictly prior 100 valid one-bar completed close returns less than current return divided by 100; current excluded from reference",
            "connors_rsi": "arithmetic mean of finite price RSI, streak RSI, and percent rank",
            "event": "strict entry below 10 from prior >=10 or above 90 from prior <=90",
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
            "price_rsi_periods": 3,
            "streak_rsi_periods": 2,
            "return_rank_periods": 100,
            "lower_extreme": 10.0,
            "upper_extreme": 90.0,
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
            "names": ["no_variation_gate", "price_rsi_only_extreme", "one_bar_stale_entry", "direction_flip"],
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
            "canonical_connors_rsi_definition_read": True,
            "repository_connors_rsi_candidate_found": False,
            "prior_wilder_rsi_stochastic_demarker_outcomes_known": True,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "canonical Connors RSI 3/2/100 extreme entry under the requested high-variation regime",
        },
        stopping_rule="Terminal first failure; no RSI periods, streak rule, rank period, extreme, variation, side, hold, clock, subset, control, or other repair.",
    )
    return {**contract, "manifest_hash": canonical_hash(contract)}


def validate(contract: dict[str, Any]) -> None:
    core = {key: value for key, value in contract.items() if key != "manifest_hash"}
    if contract.get("manifest_hash") != canonical_hash(core) or contract != build():
        raise RuntimeError("HVCRSI preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = build()
    validate(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(args.output)
