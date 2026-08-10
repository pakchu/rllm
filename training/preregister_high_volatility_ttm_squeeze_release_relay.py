"""Outcome-blind preregistration for HVTTS-24."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVTTS-24"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_ttm_squeeze_release_relay_preregistration_2026-08-11.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    contract = copy.deepcopy(template.build())
    contract.pop("manifest_hash")
    contract.update(
        protocol_version="high_volatility_ttm_squeeze_release_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-11",
        mechanism={
            "claim": "During elevated realized variation, release of a completed four-hour canonical TTM Squeeze—Bollinger Bands moving from strictly inside Keltner Channels to outside—marks volatility expansion; follow the sign of the frozen twenty-bar detrended regression momentum for twenty-four hours.",
            "side": "long when release-bar regression momentum is strictly positive; short when strictly negative",
            "why_distinct": "Bollinger and Keltner candidates trade their individual price boundaries, variance-concentration candidates use realized-return allocation, and trend-fit candidates use raw price slope. HVTTS trades only the state transition between a 20/2 Bollinger envelope and 20/1.5 ATR Keltner envelope, with direction from a fixed detrended squeeze-momentum regression; it uses no prior candidate threshold repair, flow, OI, funding, fitted outcome, reused event set, or promoted control.",
            "why_suited_to_volatile_regimes": "the signal is a volatility-compression release and completed trailing twenty-four-hour realized variation must rank in its causal upper 35%",
            "why_low_gross9_overlap_is_plausible": "sparse four-hour cross-envelope release transitions are absent from Gross9 primitives",
        },
        external_basis={
            "origin": "John Carter TTM Squeeze convention with the widely used detrended linear-regression momentum formulation",
            "fixed_definition": "20-period population-standard-deviation Bollinger Bands at 2, 20-period EMA Keltner center with Wilder ATR at 1.5, release from strict inside to not-inside, and 20-period detrended regression momentum",
            "selection_use": "published 20/2 and 20/1.5 envelopes plus standard squeeze-release momentum direction only; no incidence or outcomes",
        },
        features={
            "decision_grid": "every exact four-hour UTC boundary",
            "source_bar": "exact aggregation of 240 coherent BTCUSDT one-minute rows [T-4h,T)",
            "bollinger": "20-bar completed close SMA plus/minus 2 population standard deviations",
            "keltner": "20-bar causal EMA close plus/minus 1.5 Wilder ATR(20), reset across source gaps",
            "squeeze_on": "Bollinger upper strictly below Keltner upper and Bollinger lower strictly above Keltner lower",
            "release": "immediately prior valid completed bar squeeze_on and current valid completed bar not squeeze_on",
            "detrended_series": "close minus mean of the 20-bar high-low midpoint and 20-bar close SMA",
            "momentum": "OLS fitted value at newest index 19 over latest 20 consecutive detrended values, strict nonzero",
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
            "envelope_periods": 20,
            "bollinger_std_multiplier": 2.0,
            "keltner_atr_multiplier": 1.5,
            "momentum_periods": 20,
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
            "names": ["no_variation_gate", "close_sma_direction", "one_bar_stale_release", "direction_flip"],
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
            "canonical_ttm_squeeze_definition_read": True,
            "repository_ttm_squeeze_candidate_found": False,
            "prior_bollinger_keltner_variance_concentration_outcomes_known": True,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "canonical TTM Squeeze envelope release under the requested high-variation regime",
        },
        stopping_rule="Terminal first failure; no envelope period, multiplier, ATR convention, release, momentum, variation, side, hold, clock, subset, control, or other repair.",
    )
    return {**contract, "manifest_hash": canonical_hash(contract)}


def validate(contract: dict[str, Any]) -> None:
    core = {key: value for key, value in contract.items() if key != "manifest_hash"}
    if contract.get("manifest_hash") != canonical_hash(core) or contract != build():
        raise RuntimeError("HVTTS preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = build()
    validate(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(args.output)
