"""Outcome-blind preregistration for HVAPS-F2-24."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVAPS-F2-24"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_augen_price_spike_fade_relay_preregistration_2026-08-11.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    candidate = copy.deepcopy(template.build())
    candidate.pop("manifest_hash")
    candidate.update(
        protocol_version="high_volatility_augen_price_spike_fade_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-11",
        mechanism={
            "claim": (
                "During elevated realized variation, a completed four-hour BTC price change "
                "that newly exceeds two units of the canonical Augen Price Spike is an "
                "overextension relative to immediately preceding return volatility; fade the "
                "signed shock for twenty-four hours."
            ),
            "side": (
                "short when the signed Augen Price Spike newly becomes strictly greater than "
                "+2; long when it newly becomes strictly less than -2"
            ),
            "why_distinct": (
                "Augen Price Spike divides the current arithmetic close change by the previous "
                "close times a three-observation population standard deviation of lagged log "
                "returns. It is not a close-level z-score, daily robust outlier, range or wick "
                "pattern, ATR channel, momentum oscillator crossover, volume, flow, OI, funding, "
                "fitted outcome, reused event set, or promoted control."
            ),
            "why_suited_to_volatile_regimes": (
                "the trigger is itself a volatility-normalized signed price shock and the "
                "completed trailing twenty-four-hour realized variation must rank in its causal "
                "upper 35%"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "sparse four-hour first crossings of a three-return local-volatility shock "
                "threshold are absent from Gross9 primitives"
            ),
        },
        external_basis={
            "origin": (
                "Jeff Augen, The Volatility Edge in Options Trading; QuantConnect LEAN "
                "canonical implementation"
            ),
            "definition_source": (
                "https://github.com/QuantConnect/Lean/blob/master/Indicators/AugenPriceSpike.cs"
            ),
            "documentation_source": (
                "https://www.quantconnect.com/docs/v2/writing-algorithms/indicators/"
                "supported-indicators/augen-price-spike"
            ),
            "threshold_source": "Jeff Augen, The Volatility Edge in Options Trading, two-standard-deviation price-spike convention",
            "fixed_definition": (
                "period=3 (LEAN default); for completed bar t, lagged_log_return_t="
                "log(close[t-1]/close[t-2]); population standard deviation over the latest three "
                "lagged_log_return observations; spike_t=(close[t]-close[t-1])/(std_t*close[t-1]); "
                "warmup five completed bars; first exterior crossing of +/-2"
            ),
            "selection_use": (
                "published formula, LEAN default period, signed two-standard-deviation shock, "
                "and volatility-overextension fade interpretation only; no incidence or outcomes"
            ),
        },
        features={
            "decision_grid": "every exact four-hour UTC boundary",
            "source_bar": (
                "exact aggregation of 240 coherent BTCUSDT one-minute rows [T-4h,T)"
            ),
            "lagged_log_return": "log(close[t-1]/close[t-2]) on uninterrupted valid bars",
            "local_volatility": (
                "population standard deviation (ddof=0) of the latest three lagged log returns, "
                "including the value formed from close[t-1] and close[t-2] but excluding the "
                "current close change"
            ),
            "augen_price_spike": (
                "(close[t]-close[t-1]) divided by local_volatility[t]*close[t-1], finite with "
                "strictly positive denominator"
            ),
            "event": (
                "spike newly strictly above +2 from not above +2 gives side -1; spike newly "
                "strictly below -2 from not below -2 gives side +1"
            ),
            "variation": (
                "sqrt(sum squared completed five-minute open-to-close log returns over [T-24h,T)), "
                "finite strict positive"
            ),
            "variation_rank": (
                "strict-prior midrank over at most 180 earlier valid decisions, minimum 120, "
                "current excluded; rank>=0.65"
            ),
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
            "spike_periods": 3,
            "spike_threshold": 2.0,
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
            "names": [
                "no_variation_gate",
                "one_bar_stale_cross",
                "direction_follow",
                "zero_cross",
            ],
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
            "canonical_augen_price_spike_definition_read": True,
            "repository_augen_price_spike_candidate_found": False,
            "prior_outlier_and_shock_candidate_outcomes_known": True,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": (
                "canonical Augen Price Spike default period and published two-standard-deviation "
                "overextension under the requested high-variation regime"
            ),
        },
        stopping_rule=(
            "Terminal first failure; no period, standard-deviation convention, threshold, "
            "crossing, variation, side, hold, clock, subset, control, or other repair."
        ),
    )
    return {**candidate, "manifest_hash": canonical_hash(candidate)}


def validate(candidate: dict[str, Any]) -> None:
    core = {key: value for key, value in candidate.items() if key != "manifest_hash"}
    if candidate.get("manifest_hash") != canonical_hash(core) or candidate != build():
        raise RuntimeError("HVAPS-F2-24 preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    registration = build()
    validate(registration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(registration, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(args.output)
