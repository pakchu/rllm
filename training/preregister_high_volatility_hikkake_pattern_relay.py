"""Outcome-blind preregistration for HVHIKKAKE-C3-8."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template


POLICY_ID = "HVHIKKAKE-C3-8"
DEFAULT_OUTPUT = Path("results/high_volatility_hikkake_pattern_relay_preregistration_2026-08-11.json")


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
    core = copy.deepcopy(template.build())
    core.pop("manifest_hash")
    core.update(
        protocol_version="high_volatility_hikkake_pattern_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-11",
        mechanism={
            "claim": "During elevated completed realized variation, a confirmed hourly Hikkake identifies a failed displacement from an inside bar. Enter in the official reversal direction after the confirmation close and hold for eight hours.",
            "side": "long only for official bullish confirmation output +2 and short only for official bearish confirmation output -2",
            "why_distinct": "The official Hikkake requires an hourly inside bar, a next-bar displacement in the opposite direction, and a later close through the inside bar's far boundary within three bars. The prior inside-auction candidate instead partitions an eight-hour block into a four-hour range, a two-hour contained auction, and a direct two-hour outside close without a false-displacement or confirmation sequence. Hikkake uses no volume, flow, OI, funding, fitted outcome, reused event set, or promoted control.",
            "why_suited_to_volatile_regimes": "failed breakouts can reverse sharply when completed trailing twenty-four-hour realized variation ranks in its causal upper 35%",
            "why_low_gross9_overlap_is_plausible": "sparse, asynchronously confirmed three-to-six hourly candle-state sequences are absent from Gross9 primitives",
        },
        external_basis={
            "origin": "Daniel L. Chesler Hikkake pattern; QuantConnect LEAN canonical implementation",
            "definition_source": "https://github.com/QuantConnect/Lean/blob/master/Indicators/CandlestickPatterns/Hikkake.cs",
            "documentation_source": "https://www.quantconnect.com/docs/v2/writing-algorithms/indicators/supported-indicators/hikkake",
            "fixed_definition": "bar two is strictly inside bar one; bullish setup bar three has lower high and lower low than bar two, bearish setup bar three has higher high and higher low; confirmation in the next three bars closes above bar-two high for bullish or below bar-two low for bearish; setup outputs +1/-1 and confirmation outputs +2/-2; a new setup on a confirmation bar overwrites the prior confirmation",
            "warmup": "LEAN Period is six hourly bars; pre-ready pattern state may update but all outputs remain zero until Samples >= 6",
            "selection_use": "official pattern inequalities, confirmation horizon, overwrite precedence, output sign, and readiness only; no incidence or outcomes",
        },
        features={
            "decision_grid": "every exact hourly UTC boundary",
            "source_bar": "exact aggregation of 60 coherent BTCUSDT one-minute rows [T-1h,T), open first, high max, low min, close last",
            "inside_bar": "previous hourly high is strictly below the high two bars back and previous hourly low is strictly above the low two bars back",
            "bullish_setup": "current hourly high and low are both strictly below the inside bar's high and low",
            "bearish_setup": "current hourly high and low are both strictly above the inside bar's high and low",
            "confirmation": "within the next three hourly bars, close strictly above the stored inside-bar high for bullish or strictly below the stored inside-bar low for bearish",
            "overwrite_precedence": "if a new setup and an old confirmation occur on the same bar, emit only the new setup and replace stored state exactly as LEAN",
            "eligible_pattern": "only ready confirmation outputs +2 or -2; unconfirmed setup outputs +1/-1 and zero outputs are ineligible",
            "variation": "sqrt(sum squared completed five-minute open-to-close log returns over [T-24h,T)), finite strict positive",
            "variation_rank": "strict-prior midrank over at most 2160 earlier source-valid hourly decisions, minimum 720, current excluded; rank>=0.65",
            "recursive_reset": "hour aggregation, Hikkake state, readiness, and variation history reset after any invalid completed source hour; no imputation",
            "no_imputation": True,
        },
        clock={
            "feature_available": "hour boundary after the completed confirmation bar",
            "entry": "exact BTCUSDT open five elapsed minutes later",
            "hold": "8 elapsed hours",
            "reservation": "global half-open first-eligible reservation; exit first on equal open",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        policy={
            "indicator_period_hours": 6,
            "confirmation_bars": 3,
            "confirmation_absolute_output": 2,
            "variation_hours": 24,
            "variation_history_decisions": 2160,
            "minimum_variation_history_decisions": 720,
            "variation_rank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 8,
            "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        diagnostic_controls={
            "names": [
                "no_variation_gate",
                "initial_setup_only",
                "all_nonzero_hikkake_outputs",
                "one_hour_stale_confirmation",
                "direction_flip",
                "forced_long",
            ],
            "cannot_be_promoted": True,
        },
        source_plan={
            "bars": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close"],
                "window": ["2023-04-01T00:00:00Z", "2026-08-01T00:00:00Z"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        research_boundary={
            "canonical_hikkake_definition_read": True,
            "repository_hikkake_candidate_found": False,
            "related_inside_auction_candidate_known": True,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "official LEAN confirmation-only Hikkake reversal under the requested high-variation regime",
        },
        stopping_rule="Terminal first failure; no timeframe, inequalities, readiness, confirmation horizon, output class, variation, side, hold, clock, subset, threshold, control, or other repair.",
    )
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVHIKKAKE-C3-8 preregistration drift")


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
