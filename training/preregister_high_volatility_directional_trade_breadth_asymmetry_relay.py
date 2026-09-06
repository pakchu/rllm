"""Outcome-blind preregistration for HVDTBA-6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVDTBA-6"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_directional_trade_breadth_asymmetry_relay_preregistration_2026-08-10.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_directional_trade_breadth_asymmetry_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-10",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "During volatile BTC trading, the direction that attracts most reported executions "
                "across completed one-minute price changes represents broad transaction participation "
                "rather than notional concentration or a few large orders. At a new extreme in this "
                "directional execution-count imbalance, follow the execution-breadth direction."
            ),
            "side": "strict sign of up-minute minus down-minute reported execution count",
            "why_distinct": (
                "HVDTBA conditions each reported execution count on the sign of its completed minute "
                "and compares directional count breadth. Existing candidates use aggregate count, "
                "average ticket, temporal count concentration/backloading, taker flow, quote volume, "
                "or cross-asset count levels; none uses within-BTC directional execution-count allocation."
            ),
            "volatile_market_target": "strict-prior prior-24h realized-variation rank >=0.65",
            "why_low_gross9_overlap_is_plausible": (
                "hourly false-to-true onsets of a directional execution-breadth tail are data-driven "
                "rather than fixed funding, daily, release, or session clocks"
            ),
        },
        "features": {
            "decision_grid": "every exact UTC hour D",
            "breadth_window": "360 exact BTCUSDT bars_binance interval=1m rows [D-6h,D)",
            "minute_direction": "strict sign of log(close/open); zero-return minutes excluded",
            "minute_execution_count": "finite nonnegative integer number_of_trades",
            "direction_support": "at least 60 positive-return and 60 negative-return minutes",
            "up_count": "sum number_of_trades on positive-return minutes",
            "down_count": "sum number_of_trades on negative-return minutes",
            "directional_trade_breadth": "(up_count-down_count)/(up_count+down_count), denominator positive and result strict nonzero",
            "absolute_breadth_rank": (
                "strict-prior midrank of abs(directional_trade_breadth) over at most 2,160 valid "
                "hourly decisions, minimum 1,440, current excluded; rank>=0.85"
            ),
            "btc_variation": (
                "sum squared log(close/open) returns over exact 1,440 completed BTCUSDT one-minute "
                "rows [D-24h,D), strict positive"
            ),
            "variation_rank": "same strict-prior 2,160/1,440 valid-decision rule; rank>=0.65",
            "eligible_state": "both rank gates pass and directional_trade_breadth is strict nonzero",
            "onset": "eligible now and immediately preceding exact source-valid hour ineligible",
            "source_valid": (
                "exact unique minute grids, finite positive coherent OHLC, integer nonnegative "
                "number_of_trades, required direction support, positive count denominator; no imputation"
            ),
        },
        "clock": {
            "decision": "D after every required completed source minute",
            "entry": "exact BTCUSDT perpetual D+5m open",
            "side": "sign directional_trade_breadth",
            "hold": "6 elapsed hours",
            "reservation": "global half-open; exit first on equal-time entry",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not an input; exact realized funding only after novelty passes",
        },
        "policy": {
            "breadth_hours": 6,
            "variation_hours": 24,
            "minimum_minutes_each_direction": 60,
            "history_hours": 2160,
            "minimum_history_hours": 1440,
            "absolute_breadth_rank_min": 0.85,
            "variation_rank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 6,
            "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        "stages": {
            "train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
        },
        "source_support_gates": {
            "minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8},
            "minority_side_share_min": 0.20,
            "max_month_share": 0.45,
        },
        "novelty_gates": {
            "exact_entry_jaccard_max": 0.10,
            "candidate_near_6h_share_max": 0.35,
            "occupied_5m_bar_jaccard_max": 0.25,
            "absolute_signed_exposure_pearson_max": 0.35,
            "must_pass_before_economics": True,
        },
        "economic_gates": {
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_min": 3.0,
            "strict_mdd_max_pct": 15.0,
            "mean_gross_underlying_min_bp": 20.0,
            "weekly_signflip_one_sided_p_max": 0.10,
            "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5,
            "each_calendar_half_positive": True,
            "stop_on_first_failure": True,
            "accounting": (
                "fixed quantity, exact funding, 6bp base and 10bp stress per notional side, "
                "every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "post_stage_volatility_audit": {
            "prerequisite": "unchanged candidate passes train, test, eval, and final",
            "rv20_q90_entry_filter": False,
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "identical_clock_forced_long_residual_positive": True,
        },
        "diagnostic_controls": {
            "names": [
                "no_variation_gate", "no_breadth_magnitude_gate", "quote_volume_directional_breadth",
                "one_hour_stale_features", "direction_flip", "forced_long",
            ],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "bars": {
                "table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close", "number_of_trades", "quote_asset_volume"],
                "window": ["2023-04-01T00:00:00Z", "2026-08-01T00:00:00Z"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "repository_directional_execution_count_candidate_found": False,
            "prior_aggregate_count_ticket_and_arrival_concentration_outcomes_known": True,
            "prior_event_sets_reused": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent within-BTC directional execution-count breadth mechanism",
        },
        "stopping_rule": (
            "terminal first failure; no count definition, window, direction support, rank, history, "
            "onset, side, hold, clock, subset, comparator, control, or gate repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVDTBA preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)
