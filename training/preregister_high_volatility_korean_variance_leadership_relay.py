"""Outcome-blind preregistration for HVKVLR-6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVKVLR-6"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_korean_variance_leadership_relay_preregistration_2026-08-10.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_korean_variance_leadership_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-10",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "When KRW-BTC cash realizes much more path variation than the Binance BTCUSDT "
                "perpetual over the same completed six hours, Korean cash is leading regional price "
                "discovery rather than merely matching global volatility. If both venues agree on "
                "the final-hour direction, follow that direction while the variance impulse transmits."
            ),
            "side": "common strict sign of completed final-hour Upbit and Binance returns",
            "why_distinct": (
                "HVKVLR compares aligned regional-cash and perpetual quadratic variation. KCLR used "
                "only eight-hour return magnitude leadership; KPAR used FX-adjusted premium change; "
                "KCVSR used Upbit volume. Spot/perpetual variance candidates did not use Korean cash."
            ),
            "volatile_market_target": "Binance completed six-hour variation strict-prior rank >=0.65",
            "why_low_gross9_overlap_is_plausible": (
                "false-to-true hourly regional variance-leadership onsets are absent from Gross9 and "
                "are not tied to one fixed Asia-session boundary"
            ),
        },
        "features": {
            "decision_grid": "every exact UTC hour D",
            "aligned_window": (
                "360 exact aligned unique bars_upbit KRW-BTC and bars_binance BTCUSDT interval=1m "
                "rows [D-6h,D)"
            ),
            "source_valid": (
                "finite positive coherent OHLC for both venues at every timestamp; no duplicate, "
                "missing, nearest-time join, FX conversion, or imputation"
            ),
            "minute_return": "log(close/open) independently by venue",
            "upbit_variation": "sum squared Upbit minute returns, strict positive",
            "binance_variation": "sum squared Binance minute returns, strict positive",
            "variance_leadership": "log(upbit_variation/binance_variation), finite",
            "leadership_rank": (
                "strict-prior midrank over at most 2,160 valid hourly decisions, minimum 1,440, "
                "current excluded; rank>=0.85"
            ),
            "binance_variation_rank": "same strict-prior 2,160/1,440 rule; rank>=0.65",
            "final_hour_returns": (
                "log(last close/first open) over the final exact 60 minutes independently by venue"
            ),
            "direction_confirmation": "both final-hour returns have one strict nonzero sign",
            "eligible_state": "both rank gates and direction_confirmation pass",
            "onset": "eligible now and immediately preceding exact source-valid hour ineligible",
        },
        "clock": {
            "decision": "D after both completed aligned paths are available",
            "entry": "exact BTCUSDT perpetual D+5m open",
            "side": "common final-hour direction",
            "hold": "6 elapsed hours",
            "reservation": "global half-open; exit first on equal-time entry",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not an input; exact realized funding only after novelty passes",
        },
        "policy": {
            "window_hours": 6,
            "direction_hours": 1,
            "history_hours": 2160,
            "minimum_history_hours": 1440,
            "leadership_rank_min": 0.85,
            "binance_variation_rank_min": 0.65,
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
                "no_binance_variation_gate", "no_variance_leadership_tail",
                "return_magnitude_leadership", "one_hour_stale_features",
                "direction_flip", "forced_long",
            ],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "upbit": {
                "table": "bars_upbit", "symbol": "KRW-BTC", "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close"],
            },
            "binance": {
                "table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close"],
            },
            "window": ["2023-04-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "read_after_preregistration": True,
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_korean_premium_return_volume_outcomes_known": True,
            "repository_upbit_binance_variance_leadership_candidate_found": False,
            "prior_event_sets_reused": False,
            "prior_outcomes_used_to_set_formula_ranks_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent regional-cash variance leadership transmission",
        },
        "stopping_rule": (
            "terminal first failure; no venue, window, variation formula, rank, direction, onset, "
            "side, hold, clock, subset, control, or gate repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVKVLR preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)
