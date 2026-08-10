"""Outcome-blind preregistration for HVCARP-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVCARP-8"
ALTS = ["ADAUSDT", "BNBUSDT", "DOGEUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
SYMBOLS = ["BTCUSDT", *ALTS]
DEFAULT_OUTPUT = Path(
    "results/high_volatility_cross_alt_rank_persistence_relay_preregistration_2026-08-10.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_cross_alt_rank_persistence_relay_v1",
        "policy_id": POLICY_ID,
        "slug": "high_volatility_cross_alt_rank_persistence_relay",
        "as_of_date": "2026-08-10",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "Persistent cross-sectional leadership order across the first and second halves of a "
                "completed six-hour alt-perpetual block identifies durable relative price discovery. "
                "During elevated BTC variation, follow the common second-half alt and BTC direction "
                "when that leadership-order persistence first enters its strict-prior tail."
            ),
            "side": "common strict sign of median second-half alt return and BTC second-half return",
            "why_distinct": (
                "HVCARP compares the exact cross-alt ranks of six first-half returns with their ranks "
                "in the second half. Prior cross-alt work used breadth, residuals, or leadership, and "
                "prior BTC temporal-rank work ranked one asset's price levels against minute order; "
                "none is this fixed two-half cross-sectional leadership-order persistence geometry."
            ),
            "volatile_market_target": "BTC completed six-hour variation strict-prior rank >=0.65",
            "why_low_gross9_overlap_is_plausible": (
                "hourly false-to-true onsets of exact six-alt two-half rank persistence are absent "
                "from known Gross9 primitives"
            ),
        },
        "features": {
            "decision_grid": "every exact UTC hour D",
            "universe": SYMBOLS,
            "aligned_window": "360 exact aligned unique bars_binance interval=1m rows [D-6h,D) per symbol",
            "source_valid": (
                "finite positive coherent OHLC for every symbol and timestamp; exactly 360 rows per "
                "symbol with no duplicate, missing, nearest-time join, or imputation"
            ),
            "alt_first_half_return": (
                "for each alt, natural log(last close/first open) over the first 180 aligned minutes [D-6h,D-3h)"
            ),
            "alt_second_half_return": (
                "for each alt, natural log(last close/first open) over the last 180 aligned minutes [D-3h,D)"
            ),
            "strict_cross_section": (
                "all six first-half returns and all six second-half returns are finite and strictly "
                "distinct within their half; ties invalidate the decision and receive no tie handling"
            ),
            "rank_persistence": (
                "Spearman rank correlation across the six alts between first-half and second-half "
                "returns: assign unique ascending ranks R1_j and R2_j in {1,...,6}, then rho = "
                "1 - 6*sum_j((R1_j-R2_j)^2)/(6*(6^2-1)); rho must be finite"
            ),
            "rank_persistence_rank": (
                "strict-prior midrank of rho over at most 2,160 valid hourly decisions, minimum 1,440, "
                "current excluded; rank>=0.80"
            ),
            "btc_variation": "sum of 360 squared natural-log BTC close/open returns over [D-6h,D)",
            "btc_variation_rank": (
                "strict-prior midrank over at most 2,160 valid hourly decisions, minimum 1,440, "
                "current excluded; rank>=0.65"
            ),
            "direction_inputs": (
                "median of the six second-half alt returns and BTC natural log(last close/first open) "
                "over the same last 180 minutes"
            ),
            "direction_confirmation": (
                "the median second-half alt return is strictly nonzero and the BTC second-half return "
                "shares its sign"
            ),
            "eligible_state": "rank-persistence, BTC-variation, and direction-confirmation gates pass",
            "onset": (
                "eligible now and immediately preceding exact source-valid hour ineligible; a missing "
                "or source-invalid previous hour cannot trigger"
            ),
            "no_imputation": True,
        },
        "clock": {
            "decision": "D after all seven completed six-hour paths are available",
            "entry": "exact BTCUSDT perpetual D+5m open",
            "side": "confirmed second-half direction",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal-time entry",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not an input; exact realized funding only after novelty passes",
        },
        "policy": {
            "window_hours": 6,
            "half_window_minutes": 180,
            "cross_section_size": 6,
            "history_hours": 2160,
            "minimum_history_hours": 1440,
            "rank_persistence_rank_min": 0.80,
            "btc_variation_rank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 8,
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
                "no_btc_variation_gate",
                "no_rank_persistence_gate",
                "second_half_rank_reversal",
                "one_hour_stale_features",
                "direction_flip",
                "forced_long",
            ],
            "definitions": {
                "second_half_rank_reversal": (
                    "replace the primary strict-prior rank-persistence gate with current Spearman "
                    "rho<=-0.80 while retaining the same strict cross-section and direction rule"
                ),
                "one_hour_stale_features": "use the complete feature geometry from D-1h on the unchanged clock",
                "direction_flip": "negate the candidate side on the unchanged event set and clock",
                "forced_long": "force long on the unchanged event set and clock",
            },
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "bars": {
                "table": "bars_binance",
                "symbols": SYMBOLS,
                "interval": "1m",
                "columns": ["ts", "symbol", "open", "high", "low", "close"],
                "window": ["2023-04-01T00:00:00Z", "2026-08-01T00:00:00Z"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_cross_alt_breadth_residual_leadership_outcomes_known": True,
            "prior_btc_temporal_rank_outcomes_known": True,
            "repository_exact_cross_alt_two_half_rank_persistence_candidate_found": False,
            "prior_event_sets_reused": False,
            "prior_outcomes_used_to_set_formula_rank_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent cross-sectional leadership-order persistence",
        },
        "stopping_rule": (
            "terminal first failure; no universe, window, half, formula, rank, onset, side, hold, "
            "clock, subset, control, or gate repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVCARP preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)
