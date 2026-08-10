"""Outcome-blind preregistration for HVCACFR-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVCACFR-8"
SLUG = "high_volatility_cross_alt_correlation_fracture_resolution_relay"
ALTS = ["ADAUSDT", "BNBUSDT", "DOGEUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
SYMBOLS = ["BTCUSDT", *ALTS]
DEFAULT_OUTPUT = Path(
    "results/high_volatility_cross_alt_correlation_fracture_resolution_relay_"
    "preregistration_2026-08-10.json"
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
    core = {
        "protocol_version": (
            "high_volatility_cross_alt_correlation_fracture_resolution_relay_v1"
        ),
        "policy_id": POLICY_ID,
        "slug": SLUG,
        "as_of_date": "2026-08-10",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "A sharp completed intraday fracture from high first-half BTC correlation with the "
                "fixed six-alt median basket to lower second-half correlation marks a transfer of "
                "price discovery away from BTC. During elevated BTC variation, the basket's final "
                "two-hour direction should resolve into BTC over the next eight hours."
            ),
            "side": "strict sign of the sum of the final 120 median-alt minute returns",
            "why_distinct": (
                "HVCACFR measures the change between two within-window BTC-versus-fixed-alt-basket "
                "Pearson correlations. Prior cross-alt, ETH-disagreement, and leadership candidates "
                "did not use this exact correlation-fracture geometry, daily clock, or side rule."
            ),
            "volatile_market_target": "BTC completed twelve-hour variation strict-prior rank >=0.65",
            "why_low_gross9_overlap_is_plausible": (
                "03:05 UTC false-to-true onsets of a fixed six-alt correlation fracture are absent "
                "from known Gross9 primitives"
            ),
        },
        "features": {
            "decision_grid": "one exact daily decision D at 03:00 UTC",
            "universe": SYMBOLS,
            "aligned_window": (
                "720 exact aligned unique bars_binance interval=1m OHLC rows [D-12h,D) "
                "for each symbol"
            ),
            "source_valid": (
                "every symbol has exactly 720 unique timestamps on the common minute grid; every "
                "open, high, low, and close is finite and positive with high>=max(open,close) and "
                "low<=min(open,close); no duplicate, missing, nearest-time join, or imputation"
            ),
            "minute_return": "for each symbol and minute, natural log(close/open)",
            "median_alt_minute_return": (
                "at each aligned minute, the fixed cross-sectional median of the six alt minute "
                "returns; for six values this is the arithmetic mean of the third and fourth values "
                "after ascending sort"
            ),
            "first_half_correlation": (
                "Pearson correlation of the 360 BTC minute returns and 360 median-alt minute returns "
                "over [D-12h,D-6h), using covariance divided by the square root of the two population "
                "variances"
            ),
            "second_half_correlation": (
                "Pearson correlation of the 360 BTC minute returns and 360 median-alt minute returns "
                "over [D-6h,D), using covariance divided by the square root of the two population "
                "variances"
            ),
            "correlation_validity": (
                "both 360-element vectors in each half are finite and each has strictly positive "
                "population variance; both correlations must be finite"
            ),
            "correlation_fracture": "first_half_correlation - second_half_correlation",
            "correlation_fracture_rank": (
                "strict-prior midrank of correlation_fracture among at most 180 earlier source-valid "
                "daily decisions, minimum 90, current excluded; rank>=0.75"
            ),
            "btc_variation": (
                "sum of all 720 squared BTC natural-log close/open minute returns over [D-12h,D)"
            ),
            "btc_variation_rank": (
                "own strict-prior midrank among at most 180 earlier source-valid daily decisions, "
                "minimum 90, current excluded; rank>=0.65"
            ),
            "direction": (
                "strict sign of the sum of the final 120 median-alt minute returns over [D-2h,D); "
                "the sum must be finite and strictly nonzero"
            ),
            "btc_direction_confirmation": False,
            "eligible_state": (
                "source and correlation validity pass, correlation-fracture rank>=0.75, BTC-variation "
                "rank>=0.65, and final-120-minute median-alt direction is strict nonzero"
            ),
            "onset": (
                "eligible today and the immediately previous exact source-valid daily decision is "
                "ineligible; without a previous source-valid daily decision no onset can trigger"
            ),
            "no_imputation": True,
        },
        "clock": {
            "decision": "exact daily 03:00 UTC D after all seven completed twelve-hour paths are available",
            "entry": "exact BTCUSDT perpetual D+5m open",
            "side": "final-120-minute median-alt direction without BTC direction confirmation",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal-time entry",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact realized funding only after novelty passes",
        },
        "policy": {
            "window_minutes": 720,
            "half_window_minutes": 360,
            "direction_window_minutes": 120,
            "cross_section_size": 6,
            "prior_valid_days": 180,
            "minimum_prior_valid_days": 90,
            "correlation_fracture_rank_min": 0.75,
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
                "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, "
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
                "no_correlation_fracture_gate",
                "contemporaneous_full_window_correlation",
                "one_day_stale_features",
                "direction_flip",
                "forced_long",
            ],
            "definitions": {
                "contemporaneous_full_window_correlation": (
                    "replace correlation_fracture and its rank gate with the Pearson correlation of "
                    "the full 720 BTC and median-alt minute-return vectors, requiring finite vectors "
                    "with strictly positive population variance and its own strict-prior midrank "
                    "among at most 180 earlier source-valid daily decisions, minimum 90, current "
                    "excluded; rank>=0.75"
                ),
                "one_day_stale_features": (
                    "use the complete feature geometry from the immediately previous source-valid "
                    "daily decision on the unchanged decision, entry, and hold clock"
                ),
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
                "query_window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_cross_alt_outcomes_known": True,
            "prior_eth_disagreement_outcomes_known": True,
            "prior_leadership_outcomes_known": True,
            "repository_exact_basket_correlation_fracture_candidate_found": False,
            "prior_event_sets_reused": False,
            "prior_outcomes_used_to_set_formula_rank_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent fixed-basket within-window correlation-fracture mechanism",
        },
        "stopping_rule": (
            "terminal first failure; no universe, basket, window, half, formula, rank, onset, side, "
            "hold, clock, subset, control, or gate repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVCACFR preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(args.output)
