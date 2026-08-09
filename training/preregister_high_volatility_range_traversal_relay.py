"""Outcome-blind preregistration for HVRTR-12."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any

POLICY_ID = "HVRTR-12"
DEFAULT_OUTPUT = Path("results/high_volatility_range_traversal_relay_preregistration_2026-08-09.json")

def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_range_traversal_relay_v1", "policy_id": POLICY_ID,
        "as_of_date": "2026-08-09", "outcomes_opened": False, "source_incidence_opened": False,
        "gross9_rows_opened": False, "singleton": True,
        "mechanism": {
            "claim": "On a high-range completed BTC day, a low established before the high followed by a close in the upper quartile is an upward range traversal; the symmetric high-before-low lower-quartile close is a downward traversal. A completed traversal should relay for twelve hours.",
            "side": "long for low-before-high upper-quartile completion; short for high-before-low lower-quartile completion",
            "why_distinct": "DRCMR used range and close location without temporal ordering. LEMRR used only Tuesday/Friday and faded the later extreme after midpoint rejection. HVRTR uses every exact UTC day, the first occurrence order of both absolute extremes, and follows completed traversal rather than rejection.",
            "volatile_market_target": "completed-day log high-low range causal rank must be at least 0.65",
            "why_low_gross9_overlap_is_plausible": "one range-order-conditioned 00:05 UTC clock is absent from Gross9",
        },
        "features": {
            "source_day": "exact prior UTC day with 1,440 unique coherent BTCUSDT perpetual 1m bars",
            "daily_ohlc": "first open, maximum high, minimum low, last close; all finite positive",
            "first_high_time": "earliest minute timestamp attaining the completed-day absolute high",
            "first_low_time": "earliest minute timestamp attaining the completed-day absolute low",
            "log_range": "log(daily_high/daily_low), strict positive",
            "range_rank": "strict-prior midrank over at most 270 prior source-valid days, minimum 180; current excluded; rank>=0.65",
            "close_location": "(daily_close-daily_low)/(daily_high-daily_low)",
            "long_completion": "first_low_time < first_high_time, close_location>=0.75, and close>open",
            "short_completion": "first_high_time < first_low_time, close_location<=0.25, and close<open",
            "ties": "equal extreme times or ambiguous/flat direction are ineligible", "no_imputation": True,
        },
        "clock": {"decision": "exact D 00:00 UTC after prior day completes", "entry": "exact BTCUSDT D 00:05 UTC open", "side": "traversal completion direction", "hold": "12 elapsed hours", "reservation": "daily opportunities are nonoverlapping; exit first on equal open", "split_crossing_action": "skip", "gross_exposure": 0.5, "funding_oi_premium_rv20": "not signal inputs; funding after novelty; RV20 q90 only after all economics pass"},
        "policy": {"history_days": 270, "minimum_history_days": 180, "range_rank_min": 0.65, "upper_close_location_min": 0.75, "lower_close_location_max": 0.25, "entry_delay_minutes": 5, "hold_hours": 12, "leverage": 0.5, "base_cost_per_notional_side": 0.0006, "stress_cost_per_notional_side": 0.001},
        "stages": {"train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"], "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"], "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"], "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"]},
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.20, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.10, "candidate_near_6h_share_max": 0.35, "occupied_5m_bar_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.10, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "accounting": "fixed quantity, exact funding, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "post_stage_volatility_audit": {"prerequisite": "unchanged candidate passes train, test, eval, final", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "diagnostic_controls": {"names": ["no_range_gate", "no_extreme_order", "no_close_location", "one_day_stale_geometry", "direction_flip", "same_clock_forced_long"], "cannot_be_promoted": True},
        "source_plan": {"bars": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "high", "low", "close"], "read_only": True}, "execution_prices": "sealed until source support and Gross9 novelty pass"},
        "research_boundary": {"related_price_geometry_outcomes_known": True, "traversal_incidence_used_to_select_rule": False, "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False, "selection_basis": "independent temporal auction-completion mechanism"},
        "stopping_rule": "terminal first failure; no extreme occurrence rule, range, rank, history, close location, side, timing, hold, subset, comparator, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}

def validate(payload: dict[str, Any]) -> None:
    core = {k: v for k, v in payload.items() if k != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core): raise RuntimeError("HVRTR preregistration drift")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    result = build(); validate(result); args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"); print(args.output)
