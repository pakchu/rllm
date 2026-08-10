"""Outcome-blind preregistration for HVTAMC-8."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any

POLICY_ID = "HVTAMC-8"
SLUG = "high_volatility_trade_arrival_memory_continuation"
DEFAULT_OUTPUT = Path("results/high_volatility_trade_arrival_memory_continuation_preregistration_2026-08-10.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_trade_arrival_memory_continuation_v1",
        "policy_id": POLICY_ID, "slug": SLUG, "as_of_date": "2026-08-10",
        "outcomes_opened": False, "source_incidence_opened": False, "gross9_rows_opened": False, "singleton": True,
        "mechanism": {
            "claim": "Positive minute-to-minute memory in completed trade-arrival counts identifies persistent execution intensity rather than a one-off activity burst. When that memory enters its upper tail during elevated realized variation, persistent participation should carry the completed displacement for eight hours.",
            "side": "strict sign of the completed eight-hour BTC return",
            "why_distinct": "Trade-arrival backloading uses the final-hour share; directional trade breadth allocates counts by return sign; ticket candidates use volume per trade; price serial-autocorrelation candidates ignore executions. HVTAMC uses only lag-one serial dependence of minute execution counts for eligibility and reuses no prior event set or control.",
            "volatile_market_target": "completed eight-hour realized-variation strict-prior rank >=0.65",
            "why_low_gross9_overlap_is_plausible": "three daily onsets from execution-count memory are absent from Gross9 primitives",
        },
        "features": {
            "decision_grid": "exact 00:00, 08:00 and 16:00 UTC boundaries D",
            "source_window": "480 exact coherent bars_binance BTCUSDT interval=1m rows [D-8h,D)",
            "trade_counts": "finite nonnegative integer number_of_trades on every minute; strict positive block total and nonconstant lag vectors",
            "arrival_memory": "Pearson correlation between number_of_trades[0:479] and number_of_trades[1:480], finite in [-1,1]",
            "memory_rank": "strict-prior midrank of arrival_memory over at most 270 earlier source-valid blocks, minimum 180; current excluded; rank>=0.75",
            "realized_variation": "sum squared log(close/open) over 480 one-minute rows, finite strict positive",
            "variation_rank": "strict-prior 270/180 midrank; current excluded; rank>=0.65",
            "completed_return": "log(last close/first open), finite strict nonzero",
            "eligible_state": "memory and variation gates pass",
            "onset": "eligible now and immediately previous exact source-valid block ineligible",
            "no_imputation": True,
        },
        "clock": {"decision": "D after the full path completes", "entry": "exact BTCUSDT D+5m open", "side": "strict sign(completed_return)", "hold": "8 elapsed hours", "reservation": "global half-open; exit first on equal entry", "split_crossing_action": "skip", "gross_exposure": 0.5, "funding": "not a signal input; exact realized funding only after novelty"},
        "policy": {"window_minutes": 480, "memory_pairs": 479, "prior_blocks": 270, "minimum_prior_blocks": 180, "memory_rank_min": 0.75, "variation_rank_min": 0.65, "entry_delay_minutes": 5, "hold_hours": 8, "leverage": 0.5, "base_cost_per_notional_side": 0.0006, "stress_cost_per_notional_side": 0.001},
        "stages": {"train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"], "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"], "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"], "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"]},
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.2, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.1, "candidate_near_6h_share_max": 0.35, "occupied_5m_bar_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.1, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "accounting": "fixed quantity, exact funding, 6bp/10bp per notional side, held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "post_stage_volatility_audit": {"prerequisite": "unchanged all-stage pass", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "diagnostic_controls": {"names": ["no_memory_tail_gate", "no_variation_gate", "quote_volume_memory", "one_block_stale_geometry", "direction_flip", "forced_long"], "cannot_be_promoted": True},
        "source_plan": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "high", "low", "close", "quote_asset_volume", "number_of_trades"], "query_window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"], "read_after_preregistration": True, "execution_prices": "sealed until source and novelty pass"},
        "research_boundary": {"prior_trade_arrival_backloading_breadth_ticket_and_price_serial_dependence_outcomes_known": True, "repository_trade_count_lag_one_memory_candidate_found": False, "prior_event_sets_or_controls_reused": False, "prior_outcomes_used_to_set_formula_rank_side_hold_or_clock": False, "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False, "selection_basis": "independent serial dependence of completed minute execution counts"},
        "stopping_rule": "terminal first failure; no window, memory functional, rank, onset, side, clock, hold, subset, threshold, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVTAMC preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    registration = build(); validate(registration)
    args.output.write_text(json.dumps(registration, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
