"""Outcome-blind preregistration for HVUSCCA-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("results/high_volatility_us_cash_close_acceptance_relay_preregistration_2026-08-10.json")
EXCLUDED_LOCAL_DATES = (
    "2023-01-02", "2023-01-16", "2023-02-20", "2023-04-07", "2023-05-29", "2023-06-19", "2023-07-03", "2023-07-04", "2023-09-04", "2023-11-23", "2023-11-24", "2023-12-25",
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27", "2024-06-19", "2024-07-03", "2024-07-04", "2024-09-02", "2024-11-28", "2024-11-29", "2024-12-24", "2024-12-25",
    "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26", "2025-06-19", "2025-07-03", "2025-07-04", "2025-09-01", "2025-11-27", "2025-11-28", "2025-12-24", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-02", "2026-07-03",
)


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_us_cash_close_acceptance_relay_v1", "policy_id": "HVUSCCA-8", "as_of_date": "2026-08-10",
        "oos_outcomes_opened": False, "oos_source_incidence_opened": False, "gross9_rows_opened": False, "singleton": True,
        "mechanism": {
            "claim": "A large, efficient BTC move during the full 09:30-16:00 America/New_York cash session that finishes near its directional extreme represents institutional-session acceptance rather than a transient spike. During elevated BTC variation, follow that completed direction for eight hours.",
            "side": "strict sign of the completed cash-session BTC return",
            "why_distinct": "prior candidates use UTC blocks, overnight moves, opening ranges, macro series, or cross-asset closes; no candidate uses the DST-aware full US cash session jointly with path efficiency and terminal acceptance",
            "volatile_market_target": "strict-prior prior-24h BTC realized-variation rank >=0.65",
            "why_low_gross9_overlap_is_plausible": "sparse full-session acceptance onsets at DST-varying 16:05 ET are absent from Gross9",
        },
        "source_contract": {
            "table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m",
            "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "timezone": "America/New_York", "session": "[09:30,16:00) local",
            "eligible_dates": "Monday-Friday local dates excluding the frozen full-closure and early-close list",
            "excluded_local_dates": list(EXCLUDED_LOCAL_DATES),
            "row_validity": "unique exact UTC minute; finite positive OHLC; high>=max(open,close), low<=min(open,close), high>=low",
            "session_validity": "exact 390 expected DST-resolved minute rows with no missing, duplicate, ambiguous, or imputed row",
            "causal_availability": "15:59 local bar is complete at 16:00 local; decision at 16:00 and entry at 16:05 local",
            "transport": "read-only PostgreSQL query with SQL hash, physical count, and normalized state hash",
            "revision_boundary": "database snapshot hashes freeze replay vintage; no first-seen revision archive is claimed",
            "bounded_probe_disclosure": "only table schema and per-symbol min/max/count metadata were opened; no candidate incidence, post-entry outcome, or Gross9 row was read",
        },
        "feature_contract": {
            "session_return": "log(last minute close / first minute open)",
            "session_efficiency": "abs(session_return) divided by sum absolute log increments over [first open, each minute close]",
            "terminal_location": "(last close - minimum session low)/(maximum session high - minimum session low); zero range invalid",
            "absolute_return_rank": "strict-prior 252/126 source-session midrank of abs(session_return)",
            "efficiency_rank": "strict-prior 252/126 source-session midrank of session_efficiency",
            "btc_variation": "sum squared close-to-close log returns over exact 1,440 completed minutes ending at decision",
            "btc_variation_rank": "strict-prior 252/126 source-session midrank",
            "eligible_state": "absolute_return_rank>=0.75, efficiency_rank>=0.65, btc_variation_rank>=0.65, and positive return with terminal_location>=0.80 or negative return with terminal_location<=0.20",
            "onset": "current eligible full session and immediately previous eligible-date session ineligible", "no_imputation": True,
        },
        "oos_clock": {"start": "2023-07-01T00:00:00Z", "decision": "16:00 America/New_York", "entry": "decision+5m BTCUSDT perpetual open", "side": "sign(session_return)", "hold": "8 elapsed hours", "reservation": "chronological first eligible onset while flat; half-open intervals and exit first on equal time", "funding": "opened only after novelty passes"},
        "policy": {"history_sessions": 252, "minimum_history_sessions": 126, "absolute_return_rank_min": 0.75, "efficiency_rank_min": 0.65, "terminal_location_tail": 0.20, "btc_variation_rank_min": 0.65, "entry_delay_minutes": 5, "hold_hours": 8, "leverage": 0.5, "base_cost_per_notional_side": 0.0006, "stress_cost_per_notional_side": 0.001},
        "stages": {"train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"], "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"], "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"], "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"]},
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.20, "max_month_share": 0.45, "complete_session_required": True},
        "novelty_gates": {"exact_entry_jaccard_max": 0.10, "candidate_near_6h_share_max": 0.35, "occupied_5m_bar_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.10, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "accounting": "fixed quantity, exact funding, 6bp/10bp per notional side, favorable-then-adverse held 5m path, global HWM, full-calendar CAGR"},
        "post_stage_volatility_audit": {"prerequisite": "unchanged passes every sequential economic stage", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "diagnostic_controls": {"names": ["no_volatility_gate", "no_efficiency_gate", "no_terminal_acceptance_gate", "one_session_stale_geometry", "direction_flip", "forced_long"], "cannot_be_promoted": True},
        "research_boundary": {"prior_generic_intraday_trend_outcomes_known": True, "prior_dst_full_cash_session_acceptance_outcomes_known": False, "candidate_specific_incidence_opened": False, "candidate_specific_postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False, "selection_basis": "independent fixed-session acceptance mechanism selected before incidence or outcomes"},
        "stopping_rule": "freeze preregistration, source support, Gross9 novelty, and sequential economics; terminal first failure with no calendar, session, history, rank, efficiency, terminal-location, volatility gate, onset, side, delay, hold, subset, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    result = build(); args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n"); print(args.output)
