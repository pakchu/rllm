"""Outcome-blind preregistration for EMDFR-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT = Path("results/emerging_market_dollar_factor_relay_preregistration_2026-08-09.json")


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "emerging_market_dollar_factor_relay_v1",
        "policy_id": "EMDFR-12",
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": "A broad standardized dollar move against three emerging-market currencies during the liquid Europe-US overlap transmits inversely into volatile BTC over the next twelve hours.",
            "side": "opposite the strict sign of the median USDMXN, USDINR, and USDCNY standardized session return",
            "why_distinct": "DFSR used six developed-market FX pairs over 13:00-21:00 UTC; MXRBR and SHRR used individual or two-pair residual mechanisms. EMDFR freezes an independent three-country emerging-market dollar factor over 13:00-16:00 UTC and does not reuse any prior candidate event set.",
            "why_suited_to_volatile_regimes": "the completed pre-decision 24-hour BTC realized variation must be in its causal upper 35%",
            "why_low_gross9_overlap_is_plausible": "one sparse 16:05 UTC weekday macro clock is absent from Gross9 primitives",
        },
        "features": {
            "fx_session": "bars_polygon interval=1m rows for USDMXN, USDINR, and USDCNY with 13:00<=UTC time<16:00 on one UTC weekday",
            "fx_session_valid": "at least 165 distinct observed minutes, first timestamp no later than 13:05, last timestamp no earlier than 15:55, finite positive OHLC; no imputation",
            "canonical_dollar_returns": "positive session log return for each USD-quoted emerging-market currency pair",
            "pair_zscores": "each canonical dollar return standardized against its own at most 90 strictly prior valid sessions, minimum 60, current excluded; prior sample std positive",
            "dollar_factor": "cross-sectional median of three finite pair z-scores; strict nonzero",
            "factor_abs_rank": "strict-prior midrank of absolute dollar factor against at most 90 prior valid sessions, minimum 60, current excluded; rank>=0.70",
            "btc_realized_variation": "sqrt(sum of squared completed-hour BTC log returns over the 24 hours ending at decision time)",
            "volatility_rank": "strict-prior midrank against at most 90 valid decision days, minimum 60; current excluded; rank>=0.65",
            "no_imputation": True,
        },
        "clock": {
            "decision": "exact weekday D 16:00 UTC after the FX session and BTC variation are complete",
            "entry": "exact BTCUSDT D 16:05 UTC open",
            "side": "opposite strict emerging-market dollar-factor sign",
            "hold": "12 elapsed hours",
            "reservation": "fixed weekday decisions; global half-open, exit first on equal open",
            "funding_oi_premium_implied_vol": "not signal inputs; exact funding only for later PnL",
        },
        "policy": {
            "fx_prior_sessions": 90,
            "fx_prior_min_sessions": 60,
            "factor_absolute_rank_min": 0.70,
            "realized_variation_rank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 12,
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
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.20, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.10, "candidate_near_6h_share_max": 0.35, "occupied_5m_bar_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.10, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "future_can_rank_repair_or_reselect": False, "accounting": "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "source_plan": {
            "fx": {"table": "bars_polygon", "symbols": ["USDMXN", "USDINR", "USDCNY"], "interval": "1m", "columns": ["ts", "open", "high", "low", "close"], "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"], "materialize_after_preregistration": True},
            "completed_btc": "hash-bound completed-hour BTC source through 2026-08-01 for pre-entry variation only",
            "execution_price": "sealed until source-support and Gross9 novelty pass",
        },
        "diagnostic_controls": {"names": ["no_volatility_gate", "no_factor_tail", "raw_return_factor", "one_session_stale_factor", "direction_flip", "forced_long"], "diagnostic_controls_cannot_be_promoted": True},
        "research_boundary": {
            "prior_dfsr_mxrbr_shrr_outcomes_known": True,
            "prior_fx_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_emdf_source_set_direction_rank_hold_or_clock": False,
            "emdf_exact_incidence_opened": False,
            "emdf_post_entry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent emerging-market dollar-factor transmission mechanism for volatile BTC regimes",
        },
        "stopping_rule": "Terminal first-failure sequence: source support, Gross9 novelty, strict economics; no source set, factor, rank, side, hold, clock, session, or subset repair.",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(), indent=2) + "\n")
    print(args.output)
