"""Outcome-blind preregistration for DVURF-6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

POLICY_ID = "DVURF-6"
DEFAULT_OUTPUT = Path("results/dual_volatility_index_upper_rejection_fade_preregistration_2026-08-09.json")


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "dual_volatility_index_upper_rejection_fade_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": "When both Binance BVOL and Deribit DVOL reject their completed-hour highs while a large BTC move occurs, options volatility repricing has failed to hold its extreme and the contemporaneous BTC displacement is vulnerable to a six-hour fade.",
            "side": "opposite the strict sign of the completed BTC hour return",
            "why_distinct": "OIFAR required joint positive volatility bodies plus OI contraction; cross-venue disagreement families used relative index levels or body polarity; HVWIR used BTC candle wicks. DVURF uses simultaneous within-candle upper-rejection geometry in two volatility indexes, no OI, funding, BTC wick, index-level spread, or prior event set, and promotes no control.",
            "why_suited_to_volatile_regimes": "both volatility-index ranges must be nontrivial and the BTC completed-hour absolute return must be in its causal upper quartile",
            "why_low_gross9_overlap_is_plausible": "paired options-index wick geometry is absent from Gross9 primitives",
        },
        "features": {
            "decision_grid": "every exact UTC hour T after [T-1h,T) closes",
            "volatility_sources": "one causally available completed BVOL hour and one completed DVOL hour joined by close_time T; finite positive coherent OHLC; high>low; no fill or stale carry",
            "upper_rejection": "(high-max(open,close))/(high-low)",
            "lower_rejection": "(min(open,close)-low)/(high-low)",
            "net_upper_rejection": "upper_rejection-lower_rejection independently for BVOL and DVOL",
            "joint_rejection": "minimum of the two net_upper_rejection values; must be strictly positive and rank>=0.75",
            "joint_range": "geometric mean of (high-low)/open for BVOL and DVOL; rank>=0.50",
            "btc_hour": "60 exact BTCUSDT one-minute bars in [T-1h,T); log(last close/first open), strict nonzero; absolute-return rank>=0.75",
            "ranks": "strict-prior midrank over at most 720 paired-valid completed hours, minimum 672, current hour excluded",
            "no_imputation": True,
        },
        "clock": {
            "decision": "exact UTC hour T after all three completed source hours are causally available",
            "entry": "exact BTCUSDT T+5m open",
            "side": "opposite completed-hour BTC return sign",
            "hold": "6 elapsed hours",
            "reservation": "global half-open; earliest signal wins; exit first on equal open",
            "funding_oi_premium_rv20": "not signal inputs; exact funding only after novelty; RV20 q90 only after all economic stages pass",
        },
        "policy": {
            "prior_hours": 720,
            "prior_min_hours": 672,
            "joint_rejection_rank_min": 0.75,
            "joint_range_rank_min": 0.50,
            "btc_absolute_return_rank_min": 0.75,
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
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.20, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.10, "candidate_near_6h_share_max": 0.35, "occupied_5m_bar_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.10, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "accounting": "fixed quantity, exact funding, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "post_stage_volatility_audit": {"prerequisite": "unchanged candidate passes train, test, eval, final", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "diagnostic_controls": {"names": ["btc_shock_only", "bvol_rejection_only", "dvol_rejection_only", "no_joint_range_floor", "one_hour_stale_index_geometry", "direction_flip", "forced_long"], "cannot_be_promoted": True},
        "source_plan": {
            "bvol": {"historical": "data/binance_btc_bvol_hourly_opdr_2023_2026/BTCBVOLUSDT_1h_2023-06-20_2026-06-30.csv.gz", "july_extension": "data/binance_btc_bvol_hourly_ocdr_2026_07/BTCBVOLUSDT_1h_2026-07-01_2026-07-31.csv.gz", "availability": "hour close T"},
            "dvol": {"historical": "data/deribit_btc_dvol_1h_2023-06-20_2026-07-01.csv.gz", "july_extension": "data/deribit_btc_dvol_1h_ocdr_2026-07-01_2026-08-01.csv.gz", "availability": "close_time T"},
            "btc": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "high", "low", "close"], "read_after_preregistration": True},
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_options_reversal_and_disagreement_outcomes_known": True,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_dvurf_formula_direction_rank_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "paired volatility-index upper-rejection geometry during a large BTC hour",
        },
        "stopping_rule": "Terminal first-failure sequence: source support, Gross9 novelty, strict economics; no wick formula, rank, side, hold, clock, subset, or control repair.",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)
