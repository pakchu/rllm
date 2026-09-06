"""Outcome-blind preregistration for HVOILSR-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVOILSR-12"
DEFAULT_OUTPUT = Path("results/high_volatility_oi_lead_sponsorship_relay_preregistration_2026-08-09.json")


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_oi_lead_sponsorship_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "On a high-variation completed BTC day, unusually strong five-minute open-interest "
                "changes that lead the next five-minute price move in the completed-day direction "
                "identify informed leveraged sponsorship; that direction should relay for twelve hours."
            ),
            "side": "strict sign of the completed prior-day BTC return",
            "why_distinct": (
                "HVOILSR uses a fixed daily one-bar lead correlation between OI increments and later "
                "price returns. Prior OI candidates used endpoint change, purge, expansion, divergence, "
                "gross churn/cancellation, contemporaneous inventory work, or round-trip episodes."
            ),
            "volatile_market_target": "completed-day BTC realized-variation causal rank must be at least 0.65",
            "why_low_gross9_overlap_is_plausible": "one OI-lead-conditioned 00:05 UTC daily clock is absent from Gross9",
        },
        "features": {
            "source_day": "exact prior UTC day [D-24h,D)",
            "price_path": "288 exact coherent BTCUSDT perpetual five-minute bars aggregated from 1m",
            "oi_path": "289 exact positive BTCUSDT period=5m observations from D-24h-5m through D-5m; no imputation",
            "lead_pairs": (
                "287 pairs: d_i=log(OI_i/OI_(i-1)) ending at t and r_next=log(close/open) "
                "of the five-minute bar starting t+5m; every paired value is completed by D"
            ),
            "lead_correlation": "Pearson correlation(d_i,r_next), requiring nonzero finite variance in both vectors",
            "directional_lead_score": "sign(completed-day return)*lead_correlation",
            "realized_variation": "sqrt(sum of all 288 squared five-minute log(close/open) returns)",
            "causal_ranks": "strict-prior midranks over at most 270 earlier source-valid days, minimum 180; current excluded",
            "eligibility": "directional-lead-score rank>=0.75 and realized-variation rank>=0.65; completed-day return strict nonzero",
            "availability": "D 00:00 UTC after both completed paths",
            "no_imputation": True,
            "grid": False,
        },
        "clock": {
            "decision": "exact D 00:00 UTC",
            "entry": "exact BTCUSDT perpetual D 00:05 UTC open",
            "side": "sign of completed-day return",
            "hold": "12 elapsed hours",
            "reservation": "daily opportunities are nonoverlapping; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium_implied_vol_rv20": "only frozen OI is an input; exact funding after novelty; RV20 q90 only after all economics pass",
        },
        "policy": {
            "price_bars": 288, "oi_points": 289, "lead_pairs": 287,
            "history_days": 270, "minimum_history_days": 180,
            "lead_score_rank_min": 0.75, "variation_rank_min": 0.65,
            "entry_delay_minutes": 5, "hold_hours": 12, "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006, "stress_cost_per_notional_side": 0.001,
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
        "diagnostic_controls": {"names": ["no_lead_score_gate", "no_variation_gate", "contemporaneous_correlation", "one_day_stale_features", "direction_flip", "same_clock_forced_long"], "cannot_be_promoted": True},
        "source_plan": {
            "oi": {"table": "open_interest_binance", "symbol": "BTCUSDT", "period": "5m", "column": "sum_open_interest", "read_only": True},
            "price": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "high", "low", "close"], "read_only": True},
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {"prior_oi_family_outcomes_known": True, "oi_lead_values_used_to_select_rule": False, "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False, "selection_basis": "independent lagged inventory-information mechanism"},
        "stopping_rule": "terminal first failure; no source grid, lag, rank, history, threshold, side, timing, hold, subset, comparator, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVOILSR preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    result = build(); validate(result); args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"); print(args.output)
