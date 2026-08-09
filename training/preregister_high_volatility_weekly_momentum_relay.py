"""Outcome-blind preregistration for HVWMR-72."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

POLICY_ID = "HVWMR-72"
DEFAULT_OUTPUT = Path("results/high_volatility_weekly_momentum_relay_preregistration_2026-08-09.json")


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_weekly_momentum_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": "A completed seven-day BTC move formed under elevated realized variation represents persistent weekly risk transfer rather than a quiet drift; its direction should continue over the next seventy-two hours.",
            "side": "strict sign of the completed seven-day BTC log return",
            "why_distinct": "STCR used Monday/Thursday decisions and required common signs of separate completed three-day and fourteen-day returns without a volatility gate. HVWMR uses one Wednesday weekly decision, one exact seven-day return, and a causal weekly realized-variation state; it reuses no STCR event set or control and is not a schedule, horizon, or gate repair of STCR.",
            "why_suited_to_volatile_regimes": "completed seven-day realized variation must rank in its causal upper 40%",
            "why_low_gross9_overlap_is_plausible": "one Wednesday 00:05 UTC weekly high-variation clock is absent from Gross9 primitives",
        },
        "features": {
            "decision_grid": "every Wednesday at exact 00:00 UTC",
            "return_window": "2016 completed five-minute close-to-close log returns in [decision-7d,decision), requiring 2017 contiguous finite positive closes including the predecessor close",
            "weekly_return": "sum of all 2016 completed five-minute log returns; strict nonzero",
            "weekly_realized_variation": "sqrt(sum of all 2016 return squares)",
            "variation_rank": "strict-prior midrank over at most 52 prior source-valid weekly decisions, minimum 26, current excluded; rank>=0.60",
            "no_imputation": True,
        },
        "clock": {
            "decision": "exact Wednesday 00:00 UTC after the seven-day path is complete",
            "entry": "exact BTCUSDT Wednesday 00:05 UTC open",
            "side": "strict weekly-return sign",
            "hold": "72 elapsed hours",
            "reservation": "weekly opportunities are nonoverlapping; global half-open, exit first on equal open",
            "funding_oi_premium_implied_vol_rv20": "not signal inputs; exact funding only after novelty; RV20 q90 only after all economic stages pass",
        },
        "policy": {
            "return_bars": 2016,
            "prior_weeks": 52,
            "prior_min_weeks": 26,
            "variation_rank_min": 0.60,
            "entry_delay_minutes": 5,
            "hold_hours": 72,
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
        "diagnostic_controls": {"names": ["no_volatility_gate", "one_week_stale_features", "direction_flip", "forced_long"], "cannot_be_promoted": True},
        "source_plan": {"bars": {"historical": "hash-bound completed BTCUSDT 5m cache through 2026-06-01", "live_extension": "read-only Postgres completed bars through 2026-08-01", "columns": ["date", "open", "high", "low", "close"], "read_after_preregistration": True}, "execution_prices": "sealed until source support and Gross9 novelty pass"},
        "research_boundary": {
            "prior_scheduled_trend_outcomes_known": True,
            "prior_event_sets_reused": False,
            "prior_candidate_controls_promoted": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "selection_basis": "independent weekly high-realized-variation momentum mechanism",
        },
        "stopping_rule": "Terminal first-failure sequence: source support, Gross9 novelty, strict economics; no weekday, window, variation rank, side, hold, clock, subset, or control repair.",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    result = build(); args.output.write_text(json.dumps(result, indent=2) + "\n"); print(args.output)
