"""Outcome-blind preregistration for HVRSR-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

POLICY_ID = "HVRSR-12"
DEFAULT_OUTPUT = Path("results/high_volatility_realized_skew_reversal_relay_preregistration_2026-08-09.json")


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_realized_skew_reversal_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": "An extreme signed third moment in completed intraday BTC returns measures one-sided jump concentration rather than total variance. In an already high-variation state, the skewed flow is exhausted and reverses over the next twelve hours.",
            "side": "opposite the strict sign of completed 24-hour realized skew",
            "why_distinct": "HVSIR used positive-versus-negative second-moment mass and an intraday threshold crossing; HVSER used return-sign entropy; BPJR used bipower jump variation. HVRSR uses a daily normalized third moment and strict-prior ranks, reuses no event set, and promotes no control.",
            "why_suited_to_volatile_regimes": "completed 24-hour realized variation must rank in its causal upper 35%",
            "why_low_gross9_overlap_is_plausible": "one sparse 02:05 UTC daily third-moment clock is absent from Gross9 primitives",
        },
        "features": {
            "decision_grid": "every calendar day at exact 02:00 UTC",
            "return_window": "288 completed five-minute close-to-close log returns in [decision-24h, decision), requiring 289 contiguous finite positive source bars including the predecessor close",
            "realized_variation": "sqrt(sum(return^2))",
            "realized_skew": "sum(return^3) / sum(return^2)^(3/2), strict nonzero and finite",
            "variation_rank": "strict-prior midrank over at most 90 prior valid daily observations, minimum 60, current excluded; rank>=0.65",
            "absolute_skew_rank": "strict-prior midrank of abs(realized_skew) over at most 90 prior valid daily observations, minimum 60, current excluded; rank>=0.70",
            "no_imputation": True,
        },
        "clock": {
            "decision": "exact daily 02:00 UTC after all 288 returns are complete",
            "entry": "exact BTCUSDT 02:05 UTC open",
            "side": "opposite strict realized-skew sign",
            "hold": "12 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "funding_oi_premium": "not signal inputs; exact funding only after novelty passes",
        },
        "policy": {
            "return_bars": 288,
            "prior_days": 90,
            "prior_min_days": 60,
            "variation_rank_min": 0.65,
            "absolute_skew_rank_min": 0.70,
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
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.10, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "accounting": "fixed quantity, exact funding, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "post_stage_volatility_audit": {"prerequisite": "unchanged candidate passes train, test, eval, final", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "diagnostic_controls": {"names": ["no_volatility_gate", "no_skew_tail", "raw_third_moment", "one_day_stale_features", "direction_flip", "forced_long"], "cannot_be_promoted": True},
        "source_plan": {"bars": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "5m", "columns": ["ts", "open", "high", "low", "close"], "read_after_preregistration": True}, "execution_prices": "sealed until source support and Gross9 novelty pass"},
        "research_boundary": {
            "prior_hvsir_hvser_bpjr_outcomes_known": True,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_hvrsr_direction_rank_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "normalized third-moment exhaustion in volatile BTC regimes",
        },
        "stopping_rule": "Terminal first-failure sequence: source support, Gross9 novelty, strict economics; no moment definition, rank, side, hold, clock, subset, or control repair.",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)
