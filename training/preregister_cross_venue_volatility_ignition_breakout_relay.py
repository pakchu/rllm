"""Outcome-blind preregistration for CVVIB-6."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT = Path("results/cross_venue_volatility_ignition_breakout_relay_preregistration_2026-08-08.json")

def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def build() -> dict[str, Any]:
    core = {
        "protocol_version": "cross_venue_volatility_ignition_breakout_relay_v1",
        "policy_id": "CVVIB-6",
        "as_of_date": "2026-08-08",
        "outcomes_opened": False,
        "mechanism": {
            "claim": "At a fixed four-hour UTC boundary, joint BVOL and DVOL expansion that accelerates in the final hour after three hours of price compression marks a newly ignited volatile regime; the final-hour BTC breakout direction should relay for six hours.",
            "side": "sign of the fourth completed hour BTC return",
            "why_distinct": "CVVIB uses nonoverlapping fixed four-hour blocks, joint multi-hour implied-vol expansion, three-hour price compression, and a final-hour breakout. It uses neither opposing venue-vol bodies nor intrahour absorption nor funding direction.",
            "why_low_gross9_overlap_is_plausible": "the fixed four-hour cross-venue implied-vol ignition clock and completed-block geometry are absent from Gross9",
        },
        "clock": {
            "decision": "exact UTC four-hour boundary D after four consecutive completed source-valid hours [D-4h,D)",
            "volatility_expansion": "BVOL and DVOL each close above their respective first-hour open across the four-hour block",
            "late_ignition": "BVOL and DVOL fourth-hour normalized bodies are both strictly positive",
            "price_compression": "absolute BTC return over the first three hours <= strict-prior 180-block q50 with 168 observations",
            "price_breakout": "nonzero absolute BTC return in the fourth hour >= strict-prior 180-block q60 with 168 observations",
            "entry": "exact BTCUSDT D+5m open",
            "side": "sign of fourth-hour BTC return",
            "hold": "6 elapsed hours",
            "reservation": "global half-open; chronological signals; exit first on equal open",
            "oi": "not a signal input",
            "funding": "not a signal input; opened only for later exact PnL accounting",
            "no_imputation": True,
        },
        "policy": {"block_hours": 4, "prior_blocks": 180, "prior_min_blocks": 168, "compression_quantile": 0.50, "breakout_quantile": 0.60, "entry_delay_minutes": 5, "hold_hours": 6, "leverage": 0.5, "base_cost_per_notional_side": 0.0006, "stress_cost_per_notional_side": 0.001},
        "stages": {"train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"], "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"], "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"], "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"]},
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.20, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.10, "candidate_near_6h_share_max": 0.45, "occupied_5m_jaccard_max": 0.30, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.10, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "future_can_rank_repair_or_reselect": False, "accounting": "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "source_plan": {"volatility": "reuse hash-bound OCDR-12C BVOL/DVOL snapshot", "completed_price_features": "reuse hash-bound OLIAH completed-hour price snapshot", "execution_price": "sealed until source-support and Gross9 novelty pass"},
        "research_boundary": {"prior_candidate_incidence_known": True, "prior_candidate_outcomes_used_for_cvvib": False, "cvvib_candidate_incidence_opened": False, "cvvib_post_entry_return_or_pnl_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False},
    }
    return {**core, "manifest_hash": canonical_hash(core)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    args.output.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n"); print(args.output)
