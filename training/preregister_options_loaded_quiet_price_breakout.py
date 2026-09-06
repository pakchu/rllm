"""Outcome-blind preregistration for OLQPB-6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("results/options_loaded_quiet_price_breakout_preregistration_2026-08-08.json")


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "options_loaded_quiet_price_breakout_v1",
        "policy_id": "OLQPB-6",
        "as_of_date": "2026-08-08",
        "outcomes_opened": False,
        "mechanism": {
            "claim": (
                "simultaneously elevated Binance and Deribit implied-volatility levels, "
                "quiet completed-hour BTC movement, rising OI, and non-extreme funding "
                "identify latent inventory loading before a directional volatility release"
            ),
            "side": "sign of the quiet completed-hour BTC return",
            "why_distinct_from_recent_options_candidates": (
                "OLQPB uses implied-volatility levels rather than bodies, excludes realized "
                "shock tails, follows rather than fades price, and excludes funding extremes"
            ),
            "why_distinct_from_gross9": (
                "the trigger is a cross-venue implied-volatility level and derivatives "
                "inventory coil absent from the five frozen Gross9 sleeve mechanisms"
            ),
        },
        "clock": {
            "decision": "T after a completed UTC hour",
            "volatility": (
                "BVOL close and DVOL close each >= its strictly-prior 720h q75, each with "
                "672 observations"
            ),
            "price": (
                "60 exact BTCUSDT 1m bars in [T-1h,T); nonzero absolute return <= its "
                "strictly-prior 720h q40 with 672 observations"
            ),
            "oi": (
                "raw-time backward-asof observations at T and T-60m, each age<=5m; positive "
                "change >= strictly-prior 720h q75 with 672 observations"
            ),
            "funding": (
                "latest nonzero event at or before T; absolute rate <= strictly-prior 270-event "
                "q50 with 252 observations"
            ),
            "trigger": "false-to-true onset, prior hour source-valid and consecutive",
            "entry": "exact BTCUSDT T+5m open, all features available by T",
            "side": "sign of completed-hour return",
            "hold": "6 elapsed hours",
            "reservation": "global half-open, exit first on equal open",
            "no_imputation": True,
        },
        "policy": {
            "prior_hours": 720, "prior_min_hours": 672,
            "volatility_level_quantile": 0.75, "quiet_return_quantile": 0.40,
            "oi_change_quantile": 0.75, "funding_events": 270,
            "funding_min_events": 252, "funding_abs_quantile": 0.50,
            "oi_asof_max_age_minutes": 5, "entry_delay_minutes": 5,
            "hold_hours": 6, "leverage": 0.5,
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
            "minority_side_share_min": 0.20, "max_month_share": 0.45,
        },
        "novelty_gates": {
            "exact_entry_jaccard_max": 0.10, "candidate_near_6h_share_max": 0.45,
            "occupied_5m_jaccard_max": 0.30,
            "absolute_signed_exposure_pearson_max": 0.35,
            "must_pass_before_economics": True,
        },
        "economic_gates": {
            "absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0,
            "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0,
            "weekly_signflip_one_sided_p_max": 0.10,
            "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5,
            "each_calendar_half_positive": True, "stop_on_first_failure": True,
            "future_can_rank_repair_or_reselect": False,
            "accounting": (
                "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional "
                "side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "source_plan": {
            "vol_oi_funding": "reuse hash-bound OCDR-12C nonprice snapshot",
            "completed_hour_price": "reuse hash-bound OICER completed-hour feature snapshot",
            "execution_price": "sealed until source-support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_candidate_incidence_known": True,
            "prior_candidate_outcomes_used_for_olqpb": False,
            "olqpb_candidate_incidence_opened": False,
            "olqpb_post_entry_return_or_pnl_opened": False,
            "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False,
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n")
    print(args.output)
