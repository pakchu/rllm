"""Outcome-blind preregistration for WHVOF-6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "results/weekend_high_volatility_overreaction_fade_preregistration_2026-08-08.json"
)


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "weekend_high_volatility_overreaction_fade_v1",
        "policy_id": "WHVOF-6",
        "as_of_date": "2026-08-08",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "mechanism": {
            "claim": (
                "At fixed Saturday/Sunday UTC half-day boundaries, an unusually large completed "
                "twelve-hour BTC move while both BTC BVOL and DVOL are elevated is more likely to "
                "reflect thin-weekend overreaction than durable institutional price discovery; "
                "the move should partially reverse over the next six hours."
            ),
            "side": "opposite sign of the completed twelve-hour BTC return",
            "why_distinct": (
                "WHVOF is a weekend-liquidity overreaction clock with fixed half-day blocks and "
                "dual implied-volatility levels. HVCFF used cross-venue volatility-level fracture "
                "at every eight-hour boundary; high-volatility trend candidates traded continuation; "
                "funding and premium candidates used derivatives-flow events rather than weekends."
            ),
            "why_low_gross9_overlap_is_plausible": (
                "Gross9 has no weekend-only fixed half-day volatility-overreaction clock"
            ),
        },
        "clock": {
            "decision": "exact UTC 00:00 or 12:00 boundary D falling on Saturday or Sunday",
            "price_block": "twelve consecutive completed valid BTCUSDT hours [D-12h,D)",
            "price_tail": (
                "nonzero abs return from first hour open to final hour close >= strictly-prior "
                "540 half-day-block q60 with 504 valid observations; current block excluded"
            ),
            "volatility_regime": (
                "BVOL close and DVOL close at D are each >= own strictly-prior 720-hour q60 "
                "with 672 valid observations"
            ),
            "entry": "exact BTCUSDT D+5m open",
            "side": "negative sign of completed twelve-hour return",
            "hold": "6 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "funding_oi_premium_forbidden_as_signal_inputs": True,
            "no_imputation": True,
        },
        "policy": {
            "block_hours": 12,
            "return_prior_blocks": 540,
            "return_min_blocks": 504,
            "absolute_return_quantile": 0.60,
            "volatility_prior_hours": 720,
            "volatility_min_hours": 672,
            "volatility_level_quantile": 0.60,
            "eligible_weekdays_utc": ["Saturday", "Sunday"],
            "eligible_boundary_hours_utc": [0, 12],
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
        "source_support_gates": {
            "minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8},
            "minority_side_share_min": 0.20,
            "max_month_share": 0.45,
        },
        "novelty_gates": {
            "exact_entry_jaccard_max": 0.10,
            "candidate_near_6h_share_max": 0.35,
            "occupied_5m_jaccard_max": 0.25,
            "absolute_signed_exposure_pearson_max": 0.35,
            "must_pass_before_economics": True,
        },
        "economic_gates": {
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_min": 3.0,
            "strict_mdd_max_pct": 15.0,
            "mean_gross_underlying_min_bp": 20.0,
            "weekly_signflip_one_sided_p_max": 0.10,
            "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5,
            "each_calendar_half_positive": True,
            "stop_on_first_failure": True,
            "future_can_rank_repair_or_reselect": False,
            "accounting": (
                "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, "
                "every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "source_plan": {
            "volatility": "reuse hash-bound OCDR-12C BVOL/DVOL snapshot through 2026-08-01",
            "completed_price": "reuse hash-bound OLIAH completed-hour snapshot through 2026-08-01",
            "execution_price": "sealed until source-support and Gross9 novelty pass",
        },
        "diagnostic_controls": {
            "names": [
                "no_weekend_gate", "no_high_volatility", "no_return_tail",
                "one_block_stale_volatility", "direction_flip",
            ],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        "research_boundary": {
            "prior_price_and_volatility_source_rows_known": True,
            "prior_candidate_outcomes_used_to_define_whvof": False,
            "whvof_candidate_incidence_opened": False,
            "whvof_post_entry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
        },
        "stopping_rule": "terminal first failure; no repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n")
    print(args.output)
