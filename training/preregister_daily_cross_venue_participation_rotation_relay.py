"""Outcome-blind preregistration for DCVPR-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "results/daily_cross_venue_participation_rotation_relay_preregistration_2026-08-08.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "daily_cross_venue_participation_rotation_relay_v1",
        "policy_id": "DCVPR-12",
        "as_of_date": "2026-08-08",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "At the fixed UTC day boundary, a large completed-day rotation in normalized "
                "Upbit BTC/KRW participation relative to Binance BTC/USDT participation during "
                "an unusually volatile BTC day identifies a geographic migration of price "
                "discovery; the participation-rotation direction should relay for twelve hours."
            ),
            "side": "sign of the completed-day change in the causal Upbit/Binance volume-ratio z-score",
            "why_distinct": (
                "DCVPR uses a first difference of cross-venue participation at one daily clock. "
                "The legacy Upbit sleeve used a low intraday ratio level and was long-only; DOPDR "
                "used OI-price opposition; crypto-volatility candidates used BVOL/DVOL rather than spot participation."
            ),
            "why_suited_to_volatile_regimes": "the signal requires completed-day realized variation in its causal outer regime",
            "why_low_gross9_overlap_is_plausible": "one 00:05 UTC daily cross-venue rotation entry is absent from Gross9",
        },
        "clock": {
            "decision": "exact 00:00 UTC D after 24 consecutive source-valid completed hours [D-24h,D)",
            "participation": (
                "five-minute Upbit BTC/KRW traded value divided by Binance BTC/USDT quote value "
                "converted to KRW with causal USDKRW; causal rolling z-score over 288 completed 5m bars"
            ),
            "rotation": "z-score at D-5m minus z-score at D-24h-5m, nonzero",
            "realized_variation": "sqrt(sum of squared completed 5m BTC log returns over [D-24h,D))",
            "causal_ranks": (
                "absolute rotation and realized variation are ranked against the strictly prior "
                "180 valid daily observations; minimum 126, current excluded"
            ),
            "gates": "absolute_rotation_rank>=0.65 and realized_variation_rank>=0.65",
            "entry": "exact BTCUSDT D+5m open",
            "side": "sign(rotation)",
            "hold": "12 elapsed hours",
            "reservation": "fixed daily decisions; global half-open, exit first on equal open",
            "funding_oi": "not signal inputs; exact funding only for later PnL",
            "no_imputation": True,
        },
        "policy": {
            "participation_z_window_bars_5m": 288,
            "rotation_lag_bars_5m": 288,
            "prior_days": 180,
            "prior_min_days": 126,
            "absolute_rotation_rank_min": 0.65,
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
            "participation": "current PostgreSQL-backed Binance, Upbit, and causal USDKRW feature stack through 2026-08-01",
            "completed_price": "same completed five-minute Binance frame, outcomes excluded",
            "execution_price": "sealed until source-support and Gross9 novelty pass",
        },
        "diagnostic_controls": {
            "names": [
                "no_volatility_gate",
                "no_rotation_tail",
                "absolute_ratio_level",
                "one_day_stale_rotation",
                "direction_flip",
            ],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        "research_boundary": {
            "legacy_upbit_level_candidate_incidence_known": True,
            "legacy_candidate_outcomes_used_to_define_dcvpr": False,
            "dcvpr_candidate_incidence_opened": False,
            "dcvpr_post_entry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent participation-rotation mechanism and sparse daily clock",
        },
        "stopping_rule": (
            "Terminal first-failure sequence: source support, Gross9 novelty, strict economics; "
            "no threshold, side, hold, clock, or subset repair."
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n")
    print(args.output)
