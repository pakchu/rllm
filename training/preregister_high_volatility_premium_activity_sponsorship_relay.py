"""Outcome-blind preregistration for HVPASR-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVPASR-12"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_premium_activity_sponsorship_relay_preregistration_2026-08-09.json"
)


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_premium_activity_sponsorship_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "An unusually active completed day in the Binance BTCUSDT premium index "
                "indicates sustained derivatives repricing rather than a quiet spot drift. "
                "When the same completed BTC day also has elevated realized variation, its "
                "direction should relay for twelve hours."
            ),
            "side": "strict sign of the completed prior UTC-day BTC return",
            "why_distinct": (
                "PVIAR used signed first-half premium displacement, same-sign second-half "
                "acceleration, joint BVOL/DVOL expansion, and a six-hour premium-direction "
                "trade. PPCSR used a persistent premium-sign zero crossing. HVPASR uses the "
                "unsigned total variation of all 1,440 premium closes over a full day, takes "
                "direction only from BTC, and uses neither implied-volatility feed nor premium "
                "polarity. It reuses no prior event set or control."
            ),
            "why_not_daily_price_momentum_repair": (
                "HVPASR has no candle close-location, range breakout, day-of-week, or scheduled "
                "trend rule; the external premium-activity upper tail is mandatory."
            ),
            "why_suited_to_volatile_regimes": (
                "both premium total variation and BTC realized variation must lie in causal "
                "upper tails"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "one daily premium-activity-conditioned 00:05 UTC clock is absent from Gross9"
            ),
        },
        "features": {
            "source_day": "exact prior UTC day [D-24h,D)",
            "premium_path": (
                "1,440 exact unique bars_binance_premium BTCUSDT 1m closes; all finite; "
                "premium may cross or equal zero"
            ),
            "premium_total_variation": (
                "sum(abs(close[i]-close[i-1])) for the 1,439 within-day adjacent close pairs; "
                "strict positive"
            ),
            "btc_path": (
                "288 exact unique completed BTCUSDT 5m OHLC bars covering the same UTC day; "
                "all finite, positive, and coherent"
            ),
            "btc_return": "log(last 5m close / first 5m open); strict nonzero",
            "btc_realized_variation": (
                "sqrt(sum of squared log(close/open)) over the 288 completed bars); strict positive"
            ),
            "causal_ranks": (
                "strict-prior midranks over at most 270 earlier source-valid days, minimum 180; "
                "current day excluded"
            ),
            "eligibility": (
                "premium-total-variation rank>=0.75 and BTC-realized-variation rank>=0.65"
            ),
            "availability": "D 00:00 UTC after both exact source paths complete",
            "no_imputation": True,
            "grid": False,
        },
        "clock": {
            "decision": "exact D 00:00 UTC after the prior UTC day is complete",
            "entry": "exact BTCUSDT D 00:05 UTC open",
            "side": "strict sign of prior-day BTC return",
            "hold": "12 elapsed hours",
            "reservation": "fixed daily opportunities are nonoverlapping; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_bvol_dvol_rv20": (
                "not signal inputs; exact funding only after novelty; RV20 q90 only after all "
                "economic stages pass"
            ),
        },
        "policy": {
            "premium_minutes": 1440,
            "btc_bars_5m": 288,
            "history_days": 270,
            "minimum_history_days": 180,
            "premium_activity_rank_min": 0.75,
            "btc_variation_rank_min": 0.65,
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
            "occupied_5m_bar_jaccard_max": 0.25,
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
            "accounting": (
                "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, "
                "every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "post_stage_volatility_audit": {
            "prerequisite": "unchanged candidate passes train, test, eval, final",
            "rv20_q90_entry_filter": False,
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "identical_clock_forced_long_residual_positive": True,
        },
        "diagnostic_controls": {
            "names": [
                "no_premium_activity_gate",
                "no_btc_variation_gate",
                "one_day_stale_features",
                "direction_flip",
                "same_clock_forced_long",
            ],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        "source_plan": {
            "premium": {
                "table": "bars_binance_premium",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "close"],
                "read_only": True,
            },
            "btc": (
                "hash-bound completed BTCUSDT 5m cache through 2026-06-01 plus read-only "
                "Postgres completed extension through 2026-08-01"
            ),
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_premium_family_outcomes_known": True,
            "premium_activity_values_used_to_select_rule": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent unsigned premium-activity sponsorship mechanism",
        },
        "stopping_rule": (
            "Terminal first-failure sequence: source support, Gross9 novelty, strict economics; "
            "no day, path, rank, side, hold, timing, subset, threshold, or control repair."
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVPASR preregistration hash mismatch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
