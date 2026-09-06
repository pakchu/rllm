"""Outcome-blind preregistration for HVMTI-48."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVMTI-48"
DEFAULT_OUTPUT = Path("results/high_volatility_multi_horizon_trend_ignition_relay_preregistration_2026-08-13.json")


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_multi_horizon_trend_ignition_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-13",
        "singleton": True,
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "public_source_basis": {
            "realized_variation": "https://www.nber.org/papers/w8160",
            "claim_used": "sums of high-frequency squared returns measure completed realized variation without supplying future direction.",
            "perpetual_mechanics": "https://arxiv.org/abs/2212.06888",
            "claim_used_2": "open interest measures outstanding leveraged inventory and is used only through observations available by decision time.",
            "implementation_is_unpublished_adaptation": True,
        },
        "mechanism": {
            "claim": "When completed daily BTC realized variation crosses from below into its causal upper regime while completed five-day and twenty-day returns share one strict direction, volatility ignition is aligned with weekly and monthly risk transfer. Follow the common trend for forty-eight elapsed hours.",
            "side": "common strict sign of completed five-day and twenty-day BTC log returns",
            "why_distinct": "HVWMR is a fixed Wednesday single seven-day momentum state; STCR is a fixed Monday/Thursday 3d/14d calendar state without a volatility gate. HVMTI uses daily opportunities, standard 5d/20d trend consensus, and only a fresh below-to-above daily realized-variation onset. It reuses no prior event set or control.",
            "why_suited_to_volatile_regimes": "completed daily variation rank must cross from below to at least 0.65",
            "why_low_gross9_overlap_is_plausible": "daily volatility-onset trend-consensus events with 48-hour reservation are absent from Gross9 definitions",
        },
        "features": {
            "decision_grid": "every exact 00:00 UTC boundary D",
            "daily_path": "288 exact coherent BTCUSDT five-minute bars aggregated from 1440 unique one-minute rows in [D-24h,D)",
            "daily_realized_variation": "sqrt(sum of 288 squared five-minute intrabar log(close/open) returns), finite strict positive",
            "daily_close_history": "exact completed UTC daily closes through D-5m, no imputation",
            "return_5d": "log(C[D-1d]/C[D-6d]), finite strict nonzero",
            "return_20d": "log(C[D-1d]/C[D-21d]), finite strict nonzero",
            "trend_consensus": "strict signs of return_5d and return_20d are identical",
            "variation_rank": "strict-prior midrank over at most 90 earlier source-valid daily decisions, minimum 60, current excluded",
            "ignition": "current variation rank>=0.65 and immediately prior exact source-valid daily rank<0.65; missing prior cannot trigger",
            "availability": "D after every completed source bar is available",
            "no_imputation": True,
            "grid": False,
        },
        "clock": {
            "decision": "exact daily 00:00 UTC boundary D",
            "entry": "exact BTCUSDT D+5m open",
            "side": "common completed 5d/20d return sign",
            "hold": "48 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact held settlements only after novelty passes",
            "rv20": "not a signal input; q90 audit only after unchanged all-stage economics pass",
        },
        "policy": {"daily_bars": 288, "short_trend_days": 5, "long_trend_days": 20, "history_days": 90, "minimum_history_days": 60, "variation_rank_min": 0.65, "entry_delay_minutes": 5, "hold_hours": 48, "leverage": 0.5, "base_cost_per_notional_side": 0.0006, "stress_cost_per_notional_side": 0.001},
        "stages": {"train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"], "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"], "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"], "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"]},
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.20, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.10, "candidate_near_6h_share_max": 0.35, "occupied_5m_bar_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.10, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "accounting": "fixed quantity, exact funding, 6bp/10bp per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "post_stage_volatility_audit": {"prerequisite": "unchanged all-stage pass", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "diagnostic_controls": {"names": ["no_variation_onset", "no_trend_consensus", "one_day_stale_features", "direction_flip", "forced_long"], "cannot_be_promoted": True},
        "source_plan": {"bars": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "read_only": True}, "execution_prices": "sealed until source support and Gross9 novelty pass"},
        "research_boundary": {"prior_weekly_and_scheduled_trend_outcomes_known": True, "exact_candidate_incidence_or_outcomes_known": False, "prior_event_sets_or_controls_reused": False, "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False, "selection_basis": "independent multi-horizon trend consensus at daily realized-variation ignition"},
        "stopping_rule": "Terminal first failure: source, novelty, train, test, eval, final; no threshold, side, hold, clock, subset, source, or control repair.",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVMTI preregistration hash mismatch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
