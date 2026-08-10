"""Outcome-blind preregistration for HVVBTR-6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVVBTR-6"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_volume_bucket_toxicity_reversal_preregistration_2026-08-10.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_volume_bucket_toxicity_reversal_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-10",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "A sequence of equal-target quote-volume buckets with extreme absolute aggressive-flow "
                "imbalance measures order-flow toxicity in transaction time rather than wall time. "
                "During already volatile BTC trading, three same-direction toxic buckets represent "
                "one-sided liquidity consumption vulnerable to short-horizon inventory reversal."
            ),
            "side": "opposite the common strict sign of the final three completed bucket imbalances",
            "why_distinct": (
                "HVVBTR constructs a new endogenous clock whenever cumulative quote volume reaches a "
                "causally frozen target. Prior VPIN scans used fixed-bar rolling approximations; intrinsic-"
                "volume candidates used price lag or aggregate flow, and Gross9 has no volume-bucket clock."
            ),
            "volatile_market_target": "strict-prior prior-24h realized-variation rank >=0.65",
            "why_low_gross9_overlap_is_plausible": (
                "bucket completions and toxicity onsets occur at irregular minute-dependent times before "
                "conservative next-five-minute entry alignment"
            ),
        },
        "bucket_contract": {
            "source": "BTCUSDT bars_binance interval=1m",
            "source_bar_availability": "bar timestamp ts is available only at ts+1m",
            "first_bucket_start": "2023-04-02T00:00:00Z after one exact prior source day",
            "target_at_bucket_start": (
                "sum quote_asset_volume over the exact 1,440 completed source bars [start-24h,start) "
                "divided by 24; finite and strict positive; target remains fixed through the bucket"
            ),
            "completion": (
                "append whole consecutive one-minute bars from start until cumulative quote volume first "
                "reaches or exceeds target; no minute splitting, interpolation, or overshoot correction"
            ),
            "maximum_duration": "360 consecutive minutes; failure to complete makes the bucket invalid and terminally ends construction",
            "next_bucket_start": "the minute immediately after the completed bucket's final source bar",
            "bucket_quote_volume": "sum quote_asset_volume over all constituent bars",
            "bucket_signed_flow": "sum(2*taker_buy_quote-quote_asset_volume)",
            "bucket_imbalance": "bucket_signed_flow/bucket_quote_volume, finite and strict nonzero",
            "source_valid": (
                "every required timestamp unique and consecutive; finite positive coherent OHLC; finite "
                "nonnegative quote and taker-buy quote; taker-buy quote<=quote volume; no imputation"
            ),
        },
        "features": {
            "toxicity": "mean absolute bucket_imbalance over the current and previous 23 completed valid buckets",
            "toxicity_rank": (
                "strict-prior midrank over at most 720 prior valid bucket completions, minimum 480, "
                "current excluded; rank>=0.85"
            ),
            "direction_consensus": "final three completed bucket imbalances have one common strict sign",
            "btc_variation": (
                "sum squared log(close/open) over exact 1,440 completed one-minute bars ending at current "
                "bucket feature_available_time, strict positive"
            ),
            "variation_rank": "same strict-prior 720/480 valid-bucket rule; rank>=0.65",
            "eligible_state": "toxicity and variation ranks pass and direction_consensus holds",
            "onset": "eligible now and immediately preceding completed valid bucket ineligible",
        },
        "clock": {
            "feature_available_time": "final constituent source-bar ts+1m",
            "decision": "feature_available_time rounded upward to the first exact five-minute boundary",
            "entry": "exact BTCUSDT perpetual decision+5m open",
            "side": "negative common final-three-bucket imbalance sign",
            "hold": "6 elapsed hours",
            "reservation": "global half-open; exit first on equal-time entry",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not an input; exact realized funding only after novelty passes",
        },
        "policy": {
            "target_prior_minutes": 1440,
            "target_buckets_per_day": 24,
            "maximum_bucket_minutes": 360,
            "toxicity_buckets": 24,
            "direction_buckets": 3,
            "history_buckets": 720,
            "minimum_history_buckets": 480,
            "toxicity_rank_min": 0.85,
            "variation_rank_min": 0.65,
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
                "fixed quantity, exact funding, 6bp base and 10bp stress per notional side, "
                "every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "post_stage_volatility_audit": {
            "prerequisite": "unchanged candidate passes train, test, eval, and final",
            "rv20_q90_entry_filter": False,
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "identical_clock_forced_long_residual_positive": True,
        },
        "diagnostic_controls": {
            "names": [
                "no_variation_gate", "no_toxicity_tail", "single_last_bucket_direction",
                "one_bucket_stale_features", "direction_flip", "forced_long",
            ],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "bars": {
                "table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close", "quote_asset_volume", "taker_buy_quote"],
                "window": ["2023-04-01T00:00:00Z", "2026-08-01T00:00:00Z"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_fixed_bar_vpin_scan_outcomes_known": True,
            "prior_exact_volume_bucket_candidate_found": False,
            "prior_event_sets_reused": False,
            "prior_outcomes_used_to_set_bucket_target_ranks_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "canonical volume-synchronized toxicity exhaustion on an endogenous clock",
        },
        "stopping_rule": (
            "terminal first failure; no bucket target, duration, toxicity window, direction count, rank, "
            "history, onset, side, hold, clock, subset, control, or gate repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVVBTR preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)
