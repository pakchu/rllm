"""Outcome-blind preregistration for HVPIWR-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVPIWR-12"
DEFAULT_OUTPUT = Path("results/high_volatility_premium_index_wick_rejection_relay_preregistration_2026-08-09.json")
MARKET = Path("data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz")
MARKET_SHA = "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_premium_index_wick_rejection_relay_v1",
        "policy_id": POLICY_ID, "as_of_date": "2026-08-09", "singleton": True,
        "outcomes_opened": False, "source_incidence_opened": False, "gross9_rows_opened": False,
        "mechanism": {
            "claim": "During high BTC variation, a large one-day excursion in the Binance BTCUSDT premium index that is rejected back through the daily body marks failed leveraged demand. Follow the rejection direction for twelve hours: short after an upper excursion with a nonpositive body and long after a lower excursion with a nonnegative body.",
            "side": "-1 for dominant upper premium-index wick with nonpositive body; +1 for dominant lower wick with nonnegative body",
            "why_distinct": "POWR-12 compared five-minute perpetual-price and spot-price wicks and held one hour. HVPASR used unsigned daily premium total variation but took direction from BTC return. HVPIWR uses the premium index itself, its full-day endpoint-relative excursion geometry, no spot/perp price-wick comparison, and a twelve-hour rejection direction.",
            "why_suited_to_volatile_regimes": "both premium rejection magnitude and completed BTC realized variation must lie in causal upper tails",
            "why_low_gross9_overlap_is_plausible": "a daily 00:05 UTC derivatives-premium rejection clock is absent from Gross9 primitives",
        },
        "features": {
            "premium_source": "read-only Postgres bars_binance_premium BTCUSDT 1m",
            "source_day": "exact prior UTC day [D-24h,D) containing 1,440 unique coherent completed one-minute premium OHLC rows",
            "daily_open_close": "first one-minute open and last one-minute close",
            "upper_wick": "max(intraday high)-max(daily open,daily close)",
            "lower_wick": "min(daily open,daily close)-min(intraday low)",
            "body": "daily close-daily open; premium values may be negative or cross zero",
            "rejection": "upper_wick>lower_wick, upper_wick>0 and body<=0 maps short; lower_wick>upper_wick, lower_wick>0 and body>=0 maps long; ties and contradictions invalid",
            "magnitude": "dominant valid rejection wick",
            "magnitude_rank": "strict-prior midrank over at most 270 valid days, minimum 180, current excluded; rank>=0.70",
            "btc_variation": "sqrt(sum squared exact completed 5m BTC log returns over the same UTC day)",
            "variation_rank": "strict-prior midrank over at most 270 source-valid days, minimum 180, current excluded; rank>=0.65",
            "availability": "D 00:00 UTC after both complete daily paths", "no_imputation": True,
        },
        "clock": {"decision": "D 00:00 UTC", "entry": "D 00:05 UTC BTCUSDT open", "hold": "12 elapsed hours", "side": "premium rejection direction", "reservation": "daily opportunities nonoverlapping; exit first on equal open", "split_crossing_action": "skip", "gross_exposure": 0.5, "funding": "not a signal input; exact settlements after novelty"},
        "policy": {"premium_minutes": 1440, "btc_bars_5m": 288, "history_days": 270, "minimum_history_days": 180, "magnitude_rank_min": 0.70, "variation_rank_min": 0.65, "entry_delay_minutes": 5, "hold_hours": 12, "leverage": 0.5, "base_cost_per_notional_side": 0.0006, "stress_cost_per_notional_side": 0.001},
        "stages": {"train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"], "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"], "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"], "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"]},
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.2, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.1, "candidate_near_6h_share_max": 0.35, "occupied_5m_bar_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.1, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "accounting": "fixed quantity, exact funding, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "post_stage_volatility_audit": {"prerequisite": "unchanged candidate passes all stages", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "diagnostic_controls": {"definitions": {"no_variation_gate": "premium rejection and magnitude tail without BTC variation rank", "no_magnitude_tail": "premium rejection and high BTC variation without magnitude rank", "premium_body_direction": "daily premium body sign on primary clock", "one_day_stale_rejection": "prior exact day's rejection while current variation gates eligibility", "direction_flip": "negative primary side", "same_clock_forced_long": "side +1 on primary clock"}, "cannot_be_promoted": True},
        "source_plan": {"premium": {"table": "bars_binance_premium", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "high", "low", "close"], "read_only": True}, "historical_market": {"path": str(MARKET), "sha256": MARKET_SHA}, "live_extension": "read-only Postgres BTCUSDT 1m through 2026-08-01", "execution_prices": "sealed until source and novelty pass"},
        "research_boundary": {"prior_premium_and_wick_family_outcomes_known": True, "prior_event_sets_or_controls_reused": False, "exact_hvpiwr_incidence_or_outcomes_known": False, "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False, "selection_basis": "independent daily premium-index rejection geometry"},
        "stopping_rule": "terminal first failure; no source day, wick, body, rank, variation, side, clock, hold, subset, threshold, comparator, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVPIWR preregistration drift")
    if hashlib.sha256(MARKET.read_bytes()).hexdigest() != MARKET_SHA:
        raise RuntimeError("HVPIWR historical market drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    result = build(); validate(result); args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"); print(args.output)
