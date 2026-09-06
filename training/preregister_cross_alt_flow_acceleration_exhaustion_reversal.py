"""Outcome-blind preregistration for CAFAER-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT = Path("results/cross_alt_flow_acceleration_exhaustion_reversal_preregistration_2026-08-09.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "cross_alt_flow_acceleration_exhaustion_reversal_v1", "policy_id": "CAFAER-12",
        "as_of_date": "2026-08-09", "outcomes_opened": False, "source_incidence_opened": False, "singleton": True,
        "mechanism": {
            "claim": "When daily aggressive-flow imbalance accelerates in the same direction across at least four of six liquid alt perpetuals and BTC during a high-variation completed UTC day, broad one-sided urgency is exhausted; fade the common acceleration for twelve hours.",
            "side": "negative strict majority sign of six alt day-over-day taker-flow-imbalance changes",
            "why_distinct": "CAFLR-6 follows hourly alt-flow disagreement with BTC and failed only structural novelty. CAFAER-12 instead fades daily cross-alt and BTC flow acceleration consensus; it does not filter, retime, or subset CAFLR clocks. AFGI fits a 52-feature learner. CAFAER uses no price direction, funding, OI, premium, residual, or learned model.",
            "why_suited_to_volatile_regimes": "BTC completed-day realized-variation strict-prior rank must be at least 0.65",
            "why_low_gross9_overlap_is_plausible": "one daily 00:05 UTC acceleration-exhaustion clock is sparse and mechanism-distinct",
        },
        "features": {
            "universe": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT"],
            "day": "exact bars_binance 1m paths [D-24h,D); all 1,440 timestamps for all seven symbols",
            "flow": "sum(2*taker_buy_quote-quote_asset_volume)/sum(quote_asset_volume) per symbol and day; denominator strictly positive",
            "acceleration": "current valid daily flow imbalance minus immediately preceding valid calendar-day imbalance; gaps make the current day ineligible",
            "breadth": "at least four of six alt acceleration signs equal one strict sign",
            "confirmation": "BTC acceleration sign strict nonzero and equal to the qualifying alt majority",
            "btc_variation": "sqrt(sum squared BTC 1m log(close/open)) over completed day",
            "btc_variation_rank": "strict-prior midrank among at most 270 prior valid days; minimum 180; current excluded; rank>=0.65",
            "availability": "00:00 UTC after all exact daily paths complete", "no_imputation": True,
        },
        "clock": {"decision": "daily 00:00 UTC", "entry": "00:05 UTC BTCUSDT perpetual 5m open", "hold": "12 elapsed hours", "reservation": "global half-open; exit first on equal open", "split_crossing_action": "skip", "gross_exposure": 0.5, "funding": "exact settlements only after novelty passes"},
        "policy": {"history_observations": 270, "minimum_history_observations": 180, "variation_rank_min": 0.65, "breadth_min": 4, "entry_delay_minutes": 5, "hold_hours": 12, "leverage": 0.5, "base_cost_per_notional_side": 0.0006, "stress_cost_per_notional_side": 0.001},
        "stages": {"train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"], "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"], "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"], "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"]},
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.2, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.1, "candidate_near_6h_share_max": 0.35, "occupied_5m_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.1, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "future_can_rank_repair_or_reselect": False, "accounting": "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "source_plan": {"table": "bars_binance", "interval": "1m", "columns": ["ts", "open", "high", "low", "close", "quote_asset_volume", "taker_buy_quote"], "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT"], "read_only": True, "execution_price": "sealed until source support and Gross9 novelty pass"},
        "diagnostic_controls": {"names": ["no_volatility_gate", "three_of_six_breadth", "btc_acceleration_disagreement", "one_day_stale_acceleration", "direction_flip"], "diagnostic_controls_cannot_be_promoted": True},
        "research_boundary": {"database_metadata_only_opened_before_preregistration": True, "daily_flow_or_acceleration_values_used_to_select_rule": False, "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False, "selection_basis": "independent daily cross-alt flow-acceleration exhaustion mechanism plus user-required high volatility"},
        "stopping_rule": "terminal first-failure sequence: source support, Gross9 novelty, strict economics; no breadth, confirmation, day, side, hold, timing, volatility, or subset repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(result: dict[str, Any]) -> None:
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    if result.get("manifest_hash") != canonical_hash(core) or result.get("outcomes_opened") is not False or result.get("source_incidence_opened") is not False:
        raise RuntimeError("CAFAER preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(); validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
