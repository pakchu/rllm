"""Outcome-blind preregistration for HVTGAL-24."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVTGAL-24"
DEFAULT_OUTPUT = Path("results/high_volatility_tga_liquidity_relay_preregistration_2026-08-10.json")
SOURCE = Path("data/high_volatility_tga_liquidity_relay_sources_2020_2026/tga_closing_balance.csv.gz")
MARKET = Path("data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz")
MARKET_SHA = "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"
API = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/dts/operating_cash_balance"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_tga_liquidity_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-10",
        "singleton": True,
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "mechanism": {
            "claim": "A fall in the Treasury General Account releases public cash into the private banking system, while a rise absorbs it. In elevated BTC variation, trade the liquidity direction after a conservative publication delay.",
            "side": "negative sign of the strict five-observation TGA closing-balance change",
            "why_distinct": "The signal is an official fiscal cash-stock transmission variable, not a crypto clock, market-price continuation, FX factor, yield curve, claims release, or prior diagnostic control.",
            "volatile_market_target": "completed seven-day BTC variation causal rank at least 0.65",
            "why_low_gross9_overlap_is_plausible": "business-record dates shifted by a fixed conservative fiscal-publication lag create an external daily clock independent of Gross9 triggers",
        },
        "features": {
            "source": "official U.S. Treasury FiscalData Daily Treasury Statement Operating Cash Balance API",
            "row": "account_type exactly Treasury General Account (TGA) Closing Balance; value is finite positive open_today_bal",
            "tga_change": "current closing balance minus the fifth strictly previous valid official observation",
            "availability": "record date D plus five elapsed calendar days at 00:00 UTC; deliberately later than the stated following-business-day 4:00 p.m. publication even around a three-day weekend",
            "btc_variation": "sqrt(sum squared exact completed 5m BTC log returns over seven elapsed days ending at decision)",
            "variation_rank": "strict-prior midrank over all daily 00:00 UTC market states, at most 270 and minimum 180; current excluded; rank>=0.65",
            "no_imputation": True,
            "no_revision_backfill_assumption": "frozen current official accounting history; publication lag, endpoint response, and hashes retained",
        },
        "clock": {
            "decision": "record date D plus five elapsed calendar days at 00:00 UTC",
            "entry": "00:05 UTC exact BTCUSDT open",
            "hold": "24 elapsed hours",
            "side": "-sign(five-observation TGA change)",
            "reservation": "chronological half-open nonoverlap; exit first on equal open",
            "gross_exposure": 0.5,
            "funding": "exact only after novelty",
        },
        "policy": {"observation_lag": 5, "publication_lag_calendar_days": 5, "variation_history_days": 270, "minimum_history_days": 180, "variation_rank_min": 0.65, "entry_delay_minutes": 5, "hold_hours": 24, "leverage": 0.5, "base_cost_per_notional_side": 0.0006, "stress_cost_per_notional_side": 0.001},
        "stages": {"train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"], "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"], "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"], "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"]},
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.2, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.1, "candidate_near_6h_share_max": 0.35, "occupied_5m_bar_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.1, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "accounting": "fixed quantity, exact funding, 6bp/10bp per notional side, held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "post_stage_volatility_audit": {"prerequisite": "unchanged all-stage pass", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "diagnostic_controls": {"names": ["no_variation_gate", "one_observation_change", "ten_observation_change", "one_record_stale_change", "direction_flip", "same_clock_forced_long"], "cannot_be_promoted": True},
        "source_plan": {"endpoint": API, "filter": "record_date:gte:2020-01-01,record_date:lte:2026-07-27", "account_type": "Treasury General Account (TGA) Closing Balance", "destination": str(SOURCE), "download_after_preregistration_commit": True, "historical_market": {"path": str(MARKET), "sha256": MARKET_SHA}, "live_extension": "read-only Postgres BTCUSDT through 2026-08-01", "execution_prices": "sealed until source and novelty pass"},
        "research_boundary": {"official_endpoint_schema_and_one_example_date_opened": True, "candidate_source_panel_opened": False, "exact_candidate_incidence_or_outcomes_known": False, "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False, "selection_basis": "independent fiscal cash-stock transmission mechanism"},
        "stopping_rule": "terminal first failure; no source field, lag, change horizon, side, variation, clock, hold, subset, threshold, comparator, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(result: dict[str, Any]) -> None:
    if result["manifest_hash"] != canonical_hash({key: value for key, value in result.items() if key != "manifest_hash"}):
        raise RuntimeError("HVTGAL manifest drift")
    if SOURCE.exists():
        raise RuntimeError("HVTGAL source must not exist before preregistration")
    if hashlib.sha256(MARKET.read_bytes()).hexdigest() != MARKET_SHA:
        raise RuntimeError("HVTGAL market drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build()
    validate(report)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output)
