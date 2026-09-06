"""Outcome-blind preregistration for HVDXYFT-24."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("results/high_volatility_usd_index_forecast_threshold_relay_preregistration_2026-08-12.json")
UUP_YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/UUP"


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_usd_index_forecast_threshold_relay_v1",
        "policy_id": "HVDXYFT-24",
        "as_of_date": "2026-08-12",
        "gross9_rows_opened": False,
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "A causal one-day forecast from a bivariate US-dollar-index/BTC return VAR carries global "
                "risk and dollar-liquidity information that Bitcoin incorporates with delay. Trade only when the "
                "causal forecast magnitude is in its upper 40% and completed BTC variation is high."
            ),
            "side": "strict sign of the current causal one-step BTC forecast",
            "external_support": {
                "paper": "Arain and Snudden (2026), When Are Statistical Forecast Gains Economically Relevant? Evidence From Bitcoin Returns",
                "doi": "10.1002/for.70077",
                "paper_fixed_facts": [
                    "daily bivariate VARs generate real-time out-of-sample Bitcoin forecasts",
                    "the USD index has 0.585 directional accuracy in extreme Bitcoin moves",
                    "the USD index delivers statistically significant excess profits and among the highest cumulative and risk-adjusted returns",
                ],
                "implementation_choices_not_claimed_as_replication": [
                    "Invesco DB US Dollar Index Bullish Fund symbol UUP as the tradable USD-index proxy",
                    "VAR(1) with intercept estimated on a trailing 252 common-session window",
                    "a strict-prior q60 absolute-forecast threshold and 24-hour BTC hold from ten minutes after the US cash close",
                    "causal BTC realized-variation rank gate of 0.65",
                ],
            },
            "why_distinct": (
                "Repository collision scans found no UUP/BTC bivariate VAR or forecast-magnitude-threshold candidate. "
                "It is not the prior Shanghai VAR, dollar-factor, credit ETF ratio, Treasury-flow, or Gross9 mechanism."
            ),
            "why_suited_to_volatile_regimes": (
                "The paper reports USD directional accuracy specifically in extreme Bitcoin moves; the frozen implementation "
                "requires causal BTC variation rank >= 0.65 and forecast-magnitude rank >= 0.60."
            ),
            "why_low_gross9_overlap_is_plausible": (
                "Sparse high-confidence USD forecasts at the US cash close are absent from Gross9 primitives."
            ),
        },
        "features": {
            "uup_return": "log(split/dividend-adjusted UUP close_t / adjusted close_previous_source_session)",
            "btc_return": (
                "log(BTCUSDT official US cash-close anchor_t / anchor_previous_common_session)"
            ),
            "common_rows": "one paired row per exact official common UUP/NYSE session; missing/nonpositive values are ineligible; no imputation",
            "var": (
                "For each decision t, OLS-estimate a bivariate VAR(1) with intercept on exactly the immediately "
                "previous 252 finite paired rows, using rows (t-252..t-1) as dependent observations and their "
                "one-row lags; minimum 252 paired rows. Forecast BTC return one step from the current paired row."
            ),
            "forecast": "BTC equation intercept + lag-BTC coefficient*btc_return_t + lag-UUP coefficient*uup_return_t",
            "forecast_magnitude_rank": "strict-prior midrank of absolute forecast against at most 252 previous valid forecasts; minimum 126; current excluded; rank>=0.60",
            "btc_variation": (
                "sqrt(sum squared log(close/open)) over 1,440 exact BTCUSDT 1m bars ending at the official US cash close"
            ),
            "btc_variation_rank": (
                "strict-prior midrank against at most 252 previous valid UUP-session BTC variations; "
                "minimum 126; current excluded; rank >= 0.65"
            ),
            "forecast_magnitude_gate": True,
        },
        "clock": {
            "source_session": "Yahoo UUP adjusted row on each validated official US cash session, including frozen early closes",
            "feature_available": "official cash close plus 5 elapsed minutes",
            "entry": "exact BTCUSDT five-minute open 5 minutes after feature availability",
            "hold": "24 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium": "not signal inputs; exact funding only after novelty passes",
            "no_imputation": True,
        },
        "policy": {
            "var_lags": 1,
            "var_trailing_rows": 252,
            "var_minimum_rows": 252,
            "variation_prior_sessions": 252,
            "variation_prior_minimum": 126,
            "variation_midrank_min": 0.65,
            "forecast_magnitude_prior": 252,
            "forecast_magnitude_minimum": 126,
            "forecast_magnitude_midrank_min": 0.60,
            "feature_delay_minutes": 5,
            "entry_delay_minutes_after_feature": 5,
            "hold_hours": 24,
            "gross_exposure": 0.5,
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
                "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, every held "
                "5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "source_plan": {
            "uup": {
                "url": UUP_YAHOO_URL,
                "symbol": "UUP",
                "fields": ["timestamp", "open", "high", "low", "close", "volume", "adjusted_close"],
                "adjusted_close_required": True,
                "download_after_preregistration": True,
            },
            "btc_1m": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "close"],
                "read_only": True,
            },
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        "diagnostic_controls": {
            "names": ["no_btc_volatility_gate", "direction_flip", "one_session_stale_forecast", "uup_sign_only", "btc_ar_only", "same_clock_forced_long"],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        "post_stage_volatility_audit": {
            "rv20_q90_entry_filter": False,
            "opened_only_after_all_train_test_eval_final_economic_gates_pass": True,
            "cannot_repair_or_promote": True,
        },
        "research_boundary": {
            "source_schema_and_transport_checked": False,
            "source_values_used_to_select_rule": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "repository_uup_btc_var_threshold_candidate_found": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "published USD extreme-move directional accuracy and significant excess profitability plus exact repository absence",
        },
        "stopping_rule": (
            "terminal first-failure sequence: source support, Gross9 novelty, train/test/eval/final strict economics, "
            "then RV20 q90 audit; no predictor, lag, window, threshold, side, hold, clock, subset, model, or control repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVDXYFT preregistration hash drift")
    if value["outcomes_opened"] is not False or value["source_incidence_opened"] is not False:
        raise RuntimeError("HVDXYFT boundary drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
