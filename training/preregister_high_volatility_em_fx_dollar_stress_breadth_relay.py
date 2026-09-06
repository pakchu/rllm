"""Outcome-blind preregistration for HVDBR-12."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT = Path("results/high_volatility_em_fx_dollar_stress_breadth_relay_preregistration_2026-08-13.json")


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_em_fx_dollar_stress_breadth_relay_v1",
        "policy_id": "HVEMFX-12",
        "as_of_date": "2026-08-13",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "singleton": True,
        "gross9_rows_opened": False,
        "mechanism": {
            "claim": "A broad same-direction standardized US-dollar move across at least three of four emerging-market currency pairs identifies a global dollar-liquidity shock rather than one country's news. During elevated BTC variation, BTC relays opposite the common USDMXN, USDKRW, USDINR and USDCNY dollar direction for twelve hours.",
            "side": "opposite the common strict standardized return direction across USDMXN, USDKRW, USDINR and USDCNY",
            "why_distinct": "Major-FX dollar breadth uses six developed-market pairs in a common 13:00-21:00 liquid window. Single-country candidates use MXN, KRW, INR or CNY alone. HVEMFX aggregates four heterogeneous emerging-market local sessions completed by 22:30 UTC, standardizes each against its own causal history, and requires three-of-four breadth. It reuses no prior event set or control and uses no BTC return direction, funding, OI, premium, crypto flow, fitted outcome, or post-entry data.",
            "why_suited_to_volatile_regimes": "completed pre-entry twenty-four-hour BTC realized variation must occupy its causal upper 35 percent",
            "why_low_gross9_overlap_is_plausible": "one 22:35 UTC weekday EM-FX breadth clock is absent from Gross9 structural primitives",
        },
        "features": {
            "fx_session": "bars_polygon interval=1m observed rows for USDMXN, USDKRW, USDINR and USDCNY within each UTC weekday [00:00,22:30)",
            "fx_session_valid": "at least 240 distinct observed minutes, first timestamp no later than 05:00 UTC, last timestamp no earlier than 09:30 UTC, finite positive coherent OHLC; no imputation",
            "canonical_dollar_returns": "log(last observed session close/first observed session open); all four symbols are USDXXX so positive means dollar strength and EM-currency weakness",
            "pair_zscores": "each canonical dollar return standardized against its own at most 90 strictly prior valid sessions, minimum 60, current excluded; prior sample standard deviation positive",
            "breadth_gate": "at least three of four pair z-scores have one common strict nonzero sign and strictly outnumber the opposite sign",
            "shock_gate": "median absolute pair z-score>=0.90",
            "btc_realized_variation": "sqrt(sum squared completed-hour BTC log returns over the 24 hours ending at 22:30 UTC)",
            "variation_rank": "strict-prior midrank against at most 90 source-valid decision days, minimum 60, current excluded; rank>=0.65",
            "no_imputation": True,
        },
        "clock": {
            "decision": "exact weekday D 22:30 UTC after all four native FX sessions and BTC variation are available",
            "entry": "exact BTCUSDT D 22:35 UTC open",
            "side": "opposite common standardized EM-FX dollar direction",
            "hold": "12 elapsed hours",
            "reservation": "fixed weekday decisions; global half-open, exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium": "not signal inputs; exact funding only after novelty",
        },
        "policy": {"fx_prior_sessions": 90, "fx_prior_min_sessions": 60, "minimum_agreeing_pairs": 3, "median_absolute_pair_z_min": 0.90, "realized_variation_prior_sessions": 90, "realized_variation_min_sessions": 60, "realized_variation_rank_min": 0.65, "decision_hour_utc": 22, "decision_minute": 30, "entry_delay_minutes": 5, "hold_hours": 12, "leverage": 0.5, "base_cost_per_notional_side": 0.0006, "stress_cost_per_notional_side": 0.001},
        "stages": {
            "train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
        },
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.20, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.10, "candidate_near_6h_share_max": 0.35, "occupied_5m_bar_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.10, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "future_can_rank_repair_or_reselect": False, "accounting": "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "source_plan": {"fx": {"table": "bars_polygon", "symbols": ["USDMXN", "USDKRW", "USDINR", "USDCNY"], "interval": "1m", "columns": ["ts", "open", "high", "low", "close"], "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"], "read_after_preregistration_commit": True}, "completed_btc": "hash-bound completed-hour BTC source through 2026-08-01 for pre-entry variation only", "execution_price": "sealed until source support and Gross9 novelty pass"},
        "diagnostic_controls": {"names": ["no_volatility_gate", "no_em_fx_shock_gate", "two_pair_breadth", "one_session_stale_stress", "direction_flip", "forced_long"], "cannot_be_promoted": True},
        "research_boundary": {"prior_major_fx_and_single_em_currency_outcomes_known": True, "repository_exact_four_pair_em_fx_breadth_event_found": False, "prior_event_sets_or_controls_reused": False, "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False, "selection_basis": "independent emerging-market dollar-liquidity breadth mechanism selected from source-metadata-only audit"},
        "stopping_rule": "terminal first failure; no universe, session, validity, standardization, breadth, shock, variation, side, clock, hold, subset, comparator, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    args.output.write_text(json.dumps(build(), indent=2) + "\n"); print(args.output)
