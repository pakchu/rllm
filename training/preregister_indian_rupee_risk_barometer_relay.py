"""Outcome-blind preregistration for IRBR-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT = Path("results/indian_rupee_risk_barometer_relay_preregistration_2026-08-10.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode()).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "indian_rupee_risk_barometer_relay_v1",
        "policy_id": "IRBR-12",
        "as_of_date": "2026-08-10",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "An unusually large completed USD/INR move during the Indian cash session measures "
                "India-specific emerging-market funding stress or relief. In a high-variation BTC "
                "regime, rupee weakness maps short BTC and rupee strength maps long for twelve hours."
            ),
            "side": "negative strict sign of completed USDINR session return",
            "why_distinct": (
                "EMDFR used a three-currency standardized median during a later overlapping window. "
                "IRBR uses the India cash-session USDINR endpoint shock alone, requires its own "
                "causal magnitude tail, and reuses no EMDFR event, threshold, control, or outcome."
            ),
            "why_suited_to_volatile_regimes": (
                "BTC prior-24h realized-variation strict-prior rank must be at least 0.65"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "weekday 10:05 UTC entries driven by an India-session FX tail are absent from Gross9"
            ),
        },
        "features": {
            "fx_session": (
                "bars_polygon USDINR 1m rows [03:45,10:00) UTC on each Monday-Friday"
            ),
            "session_valid": (
                "at least 270 distinct minutes, first timestamp<=04:00, last timestamp>=09:45, "
                "finite positive coherent OHLC; no imputation"
            ),
            "fx_return": "log(last observed close/first observed open); strict nonzero",
            "absolute_fx_return_rank": (
                "strict-prior midrank of abs(fx_return) over at most 252 prior valid sessions, "
                "minimum 126, current excluded; rank>=0.70"
            ),
            "btc_variation": (
                "sqrt(sum squared log(close/open)) over exact 1440 BTCUSDT 1m rows in the prior "
                "24 elapsed hours ending at 10:00 UTC"
            ),
            "btc_variation_rank": (
                "strict-prior midrank over at most 252 prior valid weekday decisions, minimum 126, "
                "current excluded; rank>=0.65"
            ),
            "missing_duplicate_or_nonpositive": "ineligible or source failure; no imputation",
        },
        "clock": {
            "decision": "each Monday-Friday 10:00 UTC after complete FX and BTC source paths",
            "entry": "exact BTCUSDT 10:05 UTC 5m open",
            "side": "-sign(fx_return)",
            "hold": "12 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium": "not signal inputs; exact funding only after novelty passes",
            "no_imputation": True,
        },
        "policy": {
            "fx_rank_history_sessions": 252,
            "fx_rank_minimum_sessions": 126,
            "absolute_fx_return_rank_min": 0.70,
            "variation_rank_history_sessions": 252,
            "variation_rank_minimum_sessions": 126,
            "variation_rank_min": 0.65,
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
            "future_can_rank_repair_or_reselect": False,
            "accounting": (
                "fixed quantity, exact funding, 6bp base and 10bp stress per notional side, every "
                "held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "post_stage_volatility_audit": {
            "prerequisite": "unchanged candidate passes train, test, eval, final",
            "rv20_q90_entry_filter": False,
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "identical_clock_forced_long_residual_positive": True,
        },
        "source_plan": {
            "fx": {"table": "bars_polygon", "symbol": "USDINR", "interval": "1m",
                   "columns": ["ts", "open", "high", "low", "close"],
                   "window": ["2022-01-01T00:00:00Z", "2026-08-01T00:00:00Z"], "read_only": True},
            "btc": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m",
                    "columns": ["ts", "open", "close"], "read_only": True},
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        "diagnostic_controls": {
            "names": ["no_fx_tail", "no_volatility_gate", "one_session_stale_fx", "direction_flip", "forced_long"],
            "cannot_be_promoted": True,
        },
        "research_boundary": {
            "database_metadata_only_opened_before_preregistration": True,
            "prior_emerging_fx_family_outcomes_known": True,
            "prior_emerging_fx_event_sets_reused": False,
            "usdinr_values_used_to_select_rule": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent India cash-session risk channel plus high BTC variation",
        },
        "stopping_rule": (
            "terminal first failure: source support, Gross9 novelty, strict economics, then RV20 audit; "
            "no source, session, validity, rank, side, hold, clock, volatility, subset, or control repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("IRBR preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)
