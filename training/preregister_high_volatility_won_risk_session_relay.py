"""Outcome-blind preregistration for HVWRSR-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVWRSR-8"
SLUG = "high_volatility_won_risk_session_relay"
DEFAULT_OUTPUT = Path(f"results/{SLUG}_preregistration_2026-08-13.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": f"{SLUG}_v1",
        "policy_id": POLICY_ID,
        "slug": SLUG,
        "as_of_date": "2026-08-13",
        "singleton": True,
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "mechanism": {
            "claim": (
                "A large completed USDKRW move during the Seoul cash session is a regional "
                "risk repricing: won weakness denotes risk aversion and won strength denotes "
                "risk appetite. During elevated BTC variation, BTC should relay in the same "
                "canonical risk direction for eight hours."
            ),
            "side": "negative strict sign of the completed USDKRW session return",
            "why_distinct": (
                "APWSR subtracts standardized USDHKD peg pressure from standardized USDKRW "
                "and holds twelve hours; Korean cash candidates use KRW-BTC venue paths. "
                "HVWRSR uses one predeclared fiat market's own unstandardized causal shock "
                "tail at the Seoul session close, with no second currency, relative spread, "
                "crypto direction, cash venue, flow, volume, funding, OI, fitted outcome, "
                "reused event set, or promoted control."
            ),
            "why_suited_to_volatile_regimes": (
                "causal trailing BTC 24-hour realized variation at the decision must occupy "
                "its upper 35 percent"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "one weekday 08:05 UTC external-FX shock clock is absent from Gross9 primitives"
            ),
        },
        "features": {
            "fx_session": (
                "exact observed bars_polygon USDKRW one-minute rows during weekday "
                "[00:00,08:00) UTC"
            ),
            "session_valid": (
                "at least 450 distinct minutes, first no later than 00:05, last no earlier "
                "than 07:55, finite positive coherent OHLC; no imputation"
            ),
            "session_return": "log(last close/first open), finite strict nonzero",
            "shock_rank": (
                "strict-prior midrank of absolute session_return over at most 90 prior valid "
                "sessions, minimum 60, current excluded; rank>=0.75"
            ),
            "btc_realized_variation": (
                "sqrt(sum squared exact BTC perpetual one-minute log(close/open) returns over "
                "[D-24h,D))"
            ),
            "variation_rank": (
                "strict-prior midrank over at most 90 prior source-valid weekday decisions, "
                "minimum 60, current excluded; rank>=0.65"
            ),
            "no_imputation": True,
        },
        "clock": {
            "decision": "exact weekday D 08:00 UTC after FX session and BTC variation complete",
            "entry": "exact BTCUSDT perpetual D 08:05 UTC open",
            "side": "negative USDKRW session-return sign",
            "hold": "8 elapsed hours",
            "reservation": "fixed weekday decisions; global chronological half-open, exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not signal input; exact settlements only after novelty",
        },
        "policy": {
            "session_start_utc": "00:00",
            "session_end_utc": "08:00",
            "minimum_session_minutes": 450,
            "prior_sessions": 90,
            "minimum_prior_sessions": 60,
            "shock_rank_min": 0.75,
            "variation_rank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 8,
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
            "minority_side_share_min": 0.2,
            "max_month_share": 0.45,
        },
        "novelty_gates": {
            "exact_entry_jaccard_max": 0.1,
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
            "weekly_signflip_one_sided_p_max": 0.1,
            "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5,
            "each_calendar_half_positive": True,
            "stop_on_first_failure": True,
            "accounting": (
                "fixed quantity, exact funding, 6bp/10bp per notional side, every held 5m "
                "favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "post_stage_volatility_audit": {
            "prerequisite": "unchanged all-stage pass",
            "rv20_q90_entry_filter": False,
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "identical_clock_forced_long_residual_positive": True,
        },
        "diagnostic_controls": {
            "names": [
                "no_variation_gate",
                "no_shock_tail",
                "one_session_stale_won",
                "direction_flip",
                "forced_long",
            ],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "fx": {
                "table": "bars_polygon",
                "symbol": "USDKRW",
                "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close"],
                "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            },
            "btc": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "completed_variation_only": True,
            },
            "read_after_preregistration_commit": True,
            "execution_prices": "sealed until source and novelty pass",
        },
        "research_boundary": {
            "prior_apwsr_and_korean_cash_results_known": True,
            "prior_mexican_peso_source_result_known": True,
            "repository_single_usdkrw_session_shock_candidate_found": False,
            "prior_event_sets_or_controls_reused": False,
            "prior_results_used_to_set_currency_session_threshold_side_or_hold": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent Seoul-session regional fiat-risk transmission mechanism",
        },
        "stopping_rule": (
            "terminal first failure; no currency, session, validity, history, shock rank, "
            "variation, side, clock, hold, subset, comparator, or control repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVWRSR preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build()
    validate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if args.output.exists() and args.output.read_bytes() != encoded:
        raise RuntimeError(f"refusing overwrite {args.output}")
    args.output.write_bytes(encoded)
    print(args.output)
