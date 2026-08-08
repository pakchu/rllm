"""Outcome-blind preregistration for UJCVR-12."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT = Path("results/usdjpy_carry_volatility_relay_preregistration_2026-08-08.json")


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "usdjpy_carry_volatility_relay_v1",
        "policy_id": "UJCVR-12",
        "as_of_date": "2026-08-08",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": "An unusually large completed USDJPY move during the liquid 13:00-21:00 UTC weekday session measures a fresh yen-carry risk-on or risk-off impulse. When BTC has also experienced unusually high strictly pre-entry realized variation, the carry impulse should relay into BTC for twelve hours.",
            "side": "sign of the completed USDJPY session log return; yen weakening (USDJPY up) is BTC long and yen strengthening is BTC short",
            "why_distinct": "UJCVR uses one fixed liquid-session USDJPY shock. The prior hidden-safe-haven search used six-currency cross-sectional cancellation, a BTC non-absorption residual, and dense hourly clocks; the weekend FX reconciliation candidate used market-reopen gaps. Neither event set or outcome defines UJCVR.",
            "why_suited_to_volatile_regimes": "the completed pre-entry 24-hour BTC realized variation must be in its causal upper 35%",
            "why_low_gross9_overlap_is_plausible": "one 21:05 UTC weekday macro clock conditioned on a yen-carry tail is absent from Gross9",
        },
        "features": {
            "fx_session": "USDJPY bars_polygon interval=1m rows with 13:00<=UTC time<21:00 on one UTC weekday",
            "fx_session_valid": "at least 450 distinct observed minutes, first timestamp no later than 13:05, last timestamp no earlier than 20:55, finite positive OHLC; no imputation",
            "fx_return": "log(last observed close / first observed open) inside the completed session",
            "fx_return_z": "standardized against at most 90 strictly prior valid sessions, minimum 60, current excluded; prior sample std positive",
            "fx_gate": "abs(fx_return_z)>=1.0",
            "btc_realized_variation": "sqrt(sum of squared completed-hour BTC log returns over the 24 hours ending at decision time)",
            "volatility_rank": "strict-prior midrank against at most 90 valid decision days, minimum 60; current excluded; rank>=0.65",
            "no_imputation": True,
        },
        "clock": {
            "decision": "exact weekday D 21:00 UTC after the FX session and BTC variation are complete",
            "entry": "exact BTCUSDT D 21:05 UTC open",
            "side": "sign(fx_return)",
            "hold": "12 elapsed hours",
            "reservation": "fixed weekday decisions; global half-open, exit first on equal open",
            "funding_oi_premium_implied_vol": "not signal inputs; exact funding only for later PnL",
        },
        "policy": {
            "fx_prior_sessions": 90,
            "fx_prior_min_sessions": 60,
            "absolute_fx_return_z_min": 1.0,
            "realized_variation_rank_min": 0.65,
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
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.20, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.10, "candidate_near_6h_share_max": 0.35, "occupied_5m_bar_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.10, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "future_can_rank_repair_or_reselect": False, "accounting": "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "source_plan": {
            "fx": {"table": "bars_polygon", "symbol": "USDJPY", "interval": "1m", "columns": ["ts", "open", "high", "low", "close"], "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"], "materialize_after_preregistration": True},
            "completed_btc": "hash-bound completed-hour BTC source through 2026-08-01 for pre-entry variation only",
            "execution_price": "sealed until source-support and Gross9 novelty pass",
        },
        "diagnostic_controls": {"names": ["no_volatility_gate", "no_fx_tail", "one_session_stale_fx_shock", "direction_flip"], "diagnostic_controls_cannot_be_promoted": True},
        "research_boundary": {
            "prior_fx_family_outcomes_known": True,
            "prior_fx_event_sets_reused": False,
            "prior_fx_candidate_outcomes_used_to_set_ujcvr_direction_threshold_hold_or_clock": False,
            "ujcvr_candidate_incidence_opened": False,
            "ujcvr_post_entry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent yen-carry transmission mechanism for volatile BTC regimes",
        },
        "stopping_rule": "Terminal first-failure sequence: source support, Gross9 novelty, strict economics; no threshold, side, hold, clock, session, or subset repair.",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    args.output.write_text(json.dumps(build(), indent=2) + "\n"); print(args.output)
