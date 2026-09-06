"""Outcome-blind preregistration for IDRDR-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

POLICY_ID = "IDRDR-12"
DEFAULT_OUTPUT = Path("results/intraday_dollar_reversal_dominance_relay_preregistration_2026-08-09.json")


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "intraday_dollar_reversal_dominance_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": "When the cross-sectional G10 dollar factor reverses between the European-US overlap and the late US session, and the late factor fully dominates the early factor, the late global risk repricing should transmit inversely into BTC for twelve hours in an already volatile BTC regime.",
            "side": "opposite the strict sign of the late-session median standardized canonical-dollar factor",
            "why_distinct": "DFSR used one full-session continuous factor and its absolute rank; HVDBR used five-of-six full-session sign breadth. IDRDR uses the temporal sign reversal and magnitude ordering of two separately standardized sub-session factors, no full-session tail, breadth count, prior event set, or promoted control.",
            "why_suited_to_volatile_regimes": "completed pre-entry 24-hour BTC realized variation must rank in its causal upper 35%",
            "why_low_gross9_overlap_is_plausible": "one sparse 21:05 UTC weekday cross-session reversal clock is absent from Gross9 primitives",
        },
        "features": {
            "fx_source": "bars_polygon interval=1m for EURUSD, GBPUSD, USDAUD, USDCAD, USDCHF, and USDJPY on one UTC weekday",
            "early_session": "13:00<=UTC time<17:00; at least 225 distinct minutes, first<=13:05, last>=16:55",
            "late_session": "17:00<=UTC time<21:00; at least 225 distinct minutes, first<=17:05, last>=20:55",
            "source_validity": "finite positive coherent OHLC and both exact endpoint requirements for every pair and sub-session; no imputation",
            "canonical_dollar_returns": "negative log return for EURUSD and GBPUSD; positive log return for USDAUD, USDCAD, USDCHF, and USDJPY, separately by sub-session",
            "pair_standardization": "each pair and sub-session return standardized by its own at most 90 strictly prior valid sub-sessions, minimum 60, current excluded, positive sample std",
            "early_factor": "cross-sectional median of six finite early-session z-scores; strict nonzero",
            "late_factor": "cross-sectional median of six finite late-session z-scores; strict nonzero",
            "reversal": "early_factor and late_factor have opposite strict signs",
            "late_dominance": "absolute late_factor>=absolute early_factor",
            "btc_realized_variation": "sqrt(sum of squared completed-hour BTC log returns over the 24 hours ending at 21:00 UTC)",
            "volatility_rank": "strict-prior midrank over at most 90 valid decision days, minimum 60, current excluded; rank>=0.65",
            "no_imputation": True,
        },
        "clock": {
            "decision": "exact weekday D 21:00 UTC after both FX sub-sessions and BTC variation are complete",
            "entry": "exact BTCUSDT D 21:05 UTC open",
            "side": "opposite strict late dollar-factor sign",
            "hold": "12 elapsed hours",
            "reservation": "fixed weekday decisions; global half-open, exit first on equal open",
            "funding_oi_premium_implied_vol_rv20": "not signal inputs; exact funding only after novelty; RV20 q90 only after all economic stages pass",
        },
        "policy": {
            "fx_prior_sessions": 90,
            "fx_prior_min_sessions": 60,
            "subsession_min_minutes": 225,
            "late_to_early_absolute_factor_min": 1.0,
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
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.10, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "accounting": "fixed quantity, exact funding, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "post_stage_volatility_audit": {"prerequisite": "unchanged candidate passes train, test, eval, final", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "diagnostic_controls": {"names": ["no_volatility_gate", "no_reversal_gate", "no_late_dominance", "raw_return_factors", "one_session_stale_factor_geometry", "direction_flip", "forced_long"], "cannot_be_promoted": True},
        "source_plan": {"fx": {"table": "bars_polygon", "symbols": ["EURUSD", "GBPUSD", "USDAUD", "USDCAD", "USDCHF", "USDJPY"], "interval": "1m", "columns": ["ts", "open", "high", "low", "close"], "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"], "read_after_preregistration": True}, "completed_btc": "hash-bound completed-hour BTC source through 2026-08-01 for pre-entry variation only", "execution_prices": "sealed until source support and Gross9 novelty pass"},
        "research_boundary": {
            "prior_fx_family_outcomes_known": True,
            "prior_fx_event_sets_reused": False,
            "prior_fx_candidate_outcomes_used_to_set_idrdr_formula_direction_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent cross-session dollar-factor reversal and dominance transmission",
        },
        "stopping_rule": "Terminal first-failure sequence: source support, Gross9 novelty, strict economics; no session, factor, reversal, dominance, side, hold, clock, subset, or control repair.",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    result = build(); args.output.write_text(json.dumps(result, indent=2) + "\n"); print(args.output)
