"""Outcome-blind preregistration for KCVSR-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "KCVSR-12"
DEFAULT_OUTPUT = Path("results/korean_cash_volume_sponsorship_relay_preregistration_2026-08-09.json")


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "korean_cash_volume_sponsorship_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "An unusually active completed Korean cash session supplies independent spot "
                "participation to a high-variation BTC move. When Upbit KRW-BTC base turnover is "
                "in its causal upper tail, the completed Binance BTC direction should relay for "
                "twelve hours rather than represent derivatives-only noise."
            ),
            "side": "strict sign of the completed Binance BTCUSDT 00:00-08:00 UTC return",
            "why_distinct": (
                "KCLR compared unadjusted venue return magnitudes and required price-sign agreement; "
                "KPAR measured FX-adjusted kimchi-premium acceleration. KCVSR uses neither venue-price "
                "leadership nor kimchi premium: Upbit contributes only causally ranked base-volume "
                "participation while Binance supplies direction and realized variation."
            ),
            "why_suited_to_volatile_regimes": "Binance session realized variation must rank at least 0.65 causally",
            "why_low_gross9_overlap_is_plausible": "one regional cash-volume-conditioned 08:05 UTC clock is absent from Gross9 primitives",
        },
        "features": {
            "session": "exact aligned bars_upbit KRW-BTC and bars_binance BTCUSDT 1m paths [00:00,08:00) UTC; all 480 timestamps required",
            "upbit_base_volume": "sum of 480 finite nonnegative KRW-BTC base-volume observations; strict positive",
            "binance_return": "log(07:59 close/00:00 open); strict nonzero",
            "binance_realized_variation": "sqrt(sum of squared log(1m close/open)) over all 480 Binance bars; strict positive",
            "causal_ranks": "strict-prior midranks over at most 270 earlier source-valid daily sessions, minimum 180; current excluded",
            "eligibility": "Upbit base-volume rank>=0.75 and Binance realized-variation rank>=0.65",
            "availability": "08:00 UTC after both exact source paths complete",
            "missing_duplicate_nonfinite_or_nonpositive": "ineligible or source failure; no imputation",
            "grid": False,
        },
        "clock": {
            "decision": "each calendar day 08:00 UTC after both source sessions complete",
            "entry": "exact 08:05 UTC BTCUSDT perpetual open",
            "side": "strict sign of completed Binance session return",
            "hold": "12 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium_implied_vol_rv20": "not signal inputs; exact funding only after novelty; RV20 q90 only after all economic stages pass",
        },
        "policy": {
            "session_minutes": 480,
            "history_observations": 270,
            "minimum_history_observations": 180,
            "upbit_volume_rank_min": 0.75,
            "binance_variation_rank_min": 0.65,
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
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.10, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "accounting": "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "post_stage_volatility_audit": {"prerequisite": "unchanged candidate passes train, test, eval, final", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "source_plan": {
            "upbit": {"table": "bars_upbit", "symbol": "KRW-BTC", "interval": "1m", "columns": ["ts", "open", "high", "low", "close", "volume"], "read_only": True},
            "perpetual": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "high", "low", "close"], "read_only": True},
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        "diagnostic_controls": {"names": ["no_volume_gate", "no_variation_gate", "one_day_stale_features", "direction_flip"], "diagnostic_controls_cannot_be_promoted": True},
        "research_boundary": {"database_metadata_only_opened_before_preregistration": True, "upbit_volume_values_used_to_select_rule": False, "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False, "selection_basis": "independent Korean cash participation channel plus user-required high volatility"},
        "stopping_rule": "Terminal first-failure sequence: source support, Gross9 novelty, strict economics; no session, source, rank, side, hold, timing, volatility, subset, threshold, or control repair.",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("KCVSR preregistration hash mismatch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(); validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
