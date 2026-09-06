"""Outcome-blind preregistration for HVSPSR-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVSPSR-12"
DEFAULT_OUTPUT = Path("results/high_volatility_spot_participation_sponsorship_relay_preregistration_2026-08-09.json")


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_spot_participation_sponsorship_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "A high-variation BTC day whose Binance spot share of combined spot and perpetual "
                "quote turnover is unusually large reflects cash-sponsored price discovery rather "
                "than derivatives-only leverage. If spot and perpetual completed-day returns agree, "
                "their common direction should relay for twelve hours."
            ),
            "side": "common strict sign of completed prior-day Binance spot and perpetual returns",
            "why_distinct": (
                "SLVCR used hourly spot-versus-perpetual return magnitude under joint BVOL/DVOL "
                "expansion; DCVPR used Upbit/Binance participation rotation and FX; HVPASR used "
                "premium-index total variation. HVSPSR uses a full-day Binance cash share of quote "
                "turnover, no venue-return leadership, no implied-volatility feed, no FX, and no "
                "premium-index field."
            ),
            "why_suited_to_volatile_regimes": "completed perpetual realized variation must rank at least 0.65 causally",
            "why_low_gross9_overlap_is_plausible": "one daily cash-participation-conditioned 00:05 UTC clock is absent from Gross9",
        },
        "features": {
            "source_day": "exact prior UTC day [D-24h,D)",
            "spot": "1,440 exact bars_binance_spot BTCUSDT 1m rows with finite positive OHLC and nonnegative quote_asset_volume",
            "perpetual": "1,440 exact bars_binance BTCUSDT 1m rows with finite positive coherent OHLC and nonnegative quote_asset_volume",
            "returns": "log(last close/first open) independently by venue; both strict nonzero and same sign",
            "spot_participation_share": "sum spot quote_asset_volume / sum(spot plus perpetual quote_asset_volume); denominator strict positive",
            "btc_realized_variation": "sqrt(sum squared perpetual 1m log(close/open)) over all 1,440 bars; strict positive",
            "causal_ranks": "strict-prior midranks over at most 270 earlier source-valid days, minimum 180; current excluded",
            "eligibility": "spot-participation rank>=0.75 and perpetual realized-variation rank>=0.65",
            "availability": "D 00:00 UTC after both source paths complete",
            "no_imputation": True,
            "grid": False,
        },
        "clock": {
            "decision": "exact D 00:00 UTC after prior UTC day completes",
            "entry": "exact BTCUSDT perpetual D 00:05 UTC open",
            "side": "common spot/perpetual completed-day return sign",
            "hold": "12 elapsed hours",
            "reservation": "fixed daily opportunities are nonoverlapping; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium_implied_vol_rv20": "not signal inputs; exact funding only after novelty; RV20 q90 only after all economic stages pass",
        },
        "policy": {
            "session_minutes": 1440,
            "history_days": 270,
            "minimum_history_days": 180,
            "spot_participation_rank_min": 0.75,
            "btc_variation_rank_min": 0.65,
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
        "diagnostic_controls": {"names": ["no_spot_participation_gate", "no_btc_variation_gate", "no_direction_agreement", "one_day_stale_features", "direction_flip", "same_clock_forced_long"], "diagnostic_controls_cannot_be_promoted": True},
        "source_plan": {
            "spot": {"table": "bars_binance_spot", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "high", "low", "close", "quote_asset_volume"], "read_only": True},
            "perpetual": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "high", "low", "close", "quote_asset_volume"], "read_only": True},
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {"prior_spot_perpetual_family_outcomes_known": True, "spot_participation_values_used_to_select_rule": False, "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False, "selection_basis": "independent cash-share sponsorship mechanism"},
        "stopping_rule": "Terminal first-failure sequence: source support, Gross9 novelty, strict economics; no day, source, rank, agreement, side, hold, timing, subset, threshold, or control repair.",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVSPSR preregistration hash mismatch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    result = build(); validate(result); args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"); print(args.output)
