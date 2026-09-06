"""Outcome-blind preregistration for HVCMDM-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "results/high_volatility_cross_maturity_depth_migration_relay_preregistration_2026-08-10.json"
)


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_cross_maturity_depth_migration_relay_v1",
        "policy_id": "HVCMDM-8",
        "as_of_date": "2026-08-10",
        "oos_outcomes_opened": False,
        "oos_source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "During volatile BTC trading, an abrupt migration of displayed executable depth "
                "between the two live COIN-M quarterly maturities is directional only when the "
                "far maturity simultaneously becomes bid- versus ask-heavy relative to the near "
                "maturity. Trade the signed cross-maturity pressure at the migration onset."
            ),
            "side": "strict sign of far-minus-near one-percent log bid/ask pressure",
            "why_distinct": (
                "Prior depth candidates used perpetual USD-M versus perpetual COIN-M geometry, "
                "refill, shells, credibility, or reported-notional centroids. Prior quarterly "
                "candidates used price, OI, or calendar-curve roll. HVCMDM uses the displayed-depth "
                "allocation and directional pressure between two dated quarterly maturities."
            ),
            "volatile_market_target": "strict-prior BTC 30-minute realized-variation rank >=0.65",
            "why_low_gross9_overlap_is_plausible": (
                "the clock is a data-driven half-hour migration onset rather than a fixed release, "
                "funding, daily, or quarterly-expiry entry"
            ),
        },
        "source_contract": {
            "official_archive": "https://data.binance.vision/data/futures/cm/daily/bookDepth/",
            "official_repository": "https://github.com/binance/binance-public-data",
            "instrument_family": "BTCUSD_YYMMDD COIN-M delivery futures",
            "source_window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "daily_zip": "checksum-verified official bookDepth file for each required contract/day",
            "snapshot_schema": ["timestamp", "percentage", "depth", "notional"],
            "required_levels": [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5],
            "near_contract": "earliest listed quarterly expiry strictly after decision time",
            "far_contract": "second-earliest listed quarterly expiry strictly after decision time",
            "expiry_day_rule": (
                "a contract is eligible only before its encoded UTC expiry timestamp 08:00; at or "
                "after 08:00 it is removed before selecting near and far"
            ),
            "decision_grid": "every exact UTC half hour D",
            "feature_window": "all complete snapshots with timestamp in [D-30m,D)",
            "snapshot_validity": (
                "exactly one finite strictly-positive depth and notional row at every required "
                "percentage; duplicate or malformed snapshot rejected"
            ),
            "window_validity": (
                "each maturity has at least 40 valid snapshots, first no later than D-29m and last "
                "no earlier than D-1m; no interpolation, forward fill, or nearest-time join"
            ),
            "causal_availability": (
                "only snapshots timestamped before D are used; historical files replay the official "
                "real-time order-book observable, and production must reconstruct the same percentage "
                "bands from sequenced Binance COIN-M depth updates before D"
            ),
        },
        "feature_contract": {
            "near_pressure": "median over window of log(near bid depth at -1% / near ask depth at +1%)",
            "far_pressure": "same formula for far maturity",
            "term_pressure": "far_pressure-near_pressure; strict nonzero",
            "near_mass": "median over window of near(-1% depth + +1% depth)",
            "far_mass": "same formula for far maturity",
            "far_share": "far_mass/(near_mass+far_mass), denominator strict positive",
            "migration": "far_share minus immediately preceding exact half-hour far_share; prior source must be valid",
            "absolute_migration_rank": (
                "strict-prior midrank of abs(migration), current excluded, at most 1,440 valid "
                "half-hours and minimum 480"
            ),
            "absolute_term_pressure_rank": "same strict-prior 1440/480 rule for abs(term_pressure)",
            "btc_variation": (
                "sum squared close-to-close log returns from exact BTCUSDT bars_binance 1m rows "
                "[D-30m,D); all timestamps coherent and complete"
            ),
            "btc_variation_rank": "same strict-prior 1440/480 rule",
            "eligible_state": (
                "absolute_migration_rank>=0.85, absolute_term_pressure_rank>=0.75, "
                "btc_variation_rank>=0.65, and term_pressure strict nonzero"
            ),
            "onset": "current eligible state true and immediately previous exact half-hour state false",
            "no_imputation": True,
        },
        "oos_clock": {
            "start": "2023-07-01T00:00:00Z",
            "entry": "D+5m BTCUSDT perpetual open",
            "side": "sign term_pressure",
            "hold": "8 elapsed hours",
            "reservation": (
                "chronological first eligible onset while flat; intervals half-open and exit first "
                "on an equal-time entry"
            ),
            "funding": "not an input; exact settlements opened only after novelty passes",
        },
        "policy": {
            "history_observations": 1440,
            "minimum_history_observations": 480,
            "absolute_migration_rank_min": 0.85,
            "absolute_term_pressure_rank_min": 0.75,
            "btc_variation_rank_min": 0.65,
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
            "minority_side_share_min": 0.20,
            "max_month_share": 0.45,
            "every_required_contract_day_checksum_verified": True,
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
            "accounting": (
                "fixed quantity, exact funding, 6bp/10bp per notional side, favorable-then-adverse "
                "held 5m path, global HWM, full-calendar CAGR"
            ),
        },
        "post_stage_volatility_audit": {
            "prerequisite": "unchanged passes every sequential economic stage",
            "rv20_q90_entry_filter": False,
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "identical_clock_forced_long_residual_positive": True,
        },
        "diagnostic_controls": {
            "names": [
                "no_volatility_gate",
                "no_migration_gate",
                "near_pressure_only",
                "one_decision_stale_features",
                "direction_flip",
                "forced_long",
            ],
            "cannot_be_promoted": True,
        },
        "research_boundary": {
            "prior_perpetual_depth_family_outcomes_known": True,
            "prior_quarterly_price_oi_roll_outcomes_known": True,
            "candidate_specific_incidence_opened": False,
            "candidate_specific_postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "selection_basis": (
                "new dated-maturity executable-depth allocation mechanism fixed before incidence "
                "or outcomes"
            ),
        },
        "stopping_rule": (
            "freeze preregistration, source support, Gross9 novelty, and sequential economics; "
            "terminal first failure with no contract mapping, window, rank, onset, side, hold, "
            "clock, subset, or control repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)
