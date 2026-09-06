"""Outcome-blind preregistration for MFDH-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "MFDH-8"
DEFAULT_OUTPUT = Path(
    "results/monotone_funding_price_divergence_handoff_preregistration_2026-08-09.json"
)


def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "monotone_funding_price_divergence_handoff_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "Three consecutive same-signed BTC funding settlements whose absolute cash "
                "rate rises monotonically while the completed sixteen-hour BTC return moves "
                "against that funding sign identify increasingly expensive losing leverage. "
                "The price direction should persist during the next deleveraging handoff."
            ),
            "side": "strict sign of the completed 16-hour BTC log return",
            "why_distinct": (
                "MFDH uses an ordered three-settlement realized-funding path and adverse "
                "price divergence. It uses no fixed multi-day trend concordance, analog model, "
                "RV20 entry gate, implied volatility, order flow, OI, or Gross9 state."
            ),
            "volatile_market_target": (
                "leveraged crowding that becomes more expensive while losing against price; "
                "the full-calendar policy is later audited on the exact causal RV20 q90 slice"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "entries require a rare ordered funding path at actual eight-hour settlements"
            ),
        },
        "features": {
            "funding_event": "actual completed Binance BTCUSDT USD-M funding settlement S",
            "funding_path": (
                "f0 at S and the immediately preceding f1 and f2 are finite, nonzero, "
                "identically signed, and abs(f2)<abs(f1)<abs(f0)"
            ),
            "return_16h": "log(C[S-1m]/O[S-16h]) from 960 exact completed one-minute bars",
            "divergence": "return_16h is nonzero and sign(return_16h)=-sign(f0)",
            "return_rank": (
                "strict-prior midrank of abs(return_16h) is at least 0.60 among at most 270 "
                "valid settlement observations, minimum 180; current observation excluded"
            ),
            "availability": (
                "all funding events and minute bars must be present by S; missing, duplicated, "
                "non-finite, nonpositive, late, or incoherent rows make S ineligible"
            ),
            "no_imputation": True,
        },
        "rv20_stress_slice": {
            "daily_return": "log(C_d/C_{d-1}) assigned to UTC calendar day d",
            "rv20": "sqrt(365*mean(r_d^2)) over exact returns t-20 through t-1",
            "threshold": (
                "numpy linear 0.90 quantile over 756 available RV20 observations with "
                "decision timestamps strictly before t; current excluded"
            ),
            "active": "RV20(t)>=threshold(t)",
            "entry_filter": False,
            "future_use": "report-only until all sequential full-calendar stages pass",
        },
        "clock": {
            "decision": "actual funding settlement S after the settled rate is available",
            "entry": "exact BTCUSDT S+5m open",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_accounting": (
                "signal uses only completed settlements through S; exact realized funding "
                "during [entry,exit) is opened only after novelty"
            ),
        },
        "policy": {
            "funding_path_settlements": 3,
            "return_lookback_hours": 16,
            "return_rank_history": 270,
            "return_rank_minimum_observations": 180,
            "return_rank_minimum": 0.60,
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
                "fixed quantity, exact funding, 6bp base and 10bp stress per notional side, "
                "every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "post_stage_volatility_audit": {
            "prerequisite": "unchanged candidate passes train, test, eval, and final",
            "persistent_long_vol_comparator": (
                "same accepted candidate entry/exit clock and 0.5 gross, side forced long"
            ),
            "full_calendar_decomposition": "candidate net return minus comparator net return",
            "rv20_q90_decomposition": (
                "same decomposition restricted by causal RV20 state at candidate decision"
            ),
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "candidate_specific_q90_residual_positive": True,
            "comparator_cannot_satisfy_candidate_claim": True,
        },
        "diagnostic_controls": {
            "names": [
                "no_return_rank",
                "two_settlement_acceleration",
                "funding_side_instead_of_price_side",
                "direction_flip",
            ],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "funding": {
                "table": "funding_rates_binance",
                "symbol": "BTCUSDT",
                "columns": ["funding_time", "funding_rate", "mark_price"],
                "read_after_preregistration": True,
            },
            "price": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "exact_mfdh_outcomes_known": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "ranked_design_alternatives_are_not_fallback_candidates": True,
        },
        "stopping_rule": (
            "terminal first failure; no funding path, rank, history, direction, hold, RV20, "
            "threshold, subset, comparator, control, or gate repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("MFDH preregistration hash mismatch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(args.output)
