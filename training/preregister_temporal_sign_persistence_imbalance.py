"""Outcome-blind preregistration for TSPI-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "TSPI-12"
DEFAULT_OUTPUT = Path("results/temporal_sign_persistence_imbalance_preregistration_2026-08-09.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "temporal_sign_persistence_imbalance_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "The squared duration mass of positive versus negative one-minute candle-body "
                "runs over a completed UTC day measures persistent directional auction control "
                "without using return magnitude. The dominant run-duration side should persist "
                "over the following twelve hours."
            ),
            "side": "sign(sum positive run_length^2 - sum negative run_length^2)",
            "why_distinct": (
                "TSPI uses the complete distribution of maximal nonzero one-minute body-sign run "
                "durations at one weekly clock. It is not sign entropy, body count, maximum run, "
                "return magnitude, multi-horizon trend, funding, flow, volatility gating, or a model."
            ),
            "volatile_market_target": (
                "directional auction persistence is evaluated full-calendar and later decomposed "
                "on the causal RV20 q90 slice against a forced-long comparator"
            ),
            "why_low_gross9_overlap_is_plausible": "one twelve-hour Wednesday position per week",
        },
        "features": {
            "window": "exactly 1,440 BTCUSDT one-minute bars [Wednesday D-24h,D)",
            "body_sign": "sign(log(close_i/open_i)); exact zero terminates a run",
            "runs": "maximal contiguous equal nonzero body-sign sequences",
            "positive_mass": "sum squared lengths of all positive runs",
            "negative_mass": "sum squared lengths of all negative runs",
            "imbalance": "positive_mass-negative_mass; exact zero is ineligible",
            "eligibility": (
                "exact minute grid, unique timestamps, finite positive coherent OHLC, and at least "
                "one positive and one negative body; no imputation"
            ),
        },
        "rv20_stress_slice": {
            "daily_return": "log(C_d/C_{d-1}) assigned to UTC calendar day d",
            "rv20": "sqrt(365*mean(r_d^2)) over exact returns t-20 through t-1",
            "threshold": (
                "numpy linear 0.90 quantile over 756 available RV20 observations with decision "
                "timestamps strictly before t; current excluded"
            ),
            "active": "RV20(t)>=threshold(t)",
            "entry_filter": False,
            "future_use": "report-only until all sequential full-calendar stages pass",
        },
        "clock": {
            "decision": "each Wednesday 00:00 UTC after the prior 24 hours are complete",
            "entry": "exact BTCUSDT Wednesday 00:05 UTC open",
            "hold": "12 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not an input; exact realized funding only after novelty",
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
            "persistent_long_vol_comparator": "same accepted clock and 0.5 gross, side forced long",
            "full_calendar_decomposition": "candidate net return minus comparator net return",
            "rv20_q90_decomposition": "same decomposition restricted by causal RV20 at decision",
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "candidate_specific_q90_residual_positive": True,
            "comparator_cannot_satisfy_candidate_claim": True,
        },
        "diagnostic_controls": {
            "names": ["body_count", "body_magnitude", "maximum_run", "direction_flip"],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "bars": {
                "table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "exact_tspi_outcomes_known": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
        },
        "stopping_rule": (
            "terminal first failure; no weekday, window, run transform, direction, entry, hold, "
            "RV20, subset, comparator, control, or gate repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("TSPI preregistration hash mismatch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(); result = build(); validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(args.output)
