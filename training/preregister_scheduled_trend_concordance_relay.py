"""Outcome-blind preregistration for STCR-72."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "STCR-72"
DEFAULT_OUTPUT = Path("results/scheduled_trend_concordance_relay_preregistration_2026-08-09.json")
MARKET = Path("data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz")
MARKET_SHA256 = "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "scheduled_trend_concordance_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "When BTC's completed three-calendar-day and fourteen-calendar-day returns "
                "have the same sign at a low-turnover twice-weekly decision, short-horizon "
                "price discovery is aligned with the medium trend and should persist for the "
                "next three elapsed days."
            ),
            "side": "common strict sign of completed 3-day and 14-day log returns",
            "why_distinct": (
                "STCR is a fixed Monday/Thursday calendar relay with no rank, magnitude, "
                "volatility entry gate, fitted model, order flow, funding, OI, external market, "
                "Gross9 state, or prior candidate control."
            ),
            "volatile_market_target": (
                "the full-calendar policy is evaluated separately on the exact causal RV20 q90 "
                "stress slice and against an identical-clock persistent-long-vol comparator"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "only two fixed UTC decision opportunities per week with three-day reservation"
            ),
        },
        "features": {
            "reference": "BTCUSDT perpetual exact UTC calendar-day close from completed 5m bars",
            "decision_days": ["Monday", "Thursday"],
            "decision_time": "00:00 UTC after the prior UTC day is complete",
            "return_3d": "log(C[t-1]/C[t-4])",
            "return_14d": "log(C[t-1]/C[t-15])",
            "eligible": "both finite, nonzero, and identical strict sign",
            "missing_duplicate_nonpositive": "ineligible; no imputation",
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
            "entry": "exact decision time + one completed 5m bar, at 00:05 UTC",
            "hold": "72 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact realized funding only after novelty",
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
            "names": ["three_day_only", "fourteen_day_only", "direction_flip"],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "historical": {"path": str(MARKET), "sha256": MARKET_SHA256},
            "live_extension": (
                "read-only Postgres bars_binance BTCUSDT 1m through 2026-08-01; "
                "exact five-minute overlap parity required"
            ),
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "exact_stcr_outcomes_known": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
        },
        "stopping_rule": (
            "terminal first failure; no schedule, horizon, agreement, side, hold, RV20, "
            "threshold, subset, comparator, or gate repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("STCR preregistration hash mismatch")
    if hashlib.sha256(MARKET.read_bytes()).hexdigest() != MARKET_SHA256:
        raise RuntimeError("STCR historical source drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(args.output)
