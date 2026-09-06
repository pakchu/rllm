"""Outcome-sequenced preregistration for HVMPS-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVMPS-12"
DEFAULT_OUTPUT = Path("results/high_volatility_month_phase_seasonality_relay_preregistration_2026-08-10.json")
MARKET = Path("data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz")
MARKET_SHA = "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_month_phase_seasonality_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-10",
        "singleton": True,
        "oos_outcomes_opened": False,
        "oos_source_incidence_opened": False,
        "gross9_rows_opened": False,
        "pretraining_outcomes_authorized_after_preregistration": True,
        "mechanism": {
            "claim": "Recurring month-phase cash, collateral, payroll, and derivatives cycles create directional BTC demand at specific calendar dates. Directions learned only before 2023 should persist when BTC variation is elevated.",
            "side": "frozen sign of the selected pre-2023 UTC day-of-month mean twelve-hour return",
            "why_distinct": "HVMPS uses the monthly 1..28 phase axis. It is not the failed intrawweek hour-slot model, a weekend rule, a turn-of-month hard-coded subset, a macro release, or any prior diagnostic control.",
            "volatile_market_target": "completed seven-day BTC variation causal rank at least 0.65",
            "why_low_gross9_overlap_is_plausible": "only eight of twenty-eight month-phase dates are eligible and the causal volatility gate further thins fixed 00:05 UTC entries",
        },
        "training_contract": {
            "fit_window": ["2020-01-01T00:00:00Z", "2023-01-01T00:00:00Z"],
            "decision": "each exact 00:00 UTC boundary",
            "entry": "00:05 UTC exact open",
            "label": "log(open at entry plus twelve elapsed hours divided by entry open)",
            "slot": "UTC calendar day integer 1..28 only",
            "slot_floor": "at least 30 complete fit labels",
            "slot_score": "arithmetic mean of complete fit log-return labels",
            "selection": "four largest strictly positive and four smallest strictly negative means; score then day integer tie break",
            "failure": "terminal if either sign has fewer than four eligible slots",
            "grid": False,
            "refit_after_2022_12_31": False,
            "model_artifact_must_be_frozen_before_oos_incidence": True,
        },
        "oos_clock": {
            "decision": "each exact daily 00:00 UTC boundary from 2023-07-01 onward",
            "eligibility": "UTC day-of-month is one of eight frozen dates and seven-day variation rank>=0.65",
            "variation": "sqrt(sum squared exact completed 5m log returns over [decision-7d,decision))",
            "variation_rank": "strict-prior daily 00:00 midrank, current excluded, maximum 270, minimum 180",
            "entry": "00:05 UTC exact open",
            "hold": "12 elapsed hours",
            "reservation": "chronological half-open nonoverlap; exit first on equal open",
            "gross_exposure": 0.5,
        },
        "stages": {"train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"], "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"], "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"], "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"]},
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.2, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.1, "candidate_near_6h_share_max": 0.35, "occupied_5m_bar_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.1, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "accounting": "fixed quantity, exact funding, 6bp/10bp per notional side, held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "post_stage_volatility_audit": {"prerequisite": "unchanged all-stage pass", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "diagnostic_controls": {"names": ["no_variation_gate", "constant_selected_date_long", "one_day_stale_phase", "direction_flip", "same_clock_forced_long"], "cannot_be_promoted": True},
        "source_plan": {"historical_market": {"path": str(MARKET), "sha256": MARKET_SHA}, "live_extension": "read-only Postgres BTCUSDT through 2026-08-01", "execution_prices": "sealed until source support and novelty pass"},
        "research_boundary": {"prior_calendar_family_outcomes_known": True, "exact_month_phase_model_or_oos_incidence_known": False, "oos_postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False, "selection_basis": "independent month-phase periodic funding mechanism"},
        "stopping_rule": "freeze preregistration, fit only pre-2023, commit model before OOS incidence, then terminal first failure with no slot count, date, sign, variation, clock, hold, subset, comparator, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(report: dict[str, Any]) -> None:
    if report["manifest_hash"] != canonical_hash({key: value for key, value in report.items() if key != "manifest_hash"}):
        raise RuntimeError("HVMPS preregistration drift")
    if hashlib.sha256(MARKET.read_bytes()).hexdigest() != MARKET_SHA:
        raise RuntimeError("HVMPS market drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    report = build(); validate(report); args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n"); print(args.output)
