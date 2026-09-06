"""Outcome-blind preregistration for HVCRKC-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("results/high_volatility_count_return_kendall_concordance_relay_preregistration_2026-08-13.json")


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_count_return_kendall_concordance_relay_v1",
        "policy_id": "HVCRKC-8",
        "as_of_date": "2026-08-13",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": "During elevated BTC variation, positive Kendall tau-b concordance between five-minute execution activity and signed return means high-participation auctions systematically sponsor upward repricing; negative concordance means they sponsor downward repricing. When that ordinal direction agrees with completed displacement at a fresh strength-tail onset, follow it for eight elapsed hours.",
            "side": "strict sign of completed-block Kendall tau-b between log1p execution count and signed five-minute return",
            "why_distinct": "HVECE uses endpoint displacement per square-root total count; HVTCDAR compares count dispersion conditional on return sign; HVTLRR uses lagged quote turnover; HVTCR ranks time against price. HVCRKC is a BTC-only contemporaneous pairwise ordinal dependence statistic over activity and signed auction returns. It reuses no event set or control and uses no turnover, taker flow, spot data, funding, OI, premium, fitted outcome, or post-entry data.",
            "why_suited_to_volatile_regimes": "completed eight-hour realized variation must occupy its causal upper 35 percent and absolute ordinal activity-return concordance its causal upper quartile",
            "why_low_gross9_overlap_is_plausible": "02:05/10:05/18:05 UTC rank-concordance onsets are absent from Gross9 structural clocks",
        },
        "features": {
            "decision_grid": "exact 02:00/10:00/18:00 UTC boundaries D",
            "block": "96 exact epoch-aligned five-minute aggregates from 480 unique coherent BTCUSDT perpetual one-minute rows in [D-8h,D)",
            "five_minute_return": "log(last constituent close/first constituent open)",
            "five_minute_execution_count": "sum integer nonnegative number_of_trades over five constituents",
            "activity_coordinate": "log1p(five_minute_execution_count)",
            "kendall_tau_b": "over 96 activity/return pairs, (C-D)/sqrt((C+D+T_activity)*(C+D+T_return)); pairs tied in both coordinates excluded from all four counts; denominator strict positive; tau finite strict nonzero",
            "completed_displacement": "sum of 96 five-minute returns, finite strict nonzero and same sign as Kendall tau-b",
            "realized_variation": "sqrt(sum squared five-minute returns), finite strict positive",
            "strength_rank": "strict-prior midrank of abs(Kendall tau-b) over at most 270 source-valid decisions, minimum 180, current excluded; rank>=0.75",
            "variation_rank": "strict-prior midrank of realized_variation over the same 270/180 history, current excluded; rank>=0.65",
            "onset": "eligible now and immediately preceding scheduled source-valid decision ineligible; missing or invalid prior cannot trigger",
            "no_imputation": True,
        },
        "clock": {
            "decision": "completed eight-hour boundary D at 02:00/10:00/18:00 UTC",
            "entry": "exact BTCUSDT perpetual D+5m open",
            "side": "sign of Kendall tau-b",
            "hold": "8 elapsed hours",
            "reservation": "global chronological half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium": "not signal inputs; exact funding only after novelty",
        },
        "policy": {"prior_decisions": 270, "minimum_prior_decisions": 180, "strength_rank_min": 0.75, "variation_rank_min": 0.65, "decision_hours_utc": [2, 10, 18], "entry_delay_minutes": 5, "hold_hours": 8, "leverage": 0.5, "base_cost_per_notional_side": 0.0006, "stress_cost_per_notional_side": 0.001},
        "stages": {
            "train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
        },
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.20, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.10, "candidate_near_6h_share_max": 0.35, "occupied_5m_bar_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.10, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "future_can_rank_repair_or_reselect": False, "accounting": "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "post_stage_volatility_audit": {"prerequisite": "unchanged candidate passes train, test, eval and final", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "source_plan": {"bars": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "high", "low", "close", "number_of_trades"], "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"], "read_after_preregistration_commit": True}, "execution_price": "sealed until source support and Gross9 novelty pass"},
        "diagnostic_controls": {"names": ["no_strength_tail", "no_variation_gate", "pearson_instead_of_kendall", "one_decision_stale_tau", "direction_flip", "forced_long"], "cannot_be_promoted": True},
        "research_boundary": {"prior_count_turnover_and_time_rank_outcomes_known": True, "repository_exact_count_return_kendall_concordance_event_found": False, "prior_event_sets_or_controls_reused": False, "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "reversal_of_prior_candidate": False, "promoted_prior_control": False, "selection_basis": "independent ordinal execution-activity sponsorship mechanism selected from formula and source-schema audit only"},
        "stopping_rule": "terminal first failure; no aggregation, Kendall formula, history, rank, threshold, variation, confirmation, onset, clock, entry, hold, side, subset, source, comparator, cost, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core) or payload != build():
        raise RuntimeError("HVCRKC preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build(); validate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
