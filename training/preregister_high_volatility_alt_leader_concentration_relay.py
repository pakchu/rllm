"""Outcome-blind preregistration for HVALCR-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVALCR-8"
ALTS = ("ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT")
DEFAULT_OUTPUT = Path(
    "results/high_volatility_alt_leader_concentration_relay_preregistration_2026-08-13.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_alt_leader_concentration_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-13",
        "singleton": True,
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "mechanism": {
            "claim": (
                "When one liquid alt perpetual dominates the absolute completed return mass of "
                "the six-alt cross-section, the leader is carrying asset-specific marginal "
                "information rather than a broad crypto beta move. During elevated BTC variation, "
                "propagate the uniquely largest alt return direction into BTC for eight hours at "
                "a fresh upper-tail concentration state."
            ),
            "side": "strict sign of the unique largest-absolute-return alt",
            "why_distinct": (
                "Cross-alt breadth and L1 coherence measure common direction; dispersion measures "
                "spread; tail asymmetry uses median-centered cubes; rank persistence compares "
                "order through time; CLD-72 used prior residual HHI jointly with flow, turnover, "
                "BTC lag, and later breadth. HVALCR uses only the current six-return absolute-mass "
                "HHI and the unique leader sign, with no BTC return direction, flow, volume, OI, "
                "funding, premium, prior event set, fitted outcome, or promoted control."
            ),
            "why_suited_to_volatile_regimes": (
                "BTC trailing twenty-four-hour variation must occupy its causal upper 35% and "
                "leader concentration its upper quartile"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "offset three-daily cross-alt leader-concentration onsets are absent from Gross9"
            ),
        },
        "features": {
            "symbols": list(ALTS),
            "decision_grid": "exact 02:00, 10:00, and 18:00 UTC boundaries D",
            "source_block": (
                "480 exact coherent bars_binance one-minute rows [D-8h,D) for each alt and BTC"
            ),
            "alt_return": "log(last completed close/first completed open), finite strict nonzero",
            "absolute_return_share": (
                "each alt absolute return divided by the strictly positive six-alt absolute-return sum"
            ),
            "leader_concentration": "sum of squared six absolute-return shares, finite in [1/6,1]",
            "unique_leader": (
                "exactly one alt has the strict largest absolute return; ties or zero leader reject"
            ),
            "concentration_rank": (
                "strict-prior midrank over at most 270 earlier source-valid decisions, minimum 180, "
                "current excluded; rank>=0.75"
            ),
            "btc_realized_variation": (
                "sqrt(sum squared exact BTC one-minute open-to-close log returns over [D-24h,D))"
            ),
            "variation_rank": (
                "strict-prior midrank over at most 270 earlier source-valid decisions, minimum 180, "
                "current excluded; rank>=0.65"
            ),
            "eligible_state": "unique leader exists and both causal ranks pass",
            "onset": (
                "eligible now and immediately preceding exact source-valid boundary ineligible; "
                "missing prior cannot trigger"
            ),
            "no_imputation": True,
        },
        "clock": {
            "decision": "D after all seven completed paths are available",
            "entry": "exact BTCUSDT perpetual D+5m open",
            "side": "sign(unique leader alt return)",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
            "rv20": "q90 audit only after unchanged all-stage pass",
        },
        "policy": {
            "block_minutes": 480,
            "history_decisions": 270,
            "minimum_history_decisions": 180,
            "concentration_rank_min": 0.75,
            "variation_rank_min": 0.65,
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
            "minority_side_share_min": 0.2,
            "max_month_share": 0.45,
        },
        "novelty_gates": {
            "exact_entry_jaccard_max": 0.1,
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
            "weekly_signflip_one_sided_p_max": 0.1,
            "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5,
            "each_calendar_half_positive": True,
            "stop_on_first_failure": True,
            "accounting": (
                "fixed quantity, exact funding, 6bp and 10bp per notional side, every held 5m "
                "favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "post_stage_volatility_audit": {
            "prerequisite": "unchanged all-stage pass",
            "rv20_q90_entry_filter": False,
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "identical_clock_forced_long_residual_positive": True,
        },
        "diagnostic_controls": {
            "names": [
                "no_concentration_tail_gate", "no_variation_gate", "equal_weight_alt_direction",
                "one_block_stale_geometry", "direction_flip", "forced_long",
            ],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "table": "bars_binance", "symbols": ["BTCUSDT", *ALTS], "interval": "1m",
            "columns": ["ts", "symbol", "open", "high", "low", "close"],
            "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "read_after_preregistration": True,
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_cross_alt_breadth_dispersion_tail_rank_and_diffusion_outcomes_known": True,
            "repository_current_alt_absolute_return_hhi_candidate_found": False,
            "prior_event_sets_reused": False,
            "prior_outcomes_used_to_set_formula_rank_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent current cross-sectional leader-concentration channel",
        },
        "stopping_rule": (
            "terminal first failure; no universe, block, HHI, leader, history, rank, variation, "
            "side, hold, clock, subset, threshold, source, or control repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVALCR preregistration drift")
    if payload["outcomes_opened"] or payload["source_incidence_opened"] or payload["gross9_rows_opened"]:
        raise RuntimeError("HVALCR evidence boundary opened")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build()
    validate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()
