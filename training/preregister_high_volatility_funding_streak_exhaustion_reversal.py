"""Outcome-blind preregistration for HVFSE-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVFSE-8"
SLUG = "high_volatility_funding_streak_exhaustion_reversal"
DEFAULT_OUTPUT = Path(f"results/{SLUG}_preregistration_2026-08-13.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": f"{SLUG}_v1",
        "policy_id": POLICY_ID,
        "slug": SLUG,
        "as_of_date": "2026-08-13",
        "singleton": True,
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "mechanism": {
            "claim": (
                "Three or more consecutive actual BTC funding settlements with one strict sign "
                "represent repeated carry transfers by the same crowded side. When the cumulative "
                "absolute transfer enters an extreme causal tail during elevated BTC variation, "
                "the inventory is exhausted; fade the funding-streak sign for the next cycle."
            ),
            "side": "negative strict sign shared by every settlement in the active funding streak",
            "why_distinct": (
                "HVEFR uses one settlement's residual from a rolling median; HVFADR uses one "
                "first difference and intervening price divergence; HVAFC waits for adverse price "
                "and OI contraction; settlement-cash sponsorship requires directional price and "
                "flow confirmation. HVFSE uses only the run length and cumulative signed carry of "
                "consecutive realized settlements plus a volatility regime, with no funding-level "
                "residual, funding change, price direction, OI, premium, flow, fitted outcome, "
                "reused event set, or promoted control."
            ),
            "why_suited_to_volatile_regimes": (
                "completed prior-24-hour BTC variation must occupy its causal upper 35 percent"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "actual settlement streak-tail onsets are absent from Gross9 primitives"
            ),
        },
        "features": {
            "decision_grid": "each actual BTCUSDT funding_time S",
            "settlement_valid": "finite strict-nonzero realized funding_rate known at S",
            "streak": (
                "maximal consecutive suffix through S whose funding rates share one strict sign "
                "and whose adjacent actual settlement gaps are exactly eight elapsed hours"
            ),
            "minimum_streak": "at least 3 settlements including S",
            "cumulative_funding": "sum of funding_rate over the active streak; strict same sign",
            "cumulative_magnitude": "absolute cumulative_funding, finite strict positive",
            "magnitude_rank": (
                "strict-prior midrank over at most 270 earlier source-valid streak opportunities, "
                "minimum 180, current excluded; rank>=0.75"
            ),
            "btc_realized_variation": (
                "sqrt(sum squared exact BTC perpetual one-minute log(close/open) returns over "
                "[S-24h,S))"
            ),
            "variation_rank": (
                "strict-prior midrank over at most 270 earlier source-valid settlement states, "
                "minimum 180, current excluded; rank>=0.65"
            ),
            "eligible_state": "minimum streak, magnitude rank, and variation rank pass",
            "onset": (
                "eligible now and immediately preceding source-valid settlement state ineligible; "
                "missing or non-eight-hour prior state cannot trigger"
            ),
            "no_imputation": True,
        },
        "clock": {
            "decision": "actual funding settlement S after rate and completed BTC path are available",
            "entry": "exact BTCUSDT perpetual S+5m open",
            "side": "negative active funding-streak sign",
            "hold": "8 elapsed hours",
            "reservation": "global chronological half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": (
                "signal settlement occurs before entry; every exact held-interval settlement is "
                "included only after novelty"
            ),
        },
        "policy": {
            "required_settlement_gap_hours": 8,
            "minimum_streak_settlements": 3,
            "history_states": 270,
            "minimum_history_states": 180,
            "cumulative_magnitude_rank_min": 0.75,
            "variation_rank_min": 0.65,
            "onset_required": True,
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
                "fixed quantity, exact funding, 6bp/10bp per notional side, every held 5m "
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
                "no_magnitude_tail",
                "no_variation_gate",
                "single_settlement_level",
                "one_settlement_stale_streak",
                "direction_flip",
                "forced_long",
            ],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "funding": {
                "table": "funding_rates_binance",
                "symbol": "BTCUSDT",
                "columns": ["funding_time", "funding_rate"],
                "actual_timestamps": True,
            },
            "btc": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close"],
            },
            "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "read_after_preregistration_commit": True,
            "execution_prices": "sealed until source and novelty pass",
        },
        "research_boundary": {
            "prior_funding_level_change_oi_and_cash_sponsorship_results_known": True,
            "repository_exact_consecutive_funding_streak_candidate_found": False,
            "prior_event_sets_or_controls_reused": False,
            "prior_results_used_to_set_formula_ranks_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent repeated-carry crowding exhaustion mechanism",
        },
        "stopping_rule": (
            "terminal first failure; no settlement gap, minimum streak, cumulative formula, "
            "history, rank, variation, onset, side, clock, hold, subset, threshold, comparator, "
            "or control repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVFSE preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build()
    validate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if args.output.exists() and args.output.read_bytes() != encoded:
        raise RuntimeError(f"refusing overwrite {args.output}")
    args.output.write_bytes(encoded)
    print(args.output)
