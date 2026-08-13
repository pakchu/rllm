"""Outcome-blind preregistration for HVSENC-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVSENC-8"
SLUG = "high_volatility_sample_entropy_collapse_continuation"
DEFAULT_OUTPUT = Path(f"results/{SLUG}_preregistration_2026-08-13.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_sample_entropy_collapse_continuation_v1",
        "policy_id": POLICY_ID,
        "slug": SLUG,
        "as_of_date": "2026-08-13",
        "singleton": True,
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "mechanism": {
            "claim": (
                "A volatile completed BTC path with unusually low Sample Entropy has a high conditional "
                "probability that similar two-return templates remain similar for one additional return. "
                "At a fresh entropy-collapse onset, follow the completed displacement for eight hours."
            ),
            "side": "strict sign of the completed eight-hour BTC return",
            "why_distinct": (
                "Sign entropy ignores ordering, permutation entropy ranks ordinal motifs, Lempel-Ziv parses "
                "symbol strings, and recurrence determinism measures diagonal runs in a state recurrence "
                "plot. HVSENC computes the negative log conditional match probability for continuous "
                "standardized return templates under one fixed Chebyshev tolerance. It uses no prior event "
                "set, promoted control, volume, flow, funding, OI, premium, cross-asset input, or fitted outcome."
            ),
            "why_suited_to_volatile_regimes": (
                "the completed eight-hour realized variation must occupy its causal upper 35 percent while "
                "conditional path uncertainty enters its lower quartile"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "offset three-daily continuous-template entropy-collapse onsets are absent from Gross9 primitives"
            ),
        },
        "features": {
            "decision_grid": "exact 05:00, 13:00 and 21:00 UTC boundaries D",
            "source_window": (
                "480 exact unique coherent bars_binance BTCUSDT interval=1m rows [D-8h,D)"
            ),
            "five_minute_aggregation": (
                "96 exact epoch-aligned groups of five consecutive rows; OHLC first/max/min/last"
            ),
            "returns": "95 close-to-close natural-log returns from the 96 five-minute closes",
            "standardization": (
                "subtract the 95-return arithmetic mean and divide by population standard deviation; "
                "standard deviation must be finite strict positive"
            ),
            "template_dimension": 2,
            "distance": "Chebyshev maximum absolute component difference",
            "tolerance": "0.20 in standardized-return units",
            "match_counts": (
                "unordered distinct template pairs i<j; B counts length-2 pairs with distance<=0.20 and "
                "A counts the corresponding length-3 pairs with distance<=0.20; self-matches excluded"
            ),
            "sample_entropy": "-log(A/B), requiring integers B>0 and A>0 and finite entropy>=0",
            "entropy_rank": (
                "strict-prior midrank over at most 270 earlier source-valid decisions, minimum 180, current "
                "excluded; rank<=0.25"
            ),
            "realized_variation": "sqrt(sum squared 95 five-minute returns), finite strict positive",
            "variation_rank": (
                "strict-prior midrank over at most 270 earlier source-valid decisions, minimum 180, current "
                "excluded; rank>=0.65"
            ),
            "block_return": "log(last five-minute close/first five-minute open), finite strict nonzero",
            "onset": (
                "eligible now and immediately preceding scheduled source-valid decision ineligible; missing "
                "or invalid predecessor cannot trigger"
            ),
            "no_imputation": True,
        },
        "clock": {
            "decision": "D after the full completed path is available",
            "entry": "exact BTCUSDT perpetual D+5m open",
            "side": "sign(completed block return)",
            "hold": "8 elapsed hours",
            "reservation": "global chronological half-open; exit first on equal entry",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact held settlements only after novelty",
        },
        "policy": {
            "window_minutes": 480,
            "aggregation_minutes": 5,
            "template_dimension": 2,
            "tolerance_standard_deviations": 0.2,
            "history_decisions": 270,
            "minimum_history_decisions": 180,
            "entropy_rank_max": 0.25,
            "variation_rank_min": 0.65,
            "decision_hours_utc": [5, 13, 21],
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
                "fixed quantity, exact funding, 6bp/10bp per notional side, every held 5m favorable then "
                "adverse, global HWM, full-calendar CAGR"
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
                "no_entropy_gate", "no_variation_gate", "sign_entropy_low_tail",
                "one_decision_stale_entropy", "direction_flip", "forced_long",
            ],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "bars": {
                "table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close"],
            },
            "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "read_after_preregistration": True,
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "database_schema_and_coverage_metadata_only_opened_before_preregistration": True,
            "prior_entropy_lz_recurrence_hurst_and_multifractal_family_outcomes_known": True,
            "repository_continuous_sample_entropy_candidate_found": False,
            "prior_event_sets_or_controls_reused": False,
            "prior_outcomes_used_to_set_formula_tolerance_rank_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent continuous conditional-template uncertainty mechanism",
        },
        "stopping_rule": (
            "terminal first failure; no aggregation, normalization, dimension, tolerance, match law, history, "
            "rank, onset, variation, side, clock, hold, subset, threshold, comparator, or control repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core) or payload != build():
        raise RuntimeError("HVSENC preregistration drift")
    if payload["outcomes_opened"] or payload["source_incidence_opened"] or payload["gross9_rows_opened"]:
        raise RuntimeError("HVSENC evidence boundary opened")


def main() -> None:
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


if __name__ == "__main__":
    main()
