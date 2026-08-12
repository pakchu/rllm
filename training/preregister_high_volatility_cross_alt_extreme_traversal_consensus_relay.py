"""Outcome-blind preregistration for HVCATCR-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVCATCR-8"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_cross_alt_extreme_traversal_consensus_relay_"
    "preregistration_2026-08-12.json"
)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_cross_alt_extreme_traversal_consensus_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-12",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "In a high-variation crypto market, agreement across six liquid alt perpetuals "
                "on whether each completed eight-hour path established its absolute low before "
                "its absolute high, or its high before its low, measures a common directional "
                "auction traversal. A new five-of-six traversal consensus should relay into BTC "
                "for the next eight hours."
            ),
            "side": (
                "long when at least five alts have first absolute-low occurrence strictly before "
                "first absolute-high occurrence; short for the exact symmetric consensus"
            ),
            "why_distinct": (
                "HVRTR ordered BTC's own daily extremes and required terminal close location. "
                "Cross-alt close-location, wick, breadth, return-tail, cojump, rank-persistence, "
                "and directional-coherence candidates did not compare the within-path arrival "
                "order of each alt's two absolute extremes. HVCATCR uses a discrete cross-sectional "
                "consensus of six independent path orderings, no BTC return direction, terminal "
                "close location, volume, flow, funding, OI, fitted outcome, prior event set, or "
                "promoted control."
            ),
            "why_suited_to_volatile_regimes": (
                "BTC completed trailing twenty-four-hour realized variation must rank in its "
                "causal upper 35%; the signal therefore targets July-like unstable markets."
            ),
            "why_low_gross9_overlap_is_plausible": (
                "new onsets of a five-of-six cross-alt extrema-arrival ordering at offset "
                "03:05/11:05/19:05 UTC clocks are absent from Gross9 primitives"
            ),
        },
        "features": {
            "universe": ["ADAUSDT", "BNBUSDT", "DOGEUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
            "alt_block": "exact 480 one-minute bars [D-8h,D) for every alt; no imputation",
            "alt_traversal_side": (
                "+1 when the first timestamp attaining the block absolute low is earlier than "
                "the first timestamp attaining the block absolute high; -1 for the strict reverse"
            ),
            "consensus": (
                "eligible directional state only when at least five of six traversal sides equal "
                "+1 or at least five equal -1; ties are impossible within one valid positive range"
            ),
            "trigger": (
                "false-to-true onset of the conjunction of traversal consensus and high variation "
                "against the previous source-valid decision"
            ),
            "btc_variation": (
                "sum squared log(close/open) over exact 1440 BTCUSDT one-minute bars [D-24h,D)"
            ),
            "variation_rank": (
                "strict-prior midrank against at most 270 prior valid decisions; minimum 180; "
                "current excluded; rank at least 0.65"
            ),
            "availability": "D after every required one-minute source row is complete",
            "invalid": (
                "missing, duplicate, nonpositive, inconsistent OHLC, flat-range, or tied first "
                "high/low timestamp makes the decision source-invalid; no imputation"
            ),
        },
        "clock": {
            "decision": "exact 03:00, 11:00, and 19:00 UTC after the completed eight-hour block",
            "entry": "exact BTCUSDT open five elapsed minutes after decision",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        "policy": {
            "block_minutes": 480,
            "consensus_min": 5,
            "variation_hours": 24,
            "history_decisions": 270,
            "minimum_history_decisions": 180,
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
                "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, "
                "every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "post_stage_volatility_audit": {
            "prerequisite": "unchanged candidate passes train, test, eval, and final",
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "identical_clock_forced_long_residual_positive": True,
        },
        "source_plan": {
            "table": "bars_binance",
            "interval": "1m",
            "columns": ["ts", "symbol", "open", "high", "low", "close"],
            "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "read_only": True,
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        "diagnostic_controls": {
            "names": [
                "no_traversal_consensus_gate",
                "no_variation_gate",
                "close_location_consensus_instead_of_extreme_order",
                "one_block_stale_features",
                "direction_flip",
                "forced_long",
            ],
            "cannot_be_promoted": True,
        },
        "research_boundary": {
            "repository_collision_search_opened": True,
            "exact_cross_alt_extreme_traversal_consensus_candidate_found": False,
            "prior_single_asset_range_traversal_and_cross_alt_outcomes_known": True,
            "prior_event_sets_reused": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": (
                "independent cross-sectional consensus of within-path extreme arrival order under "
                "the user-required high-volatility regime"
            ),
        },
        "stopping_rule": (
            "terminal first-failure sequence: source support, Gross9 novelty, train/test/eval/final "
            "strict economics, then RV20 q90 audit; no universe, block, extrema convention, "
            "consensus, onset, history, rank, variation, side, hold, clock, subset, threshold, "
            "source, or control repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build()
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if args.output.exists() and args.output.read_bytes() != encoded:
        raise RuntimeError(f"refusing to overwrite mismatched preregistration: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(json.dumps({"policy_id": POLICY_ID, "output": str(args.output), "sha256": hashlib.sha256(encoded).hexdigest()}))


if __name__ == "__main__":
    main()
