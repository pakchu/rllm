"""Outcome-blind preregistration for HVLZC-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVLZC-8"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_lempel_ziv_compressibility_relay_preregistration_2026-08-10.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_lempel_ziv_compressibility_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-10",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "A high-variation completed eight-hour BTC block whose one-minute return-sign "
                "sequence has unusually low Lempel-Ziv complexity contains repeated directional "
                "motifs rather than memoryless churn. When the full block and final two hours "
                "agree in direction, the structured price-discovery path should continue for eight hours."
            ),
            "side": "common strict sign of completed eight-hour and final-two-hour returns",
            "why_distinct": (
                "HVLZC parses repeated subsequences with LZ76. Sign entropy uses only marginal sign "
                "frequencies, sign-run asymmetry summarizes run lengths, serial autocorrelation uses "
                "one lag, and ordinal entropy ranks prices. HVLZC uses no flow, volume, funding, OI, "
                "cross-assets, prior event rows, fitted outcome model, or controls."
            ),
            "why_suited_to_volatile_regimes": (
                "the causal realized-variation tail restricts activation to volatile blocks while "
                "low compressibility distinguishes structured displacement from high-volatility noise"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "three fixed UTC sequence-compressibility onsets are absent from Gross9 primitives"
            ),
        },
        "features": {
            "decision_grid": "exact 00:00/08:00/16:00 UTC boundaries",
            "block": "480 exact coherent BTCUSDT bars_binance interval=1m rows [D-8h,D)",
            "returns": "479 close-to-close log returns encoded -1, 0, +1 by strict sign",
            "lz76_phrase_count": (
                "left-to-right exhaustive-history parsing; each phrase is the shortest substring "
                "not previously occurring in the already parsed prefix, with final suffix counted once"
            ),
            "normalized_complexity": "phrase_count*log(479)/479 using natural log; strict finite positive",
            "complexity_rank": (
                "strict-prior midrank over at most 270 valid blocks, minimum 180, current excluded; rank<=0.25"
            ),
            "realized_variation": "sqrt(sum squared one-minute returns), strict positive",
            "variation_rank": (
                "strict-prior midrank over at most 270 valid blocks, minimum 180, current excluded; rank>=0.65"
            ),
            "block_return": "log(last close/first open) over eight hours",
            "late_return": "log(last close/open at D-2h) over final two hours",
            "direction_confirmation": "block and late returns have one strict nonzero sign",
            "onset": "eligible now and immediately prior exact valid block ineligible; missing prior opportunity cannot trigger",
            "no_imputation": True,
        },
        "clock": {
            "decision": "completed eight-hour boundary",
            "entry": "exact BTCUSDT perpetual D+5m open",
            "side": "common completed-return sign",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        "policy": {
            "alphabet_size": 3,
            "parser": "LZ76 exhaustive history",
            "history_blocks": 270,
            "minimum_history_blocks": 180,
            "variation_rank_min": 0.65,
            "complexity_rank_max": 0.25,
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
            "occupied_5m_jaccard_max": 0.25,
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
            "prerequisite": "unchanged candidate passes train, test, eval, final",
            "rv20_q90_entry_filter": False,
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "identical_clock_forced_long_residual_positive": True,
        },
        "diagnostic_controls": {
            "names": [
                "no_complexity_tail",
                "no_variation_gate",
                "raw_below_unit_complexity",
                "one_block_stale_geometry",
                "direction_flip",
                "forced_long",
            ],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "bars": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close"],
                "window": ["2023-04-01T00:00:00Z", "2026-08-01T00:00:00Z"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_sign_entropy_and_run_family_outcomes_known": True,
            "repository_lempel_ziv_price_path_candidate_found": False,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_hvlzc_formula_ranks_direction_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "fixed normalized LZ76 return-sign sequence complexity",
        },
        "stopping_rule": (
            "Terminal first-failure sequence: source support, Gross9 novelty, strict economics; "
            "no formula, rank, side, hold, clock, subset, or control repair."
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)
