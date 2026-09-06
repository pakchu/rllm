"""Outcome-blind preregistration for HVBDRR-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

POLICY_ID = "HVBDRR-8"
ALTS = ("ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT")
DEFAULT_OUTPUT = Path(
    "results/high_volatility_btc_dominance_rotation_relay_preregistration_2026-08-09.json"
)


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_btc_dominance_rotation_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "candidate_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "When BTC develops an unusually large completed-block return residual versus the "
                "median of six liquid alts while cross-alt return dispersion and BTC variation are "
                "both elevated, capital is rotating between BTC and speculative crypto beta rather "
                "than moving the complex uniformly. The BTC-relative direction should persist for "
                "eight elapsed hours."
            ),
            "side": "strict sign of BTC block return minus the six-alt median block return",
            "why_distinct": (
                "CABUR follows the alt majority when BTC underreacts in absolute magnitude. CABLR "
                "uses sign opposition. HVBFRR uses a daily causal OLS coefficient and raw alt mean. "
                "HVBDRR instead requires an extreme cross-sectional BTC-minus-alt-median residual "
                "jointly with elevated alt dispersion at three fixed four-hour clocks."
            ),
            "volatile_market_target": "BTC prior-24h realized-variation causal rank at least 0.65",
            "why_low_gross9_overlap_is_plausible": (
                "Gross9 contains no seven-asset BTC-dominance residual plus dispersion primitive"
            ),
        },
        "features": {
            "source_panel": "hash-bound CABUR seven-symbol source-only four-hour block panel",
            "universe": ["BTCUSDT", *ALTS],
            "blocks": "exact [00:00,04:00), [08:00,12:00), or [16:00,20:00) UTC blocks",
            "returns": "log(last completed minute close/first minute open) independently",
            "alt_factor": "cross-sectional median of the six finite alt block returns",
            "btc_dominance_residual": "BTC block return minus alt_factor; strict nonzero",
            "residual_rank": (
                "strict-prior midrank of absolute residual over at most 270 valid blocks, minimum "
                "180, current excluded; rank>=0.70"
            ),
            "alt_dispersion": "median absolute deviation of the six alt returns about alt_factor",
            "dispersion_rank": (
                "strict-prior midrank over at most 270 valid blocks, minimum 180, current excluded; "
                "rank>=0.65"
            ),
            "btc_realized_variation": (
                "sqrt(sum squared BTC one-minute close-to-close returns over exact 24 hours ending "
                "at decision)"
            ),
            "variation_rank": (
                "strict-prior midrank over at most 270 valid decisions, minimum 180, current "
                "excluded; rank>=0.65"
            ),
            "no_imputation": True,
        },
        "clock": {
            "decision": "exact 04:00, 12:00, or 20:00 UTC after the four-hour block completes",
            "entry": "exact BTCUSDT decision+5m open",
            "side": "sign of btc_dominance_residual",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium_rv20": (
                "not signal inputs; exact funding after novelty; RV20 q90 only after all economics pass"
            ),
        },
        "policy": {
            "prior_blocks": 270,
            "minimum_prior_blocks": 180,
            "absolute_residual_rank_min": 0.70,
            "alt_dispersion_rank_min": 0.65,
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
                "fixed quantity, exact funding, 6bp base and 10bp stress per notional side, every "
                "held 5m favorable then adverse, global HWM, full-calendar CAGR"
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
                "no_residual_tail",
                "no_dispersion_gate",
                "no_variation_gate",
                "alt_factor_direction",
                "one_block_stale_geometry",
                "direction_flip",
                "same_clock_forced_long",
            ],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "features": {
                "path": (
                    "data/cross_alt_breadth_underreaction_relay_sources_2023_2026/"
                    "cross_alt_breadth_underreaction_relay_preentry_features.csv.gz"
                ),
                "read_only": True,
                "reuse_requires_hash_binding": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_cross_alt_candidate_incidence_and_outcomes_known": True,
            "exact_hvbdrr_incidence_or_outcomes_known": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent BTC-dominance rotation and cross-alt dispersion mechanism",
        },
        "stopping_rule": (
            "terminal first failure; no universe, block, factor, dispersion, threshold, side, hold, "
            "timing, subset, comparator, or control repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVBDRR preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(args.output)
