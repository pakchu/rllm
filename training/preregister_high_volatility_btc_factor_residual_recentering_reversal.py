"""Outcome-blind preregistration for HVBFRRCR-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_cross_structure_action_vote as gates

POLICY_ID = "HVBFRRCR-8"
DEFAULT_OUTPUT = Path("results/high_volatility_btc_factor_residual_recentering_reversal_preregistration_2026-08-18.json")
SOURCE = Path("data/cross_alt_breadth_underreaction_relay_sources_2023_2026/cross_alt_breadth_underreaction_relay_preentry_features.csv.gz")
SOURCE_SHA = "8270b0318d11d16b6b384e64bfaac77ef4bbc4a701dd347c2ababbb093061eae"
ALTS = ("ethusdt_return", "solusdt_return", "bnbusdt_return", "xrpusdt_return", "dogeusdt_return", "adausdt_return")

def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build() -> dict[str, Any]:
    contract = gates.build()
    core = {
        "protocol_version": "high_volatility_btc_factor_residual_recentering_reversal_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-18",
        "singleton": True,
        "candidate_family": [POLICY_ID],
        "candidate_family_size": 1,
        "source_incidence_opened": False,
        "outcomes_opened": False,
        "gross9_rows_opened": False,
        "mechanism": {
            "claim": "A BTC-specific displacement that was a robust cross-sectional outlier and then remains same-signed but contracts at the next sampled auction is already recentering toward the broad crypto factor; trade the remaining convergence during elevated BTC variation.",
            "side": "opposite strict sign of the current BTC-minus-six-alt-median residual",
            "why_distinct": "HVBFRR fits daily OLS and follows a current residual; HVBDRR follows a current residual tail plus contemporaneous dispersion. HVBFRRCR uses two consecutive frozen sampled auctions, requires a prior robust two-MAD BTC outlier and an already-completed same-sign magnitude contraction, and trades convergence without fitted beta, outcome fit, OI, funding, premium, options, or promoted control.",
            "why_suited_to_volatile_regimes": "current causal BTC realized-variation rank must be at least 0.65",
            "why_low_gross9_overlap_is_plausible": "irregular two-auction cross-sectional residual recentering clocks are absent from Gross9 primitives",
        },
        "features": {
            "source": {"path": str(SOURCE), "sha256": SOURCE_SHA},
            "universe": ["BTCUSDT", *[name.removesuffix("_return").upper() for name in ALTS]],
            "decision_grid": "exact 04:00, 12:00, and 20:00 UTC frozen source decisions",
            "previous_decision": "immediately previous source-valid decision exactly eight elapsed hours earlier",
            "alt_factor": "cross-sectional median of the six finite alt block returns independently at previous and current decisions",
            "btc_factor_residual": "BTC block return minus contemporaneous six-alt median",
            "alt_mad": "median absolute deviation of six alt returns about their contemporaneous median; finite strict positive",
            "prior_outlier": "absolute previous residual divided by previous alt MAD is at least 2.0",
            "recentering": "previous and current residuals are strict nonzero with equal sign and absolute current residual strictly below absolute previous residual",
            "variation_gate": "current frozen causal variation_rank>=0.65",
            "no_imputation": True,
        },
        "policy": {
            "prior_standardized_residual_min": 2.0,
            "variation_rank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 8,
            "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.0010,
        },
        "clock": {
            "entry": "current decision+5m exact BTCUSDT open",
            "hold": "8 elapsed hours",
            "reservation": "sampled opportunities are half-open and nonoverlapping; exit first on equal entry",
            "funding": "not a signal input; exact held settlements only after source and Gross9 pass",
        },
        "stages": contract["stages"],
        "source_support_gates": contract["source_support_gates"],
        "gross9_novelty_gates": contract["gross9_novelty_gates"],
        "economic_gates": contract["economic_gates"],
        "diagnostic_controls": {"names": ["no_prior_outlier", "no_recentering", "no_variation_gate", "continuation_side", "forced_long"], "cannot_be_promoted": True},
        "research_boundary": {
            "prior_HVBFRR_source_failure_known": True,
            "prior_HVBDRR_train_failure_known": True,
            "repository_two_auction_btc_factor_residual_recentering_found": False,
            "source_header_and_three_sample_rows_opened_before_preregistration": True,
            "full_candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "classification": "exploratory discovery; not fresh confirmatory evidence",
            "selection_basis": "robust cross-sectional overshoot followed by completed residual contraction",
        },
        "stopping_rule": "source, Gross9, train/test/eval/final; terminal first failure; no universe, factor, MAD, outlier threshold, recentering, variation, side, clock, hold, subset, comparator, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}

def validate(value: dict[str, Any]) -> None:
    if value != build():
        raise RuntimeError("HVBFRRCR-8 preregistration drift")
    if sha256(SOURCE) != SOURCE_SHA:
        raise RuntimeError("HVBFRRCR-8 source drift")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    value = build(); validate(value); args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"); print(args.output)
