"""Outcome-blind preregistration for FSVUR-6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "results/funding_settlement_volatility_unwind_relay_preregistration_2026-08-08.json"
)


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "funding_settlement_volatility_unwind_relay_v1",
        "policy_id": "FSVUR-6",
        "as_of_date": "2026-08-08",
        "outcomes_opened": False,
        "mechanism": {
            "claim": (
                "An extreme realized funding transfer following a volatile, funding-aligned "
                "eight-hour BTC move identifies crowded directional inventory. If the next "
                "completed hour reverses while both Binance BVOL and Deribit DVOL contract, "
                "the cash-settlement unwind direction should relay for six hours."
            ),
            "side": "sign of the completed post-settlement one-hour BTC return",
            "why_distinct": (
                "FSVUR is anchored to actual funding cash-transfer timestamps and uses the "
                "settled rate, the preceding eight-hour price path, and joint post-settlement "
                "implied-volatility contraction. It does not use DEPAR's opposing venue-vol "
                "state or its within-hour half-path absorption rule."
            ),
            "why_low_gross9_overlap_is_plausible": (
                "Gross9 has no actual-funding-settlement plus cross-venue volatility-reset clock."
            ),
        },
        "clock": {
            "event": "actual Binance BTCUSDT USD-M funding settlement S",
            "funding_extreme": (
                "abs settled rate at S >= strict-prior 270-settlement q60 with 252 observations"
            ),
            "pre_settlement_price": (
                "nonzero BTCUSDT return from S-8h open to S open; sign equals funding-rate sign; "
                "absolute return >= strict-prior 270-settlement q60 with 252 observations"
            ),
            "post_settlement_price": (
                "nonzero completed BTCUSDT return from S open to S+1h close with sign opposite "
                "the pre-settlement return"
            ),
            "volatility_reset": (
                "normalized BVOL body and normalized DVOL body over [S,S+1h) are both strictly negative"
            ),
            "decision": "S+1h, after the price hour and both volatility bars are complete",
            "entry": "exact BTCUSDT S+1h+5m open",
            "side": "sign of post-settlement one-hour return",
            "hold": "6 elapsed hours",
            "reservation": "global half-open, exit first on equal open",
            "oi": "not a signal input",
            "no_imputation": True,
        },
        "policy": {
            "prior_settlements": 270,
            "prior_min_settlements": 252,
            "absolute_funding_quantile": 0.60,
            "absolute_pre_settlement_return_quantile": 0.60,
            "entry_delay_minutes": 5,
            "hold_hours": 6,
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
            "candidate_near_6h_share_max": 0.45,
            "occupied_5m_jaccard_max": 0.30,
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
            "future_can_rank_repair_or_reselect": False,
            "accounting": (
                "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, "
                "every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "source_plan": {
            "funding": "exact realized Binance BTCUSDT settlement rate and timestamp",
            "volatility": "reuse hash-bound OCDR-12C BVOL/DVOL snapshot",
            "completed_price_features": "causal completed 5m bars only through S+1h",
            "execution_price": "sealed until source-support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_candidate_incidence_known": True,
            "prior_candidate_outcomes_used_for_fsvur": False,
            "fsvur_candidate_incidence_opened": False,
            "fsvur_post_entry_return_or_pnl_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n")
    print(args.output)
