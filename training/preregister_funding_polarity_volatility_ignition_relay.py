"""Outcome-blind preregistration for FPVIR-6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "results/funding_polarity_volatility_ignition_relay_preregistration_2026-08-08.json"
)


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "funding_polarity_volatility_ignition_relay_v1",
        "policy_id": "FPVIR-6",
        "as_of_date": "2026-08-08",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "mechanism": {
            "claim": (
                "At an actual BTC perpetual funding settlement, a sign reversal whose absolute "
                "rate is larger than the immediately previous settlement marks a rotation in "
                "leveraged directional demand. If both BTC BVOL and DVOL are already elevated, "
                "the new funding polarity should relay for six hours."
            ),
            "side": "sign of the newly settled nonzero funding rate",
            "why_distinct": (
                "FPVIR uses a discrete funding-polarity rotation plus elevated cross-venue "
                "implied-volatility levels. OCDR used a funding magnitude tail, OI expansion, "
                "DVOL body leadership, the opposite funding side, and a twelve-hour hold. "
                "FSVUR/FSVCCR used pre- and post-settlement BTC price paths and volatility cooling."
            ),
            "why_not_pviar_repair": (
                "FPVIR uses no premium-index field, no intrahour acceleration, and no PVIAR "
                "control or direction reversal."
            ),
            "why_low_gross9_overlap_is_plausible": (
                "actual funding-polarity rotations under dual crypto implied-volatility elevation "
                "are absent from all Gross9 structural clocks"
            ),
        },
        "clock": {
            "event": "actual Binance BTCUSDT USD-M funding settlement S",
            "funding_rotation": (
                "current and immediately previous exact settlements are nonzero, have opposite "
                "signs, and abs(current rate)>=abs(previous rate)"
            ),
            "volatility_regime": (
                "BVOL close and DVOL close for the exact completed hour ending S are each >= "
                "their own strictly-prior 720-hour q60 with 672 valid observations"
            ),
            "availability": (
                "funding event and both completed volatility candles are available at S; missing, "
                "duplicate, stale, nonpositive, or non-finite source values are ineligible"
            ),
            "entry": "exact BTCUSDT S+5m open",
            "side": "sign of current settled funding rate",
            "hold": "6 elapsed hours",
            "reservation": "funding settlements are separated by about eight hours; global half-open",
            "btc_price_oi_premium_forbidden_as_signal_inputs": True,
            "no_imputation": True,
        },
        "policy": {
            "volatility_prior_hours": 720,
            "volatility_min_hours": 672,
            "volatility_level_quantile": 0.60,
            "minimum_funding_amplitude_ratio": 1.0,
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
            "future_can_rank_repair_or_reselect": False,
            "accounting": (
                "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, "
                "every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "source_plan": {
            "funding": "reuse hash-bound OCDR-12C exact funding settlement snapshot through 2026-08-01",
            "volatility": "reuse hash-bound OCDR-12C BVOL/DVOL completed-hour snapshot through 2026-08-01",
            "execution_price": "sealed until source-support and Gross9 novelty pass",
        },
        "diagnostic_controls": {
            "names": [
                "no_high_volatility", "no_sign_rotation", "no_amplitude_expansion",
                "one_settlement_stale_volatility", "direction_flip",
            ],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        "research_boundary": {
            "prior_funding_and_volatility_source_rows_known": True,
            "prior_candidate_outcomes_used_to_define_fpvir": False,
            "fpvir_candidate_incidence_opened": False,
            "fpvir_post_entry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
        },
        "stopping_rule": "terminal first failure; no repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n")
    print(args.output)
