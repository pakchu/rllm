"""Outcome-blind preregistration for FSVIBR-6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "results/funding_settlement_volatility_ignition_breakout_relay_preregistration_2026-08-08.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "funding_settlement_volatility_ignition_breakout_relay_v1",
        "policy_id": "FSVIBR-6",
        "as_of_date": "2026-08-08",
        "outcomes_opened": False,
        "mechanism": {
            "claim": (
                "Immediately after an actual BTC perpetual funding cash settlement, simultaneous "
                "BVOL and DVOL expansion plus an unusually large completed-hour BTC move marks "
                "new cross-venue risk discovery after old inventory has been cash-settled; the "
                "post-settlement breakout direction should relay for six hours."
            ),
            "side": "sign of the completed post-settlement one-hour BTC return",
            "why_distinct": (
                "FSVIBR uses settlement as a scheduled inventory-clearing anchor followed by joint "
                "volatility ignition and a price breakout. FSVUR and FSVCCR required extreme funding, "
                "a pre-settlement trend, and post-settlement volatility cooling; CVVIB used fixed "
                "four-hour compression blocks without a funding settlement anchor."
            ),
            "why_low_gross9_overlap_is_plausible": (
                "actual funding settlement plus cross-venue implied-volatility ignition is absent "
                "from Gross9 and permits at most three source decisions daily"
            ),
        },
        "clock": {
            "event": "actual nonzero Binance BTCUSDT USD-M funding settlement S",
            "post_settlement_hour": "exact completed hour [S,S+1h)",
            "volatility_ignition": (
                "normalized BVOL and DVOL bodies over [S,S+1h) are both strictly positive"
            ),
            "price_breakout": (
                "nonzero absolute BTC return over [S,S+1h) >= strictly-prior 270-settlement "
                "q60, requiring 252 valid prior settlement observations"
            ),
            "decision": "S+1h after funding, BTC price, BVOL, and DVOL are complete",
            "entry": "exact BTCUSDT S+1h+5m open",
            "side": "sign of post-settlement one-hour return",
            "hold": "6 elapsed hours",
            "reservation": "settlements are normally eight hours apart; global half-open",
            "funding_rate": "event identity only; magnitude and sign are not signal inputs",
            "oi": "not a signal input",
            "no_imputation": True,
        },
        "policy": {
            "prior_settlements": 270,
            "prior_min_settlements": 252,
            "absolute_post_settlement_return_quantile": 0.60,
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
            "funding": "frozen train settlements plus causal Postgres funding_rates_binance later settlements",
            "volatility": "reuse hash-bound OCDR-12C BVOL/DVOL snapshot through 2026-08-01",
            "completed_price": "reuse hash-bound OLIAH completed-hour snapshot through 2026-08-01",
            "execution_price": "sealed until source-support and Gross9 novelty pass",
        },
        "diagnostic_controls": {
            "names": [
                "bvol_only_expansion",
                "dvol_only_expansion",
                "no_return_tail",
                "one_settlement_stale_volatility",
                "direction_flip",
            ],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        "research_boundary": {
            "prior_settlement_candidate_incidence_known": True,
            "prior_candidate_outcomes_used_to_define_fsvibr": False,
            "fsvibr_candidate_incidence_opened": False,
            "fsvibr_post_entry_return_or_pnl_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n")
    print(args.output)
