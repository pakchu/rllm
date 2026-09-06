"""Outcome-blind preregistration for HVEJCR-6."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_eth_beta_catchup_relay as base


DEFAULT_OUTPUT = Path(
    "results/high_volatility_eth_jump_catchup_relay_preregistration_2026-08-09.json"
)


def canonical_hash(payload: Any) -> str:
    return base.canonical_hash(payload)


def build() -> dict[str, Any]:
    payload = base.build()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    core.update(
        {
            "protocol_version": "high_volatility_eth_jump_catchup_relay_v1",
            "policy_id": "HVEJCR-6",
            "mechanism": {
                "claim": (
                    "An extreme completed five-minute ETH return relative to strictly prior ETH "
                    "bipower variation, during a high-volatility BTC day, should lead BTC when BTC's "
                    "simultaneous absolute return is at most half as large."
                ),
                "side": "strict sign of the completed ETH five-minute jump return",
                "why_distinct": (
                    "HVEJCR is a sparse one-bar cross-asset lead-lag event with a muted simultaneous "
                    "BTC response. HVEBCR used continuous six-hour beta residuals; HVELR/HVEDR used "
                    "fixed eight-hour blocks and two-hour returns. No prior threshold, clock, control, "
                    "artifact, or outcome is reused."
                ),
                "why_suited_to_volatile_regimes": "BTC prior-day range must exceed frozen 2023H1 q60",
                "why_low_gross9_overlap_is_plausible": (
                    "rare ETH idiosyncratic jump times are not inputs to any Gross9 sleeve"
                ),
            },
            "features": {
                "sources": "aligned exact BTCUSDT and ETHUSDT 5m OHLC aggregated from five 1m rows",
                "returns": "completed close-to-close five-minute log returns for each asset",
                "eth_bipower_variation": (
                    "pi/2 times sum of adjacent absolute ETH-return products over strictly prior "
                    "288 completed returns, excluding the current return"
                ),
                "eth_jump_score": "abs(current ETH return)/sqrt(prior ETH bipower variation/288)",
                "btc_range_volatility": "BTC high/low-1 over strictly prior 288 completed bars",
                "calibration": "2023H1 BTC range q60 and ETH jump-score q99; no post-entry outcomes",
                "muted_btc_response": "abs(current BTC return) <= 0.5*abs(current ETH return)",
                "onset": (
                    "false-to-true ETH jump score>=q99 while BTC range>=q60 and muted BTC response; "
                    "ETH return finite and nonzero"
                ),
                "no_imputation": True,
            },
            "clock": {
                "decision": "every completed aligned 5m bar",
                "entry": "next exact 5m BTCUSDT open",
                "hold": "6 elapsed hours",
                "reservation": "global half-open; exit first on equal open",
                "funding_oi_premium_implied_vol": "not signal inputs; exact BTC funding only after novelty",
            },
            "policy": {
                "bipower_prior_returns": 288,
                "range_bars": 288,
                "range_quantile": 0.60,
                "eth_jump_score_quantile": 0.99,
                "btc_to_eth_absolute_return_ratio_max": 0.50,
                "entry_delay_minutes": 5,
                "hold_hours": 6,
                "leverage": 0.5,
                "base_cost_per_notional_side": 0.0006,
                "stress_cost_per_notional_side": 0.001,
            },
            "diagnostic_controls": {
                "names": [
                    "no_volatility_gate",
                    "no_muted_btc_response_gate",
                    "one_bar_stale_jump_inputs",
                    "direction_flip",
                ],
                "diagnostic_controls_cannot_be_promoted": True,
            },
            "research_boundary": {
                "prior_cross_asset_candidate_incidence_and_outcomes_known": True,
                "prior_outcomes_used_to_set_exact_eth_jump_rule": False,
                "exact_candidate_incidence_opened": False,
                "exact_postentry_return_or_pnl_opened": False,
                "gross9_rows_opened": False,
                "candidate_count": 1,
                "grid": False,
                "repair_of_prior_candidate": False,
                "promoted_prior_control": False,
            },
            "stopping_rule": (
                "terminal first-failure sequence; no jump, response, side, hold, or gate repair"
            ),
        }
    )
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(args.output)
