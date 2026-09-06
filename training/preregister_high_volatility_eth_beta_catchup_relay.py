"""Outcome-blind preregistration for HVEBCR-12."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_eth_disagreement_resolution_relay as base


DEFAULT_OUTPUT = Path(
    "results/high_volatility_eth_beta_catchup_relay_preregistration_2026-08-09.json"
)


def canonical_hash(payload: Any) -> str:
    return base.canonical_hash(payload)


def build() -> dict[str, Any]:
    payload = base.build()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    core.update(
        {
            "protocol_version": "high_volatility_eth_beta_catchup_relay_v1",
            "policy_id": "HVEBCR-12",
            "mechanism": {
                "claim": (
                    "When ETH's completed six-hour return is extreme relative to its source-only "
                    "contemporaneous beta to BTC during a high-volatility BTC day, ETH has led crypto "
                    "price discovery and BTC should catch up in the residual direction over twelve hours."
                ),
                "side": "strict sign of ETH six-hour return minus frozen beta times BTC six-hour return",
                "why_distinct": (
                    "HVEBCR is a continuous beta-residual tail onset. Prior HVELR/HVEDR used fixed "
                    "eight-hour boundaries, final-two-hour raw direction gates, and six-hour holds. "
                    "No prior clock, threshold, control, fitted artifact, or outcome is reused."
                ),
                "why_suited_to_volatile_regimes": "BTC prior-day range must exceed frozen 2023H1 q60",
                "why_low_gross9_overlap_is_plausible": (
                    "continuous ETH beta-residual onsets are absent from every Gross9 sleeve"
                ),
            },
            "features": {
                "sources": "aligned exact BTCUSDT and ETHUSDT 5m OHLC aggregated from five 1m rows",
                "six_hour_returns": "log(last completed close / close 72 completed bars earlier)",
                "btc_range_volatility": "BTC high/low-1 over prior 288 completed 5m bars",
                "beta": (
                    "zero-intercept OLS sum(BTC_return*ETH_return)/sum(BTC_return^2) on finite "
                    "2023H1 six-hour source returns only"
                ),
                "residual": "ETH six-hour return - frozen beta*BTC six-hour return",
                "calibration": "2023H1 BTC range q60 and absolute residual q95; no post-entry outcomes",
                "onset": (
                    "false-to-true abs(residual)>=q95 while BTC range>=q60; residual finite and nonzero"
                ),
                "no_imputation": True,
            },
            "clock": {
                "decision": "every completed aligned 5m bar",
                "entry": "next exact 5m BTCUSDT open",
                "hold": "12 elapsed hours",
                "reservation": "global half-open; exit first on equal open",
                "funding_oi_premium_implied_vol": "not signal inputs; exact BTC funding only after novelty",
            },
            "policy": {
                "return_bars": 72,
                "range_bars": 288,
                "range_quantile": 0.60,
                "absolute_residual_quantile": 0.95,
                "entry_delay_minutes": 5,
                "hold_hours": 12,
                "leverage": 0.5,
                "base_cost_per_notional_side": 0.0006,
                "stress_cost_per_notional_side": 0.001,
            },
            "source_plan": {
                "binance_1m": {
                    "table": "bars_binance",
                    "symbols": ["BTCUSDT", "ETHUSDT"],
                    "interval": "1m",
                    "aggregate": "exact date_bin 5m with source_rows=5",
                    "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
                    "materialize_after_preregistration": True,
                },
                "execution_price": "sealed until source-support and Gross9 novelty pass",
            },
            "diagnostic_controls": {
                "names": [
                    "no_volatility_gate",
                    "no_residual_tail_gate",
                    "unit_beta_raw_spread",
                    "one_bar_stale_features",
                    "direction_flip",
                ],
                "diagnostic_controls_cannot_be_promoted": True,
            },
            "research_boundary": {
                "prior_cross_asset_candidate_incidence_and_outcomes_known": True,
                "prior_outcomes_used_to_set_exact_beta_residual_rule": False,
                "exact_candidate_incidence_opened": False,
                "exact_postentry_return_or_pnl_opened": False,
                "gross9_rows_opened": False,
                "candidate_count": 1,
                "grid": False,
                "repair_of_prior_candidate": False,
                "promoted_prior_control": False,
            },
            "stopping_rule": (
                "terminal first-failure sequence; no beta, threshold, side, hold, or gate repair"
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
