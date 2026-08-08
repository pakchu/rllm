"""Outcome-blind preregistration for HVELR-6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("results/high_volatility_eth_leadership_relay_preregistration_2026-08-09.json")


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_eth_leadership_relay_v1",
        "policy_id": "HVELR-6",
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": "At a fixed eight-hour boundary in an unusually volatile BTC block, a same-direction final-two-hour ETH move that is causally extreme and at least 1.5 times the BTC move identifies crypto-beta price discovery led by ETH. Follow the shared direction in BTC for six elapsed hours.",
            "side": "common strict nonzero sign of final-two-hour ETHUSDT and BTCUSDT returns",
            "why_distinct": "HVELR uses cross-asset ETH-versus-BTC price leadership on complete fixed blocks. SLVCR used same-asset Binance spot-versus-perpetual partial transmission under BVOL/DVOL expansion; EBLR used liquidation event feeds; cross-sectional diffusion used a broader legacy basket and a different research window. No prior control is promoted.",
            "why_suited_to_volatile_regimes": "completed BTC eight-hour realized variation must rank in its causal upper 35%",
            "why_low_gross9_overlap_is_plausible": "three fixed UTC cross-asset leadership clocks and an ETH tail gate are absent from Gross9",
        },
        "features": {
            "btc_source": "BTCUSDT bars_binance interval=1m OHLC",
            "eth_source": "ETHUSDT bars_binance interval=1m OHLC",
            "block_valid": "480 exact aligned distinct minute rows per symbol in [decision-8h,decision), finite positive coherent OHLC; no imputation",
            "late_returns": "log(close at decision-1m/open at decision-2h) independently for ETH and BTC",
            "direction_gate": "ETH and BTC final-two-hour returns have the same strict nonzero sign",
            "leadership_gate": "abs(ETH return)>=1.5*abs(BTC return)",
            "eth_tail_rank": "strict-prior midrank of abs ETH final-two-hour return among at most 270 valid fixed blocks, minimum 180, current excluded; rank>=0.80",
            "btc_realized_variation": "sum squared BTC 1m close-to-close log returns within completed 8h block",
            "btc_variation_rank": "strict-prior midrank among at most 270 valid fixed blocks, minimum 180, current excluded; rank>=0.65",
            "no_imputation": True,
        },
        "clock": {
            "decision": "exact 00:00,08:00,16:00 UTC",
            "entry": "exact decision+5m BTCUSDT open",
            "hold": "6 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "funding_oi_premium_implied_vol": "not signal inputs; exact BTC funding only after novelty passes",
        },
        "policy": {
            "history_observations": 270,
            "minimum_history_observations": 180,
            "btc_variation_rank_min": 0.65,
            "eth_absolute_return_rank_min": 0.80,
            "minimum_eth_to_btc_absolute_return_ratio": 1.5,
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
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.2, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.1, "candidate_near_6h_share_max": 0.35, "occupied_5m_bar_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.1, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "future_can_rank_repair_or_reselect": False, "accounting": "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "source_plan": {
            "binance_1m": {"table": "bars_binance", "symbols": ["BTCUSDT", "ETHUSDT"], "interval": "1m", "columns": ["ts", "symbol", "open", "high", "low", "close"], "window": ["2023-04-01T00:00:00Z", "2026-08-01T00:00:00Z"], "materialize_after_preregistration": True},
            "execution_price": "sealed until source-support and Gross9 novelty pass",
        },
        "diagnostic_controls": {"names": ["no_volatility_gate", "no_eth_tail_gate", "no_leadership_ratio_gate", "one_boundary_stale_geometry", "direction_fade"], "diagnostic_controls_cannot_be_promoted": True},
        "research_boundary": {"prior_cross_asset_candidate_incidence_and_outcomes_known": True, "prior_candidate_outcomes_used_to_set_hvelr_direction_threshold_hold_or_clock": False, "hvelr_candidate_incidence_opened": False, "hvelr_post_entry_return_or_pnl_opened": False, "gross9_rows_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False, "selection_basis": "independent high-volatility cross-asset price-leadership mechanism"},
        "stopping_rule": "Terminal first-failure sequence: source support, Gross9 novelty, strict economics; no threshold, side, hold, clock, source, or subset repair.",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(), indent=2) + "\n")
    print(args.output)
