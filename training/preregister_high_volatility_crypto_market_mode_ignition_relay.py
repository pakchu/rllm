"""Outcome-blind preregistration for HVCMMI-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVCMMI-8"
SLUG = "high_volatility_crypto_market_mode_ignition_relay"
ALTS = ["ADAUSDT", "BNBUSDT", "DOGEUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
DEFAULT_OUTPUT = Path("results/high_volatility_crypto_market_mode_ignition_relay_preregistration_2026-08-10.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_crypto_market_mode_ignition_relay_v1",
        "policy_id": POLICY_ID, "slug": SLUG, "as_of_date": "2026-08-10",
        "outcomes_opened": False, "source_incidence_opened": False,
        "gross9_rows_opened": False, "singleton": True,
        "mechanism": {
            "claim": (
                "When the first eigenmode of six major altcoin minute returns newly enters its "
                "own upper-quartile dominance state, fragmented coin-specific trading has become "
                "a coherent crypto market impulse. During elevated BTC variation, the oriented "
                "common factor's final-hour direction should transmit into BTC for eight hours."
            ),
            "side": "strict sign of the final-hour return vector projected onto the positively oriented current PC1 loading",
            "why_distinct": (
                "HVCMMI uses the eigenvalue share and oriented loading of an alt-return correlation "
                "matrix. HVCACFR used two BTC-versus-alt-median correlations; AFGI used price-free "
                "aggressive-flow geometry. No prior return-market-mode event set or control is reused."
            ),
            "literature_context": {
                "paper": "Liu, Tsyvinski and Wu (2022), Common Risk Factors in Cryptocurrency",
                "doi": "10.1111/jofi.13119",
                "supported_scope": "a cryptocurrency market factor is a distinct common risk factor",
                "implementation_is_not_a_published_replication": True,
            },
            "volatile_market_target": "BTC completed eight-hour variation strict-prior rank >=0.65",
            "why_low_gross9_overlap_is_plausible": "sparse eight-hour alt-return eigenmode onsets are absent from Gross9 primitives",
        },
        "features": {
            "decision_grid": "exact 00:00, 08:00 and 16:00 UTC boundaries",
            "universe": ALTS,
            "aligned_window": "480 exact aligned bars_binance 1m OHLC rows [D-8h,D) for every alt",
            "source_valid": (
                "each symbol has the exact common minute grid; finite positive OHLC with valid high/low; "
                "no duplicate, missing, nearest-time join or imputation"
            ),
            "minute_return": "natural log(close/open)",
            "standardization": "within-window population demean and population standard deviation per alt; every scale strictly positive",
            "correlation": "float64 six-by-six population correlation matrix of standardized minute returns",
            "eigendecomposition": (
                "numpy.linalg.eigh; descending eigenvalues; clip only tiny negatives in [-1e-12,0] to zero; "
                "nonfinite or materially negative eigenvalue invalidates the state"
            ),
            "pc1_variance_share": "largest eigenvalue divided by the sum of six clipped eigenvalues",
            "pc1_orientation": (
                "make loading sum positive; if abs(sum)<=1e-12, make the first frozen-universe maximum-absolute loading positive"
            ),
            "mode_rank": "strict-prior midrank of PC1 share over at most 270 earlier valid blocks, minimum 180; current excluded",
            "mode_dominant": "mode_rank>=0.75",
            "onset": "current mode_dominant and immediately previous exact source-valid block mode_rank<0.75",
            "direction_score": (
                "dot(oriented current PC1 loading, six final-60-minute cumulative log returns); finite and strict nonzero"
            ),
            "btc_variation": "sum of 480 squared BTC minute log(close/open) returns over [D-8h,D)",
            "btc_variation_rank": "strict-prior midrank over at most 270 earlier source-valid blocks, minimum 180; current excluded; rank>=0.65",
            "btc_return_direction_input": False, "no_imputation": True,
        },
        "clock": {
            "decision": "exact eight-hour boundary D after all source paths complete",
            "entry": "exact BTCUSDT D+5m open", "side": "strict sign(direction_score)",
            "hold": "8 elapsed hours", "reservation": "global half-open; exit first on equal entry",
            "split_crossing_action": "skip", "gross_exposure": 0.5,
            "funding": "not a signal input; exact realized funding only after novelty",
        },
        "policy": {
            "window_minutes": 480, "direction_minutes": 60, "cross_section_size": 6,
            "prior_blocks": 270, "minimum_prior_blocks": 180,
            "mode_rank_min": 0.75, "btc_variation_rank_min": 0.65,
            "entry_delay_minutes": 5, "hold_hours": 8, "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006, "stress_cost_per_notional_side": 0.001,
        },
        "stages": {
            "train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
        },
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.2, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.1, "candidate_near_6h_share_max": 0.35, "occupied_5m_bar_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {
            "absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0,
            "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0,
            "weekly_signflip_one_sided_p_max": 0.1, "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True,
            "stop_on_first_failure": True,
            "accounting": "fixed quantity, exact funding, 6bp/10bp per notional side, held 5m favorable then adverse, global HWM, full-calendar CAGR",
        },
        "post_stage_volatility_audit": {"prerequisite": "unchanged all-stage pass", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "diagnostic_controls": {
            "names": ["no_btc_variation_gate", "no_mode_onset", "equal_weight_final_hour", "one_block_stale_geometry", "direction_flip", "forced_long"],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "table": "bars_binance", "symbols": ["BTCUSDT", *ALTS], "interval": "1m",
            "columns": ["ts", "symbol", "open", "high", "low", "close"],
            "query_window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "read_after_preregistration": True, "execution_prices": "sealed until source and novelty pass",
        },
        "research_boundary": {
            "prior_cross_alt_correlation_and_flow_geometry_results_known": True,
            "repository_return_market_mode_candidate_found": False,
            "prior_event_sets_or_controls_reused": False,
            "prior_outcomes_used_to_set_formula_rank_side_hold_or_clock": False,
            "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False, "candidate_count": 1, "grid": False,
            "repair_of_prior_candidate": False, "promoted_prior_control": False,
            "selection_basis": "independent six-alt return eigenmode ignition mechanism",
        },
        "stopping_rule": "terminal first failure; no universe, window, eigenmode, rank, onset, side, variation, clock, hold, subset or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload["manifest_hash"] != canonical_hash(core):
        raise RuntimeError("HVCMMI preregistration hash mismatch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(); result = build(); validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n"); print(args.output)
