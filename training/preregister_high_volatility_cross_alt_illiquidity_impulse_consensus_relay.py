"""Outcome-blind preregistration for HVCIIC-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

POLICY_ID = "HVCIIC-8"
SLUG = "high_volatility_cross_alt_illiquidity_impulse_consensus_relay"
ALTS = ("ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT")
DEFAULT_OUTPUT = Path(f"results/{SLUG}_preregistration_2026-08-13.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_cross_alt_illiquidity_impulse_consensus_relay_v1",
        "policy_id": POLICY_ID,
        "slug": SLUG,
        "as_of_date": "2026-08-13",
        "singleton": True,
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "mechanism": {
            "claim": "When at least four of six liquid alt perpetuals make same-direction completed displacements with unusually high absolute return per unit of their own normal transaction capacity, the move reflects a common liquidity vacuum rather than ordinary broad participation. During elevated BTC variation, propagate that common illiquidity-impulse direction into BTC for eight hours.",
            "side": "strict common sign among at least four alts whose symbol-specific illiquidity impulse ranks at or above 0.80",
            "why_distinct": "HVCARTC uses the cross-sectional covariance of returns and turnover surprises. HVLIR and HVPIAR use BTC-only impact. HVDCAFC and HVDQOFS use taker-flow direction. HVCIIC instead normalizes each alt's completed absolute return by its own causal turnover baseline and requires a breadth consensus of symbol-specific impact tails; it uses no BTC directional return, taker split, OI, funding, premium, fitted outcome, prior event set, or promoted control.",
            "why_suited_to_volatile_regimes": "completed trailing twenty-four-hour BTC realized variation must rank in its causal upper 35 percent, directly selecting July-like volatile states",
            "why_low_gross9_overlap_is_plausible": "offset eight-hour cross-alt symbol-normalized illiquidity breadth clocks are absent from Gross9 primitives",
        },
        "features": {
            "decision_grid": "exact 02:00, 10:00, and 18:00 UTC boundaries D",
            "alt_block": "480 exact coherent bars_binance one-minute rows per alt over [D-8h,D)",
            "alt_return": "log(last completed close/first completed open), finite strict nonzero",
            "alt_quote_turnover": "sum quote_asset_volume, finite strict positive",
            "turnover_baseline": "per-symbol median of at most 270 strictly prior source-valid blocks, minimum 180, current excluded",
            "normalized_turnover": "current alt quote turnover divided by its own causal turnover baseline, finite strict positive",
            "illiquidity_impulse": "absolute alt_return divided by normalized_turnover, finite strict positive",
            "impulse_rank": "per-symbol strict-prior midrank of illiquidity_impulse over at most 270 source-valid blocks, minimum 180, current excluded",
            "tail_alt": "impulse_rank>=0.80",
            "consensus": "at least four tail alts share one strict alt_return sign; if both signs could satisfy, larger breadth wins and an equal breadth tie rejects",
            "consensus_strength": "median impulse_rank among tail alts confirming the selected side",
            "btc_realized_variation": "sqrt(sum squared exact BTCUSDT one-minute log(close/open) returns over [D-24h,D)), finite strict positive",
            "variation_rank": "strict-prior midrank over at most 270 earlier source-valid decisions, minimum 180, current excluded; rank>=0.65",
            "eligible_state": "source-valid consensus breadth and BTC variation gate pass",
            "onset": "eligible now and immediately previous exact source-valid decision ineligible; missing prior cannot trigger",
            "no_imputation": True,
        },
        "clock": {
            "decision": "D after all completed source paths are available",
            "entry": "exact BTCUSDT perpetual D+5m open",
            "hold": "8 elapsed hours",
            "reservation": "global chronological half-open; exit first on equal entry",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact realized funding only after novelty passes",
        },
        "policy": {
            "block_minutes": 480,
            "history_blocks": 270,
            "minimum_history_blocks": 180,
            "impulse_rank_min": 0.80,
            "minimum_consensus_breadth": 4,
            "variation_hours": 24,
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
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.2, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.1, "candidate_near_6h_share_max": 0.35, "occupied_5m_bar_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.1, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "accounting": "fixed quantity, exact funding, 6bp/10bp per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "post_stage_volatility_audit": {"prerequisite": "unchanged all-stage pass", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "diagnostic_controls": {"names": ["no_impulse_tail", "no_variation_gate", "raw_return_tail_consensus", "one_block_stale_impulse", "direction_flip", "forced_long"], "cannot_be_promoted": True},
        "source_plan": {"bars": {"table": "bars_binance", "symbols": ["BTCUSDT", *ALTS], "interval": "1m", "columns": ["ts", "symbol", "open", "high", "low", "close", "quote_asset_volume"], "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"], "read_after_preregistration": True}, "execution_prices": "sealed until source support and Gross9 novelty pass"},
        "research_boundary": {"prior_cross_alt_return_turnover_and_impact_family_outcomes_known": True, "repository_symbol_normalized_illiquidity_tail_breadth_candidate_found": False, "prior_event_sets_or_controls_reused": False, "prior_outcomes_used_to_set_formula_rank_side_hold_or_clock": False, "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False, "selection_basis": "independent cross-sectional common liquidity-vacuum transmission mechanism"},
        "stopping_rule": "terminal first failure; no universe, block, normalization, impulse, rank, breadth, variation, onset, side, clock, hold, subset, threshold, comparator, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {k: v for k, v in value.items() if k != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVCIIC preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and json.loads(args.output.read_text()) != result:
        raise RuntimeError(f"refusing overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
