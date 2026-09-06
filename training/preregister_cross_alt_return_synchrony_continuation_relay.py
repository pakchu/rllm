"""Outcome-blind preregistration for CARSC-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "results/cross_alt_return_synchrony_continuation_relay_preregistration_2026-08-10.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "cross_alt_return_synchrony_continuation_relay_v1",
        "policy_id": "CARSC-8",
        "as_of_date": "2026-08-10",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "When one-minute BTC returns move synchronously with a broad liquid-alt complex "
                "throughout a completed high-variation eight-hour block, the BTC displacement is "
                "a market-wide risk repricing rather than an isolated endpoint print. If at least "
                "four of six alt block returns confirm BTC, follow BTC for eight hours."
            ),
            "side": "strict sign of the completed BTC eight-hour return",
            "why_distinct": (
                "CARSC uses the median of six contemporaneous within-block Pearson correlations "
                "between 480 aligned BTC and alt one-minute log(open-to-close) returns. Endpoint "
                "breadth ignores the intrablock path; turnover diffusion uses quote-volume shares; "
                "tail dependence uses joint extremes; factor residuals remove a fitted common factor. "
                "No flow, funding, OI, premium, prior events, or controls are signal inputs."
            ),
            "volatile_market_target": (
                "BTC completed-block variation must rank in its causal upper 35%; high synchrony "
                "then identifies volatility transmitted across the crypto complex"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "three fixed UTC cross-alt synchrony onsets with eight-hour reservation are absent from Gross9 primitives"
            ),
        },
        "features": {
            "universe": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT"],
            "decision_grid": "exact 00:00/08:00/16:00 UTC boundaries",
            "block": "480 exact coherent aligned bars_binance interval=1m rows [D-8h,D) for every symbol; no imputation",
            "minute_return": "log(close/open) for each completed one-minute bar",
            "pair_correlation": "Pearson correlation of BTC and each alt's 480 aligned minute returns; finite nonzero variance required",
            "synchrony": "median of the six finite BTC-alt pair correlations",
            "synchrony_rank": "strict-prior midrank over at most 270 jointly valid blocks, minimum 180, current excluded; rank>=0.70",
            "btc_variation": "sqrt(sum squared BTC minute returns), strict positive",
            "btc_variation_rank": "strict-prior midrank over at most 270 valid blocks, minimum 180, current excluded; rank>=0.65",
            "block_return": "log(last close/first open) independently for all seven symbols",
            "direction_confirmation": "BTC nonzero and at least four of six strict nonzero alt signs equal BTC",
            "onset": "eligible now and immediately prior exact jointly valid block ineligible; missing prior opportunity cannot trigger",
        },
        "clock": {
            "decision": "completed eight-hour boundary",
            "entry": "exact BTCUSDT perpetual D+5m open",
            "side": "completed BTC return sign",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        "policy": {
            "history_blocks": 270,
            "minimum_history_blocks": 180,
            "synchrony_rank_min": 0.70,
            "variation_rank_min": 0.65,
            "alt_confirmation_min": 4,
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
            "accounting": (
                "fixed quantity, exact funding, 6bp base and 10bp stress per notional side, "
                "every held 5m favorable then adverse, global HWM, full-calendar CAGR"
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
            "names": ["no_synchrony_gate", "raw_median_correlation_above_half", "one_block_stale_synchrony", "direction_flip", "forced_long"],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "market": {
                "table": "bars_binance",
                "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT"],
                "interval": "1m",
                "columns": ["symbol", "ts", "open", "high", "low", "close"],
                "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "database_metadata_only_opened_before_preregistration": True,
            "repository_cross_alt_return_synchrony_candidate_found": False,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_carsc_formula_ranks_direction_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "fixed contemporaneous intrablock return-synchrony transmission mechanism",
        },
        "stopping_rule": (
            "Terminal first-failure sequence: source support, Gross9 novelty, strict economics; "
            "no correlation definition, rank, breadth, variation, side, hold, clock, universe, subset, or control repair."
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(result: dict[str, Any]) -> None:
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    if result["manifest_hash"] != canonical_hash(core):
        raise RuntimeError("CARSC preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(args.output)
