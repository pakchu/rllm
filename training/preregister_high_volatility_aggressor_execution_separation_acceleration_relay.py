"""Outcome-blind preregistration for HVAESAR-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

POLICY_ID = "HVAESAR-8"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_aggressor_execution_separation_acceleration_relay_"
    "preregistration_2026-08-12.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_aggressor_execution_separation_acceleration_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-12",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "Within a completed high-variation BTC auction, aggressive buyers and residual "
                "aggressive sellers transact at different aggregate VWAPs. When the strict "
                "buy-versus-sell execution-price separation keeps one sign and expands from the "
                "first four hours to the second four hours, directional price control is "
                "accelerating rather than merely reflecting a static path level; follow that "
                "execution-price-dominant side for eight hours."
            ),
            "side": "common strict sign of first-half and second-half log(buy VWAP / sell VWAP)",
            "why_distinct": (
                "AVFCR used one six-hour aggregate separation only when its sign contradicted "
                "aggregate signed taker notional; its no-contradiction diagnostic retained a "
                "static level on the same daily clock. LFIC paired flow with the next bar return. "
                "HVAESAR compares two disjoint four-hour buy/sell execution-price separations and "
                "requires same-sign magnitude acceleration. It does not reuse either event set or "
                "control and uses no completed BTC return direction, funding, OI, premium, fitted "
                "outcome, or cross-asset input."
            ),
            "why_suited_to_volatile_regimes": (
                "completed trailing twenty-four-hour BTC realized variation must rank in its "
                "causal upper 35%, explicitly targeting July-like volatile markets"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "new onsets of disjoint-half aggressor execution-price acceleration at offset "
                "02:05/10:05/18:05 UTC are absent from Gross9 primitives"
            ),
        },
        "features": {
            "block": "exact 480 BTCUSDT one-minute rows [D-8h,D), split into two exact 240-row halves",
            "buy_base": "sum(taker_buy_base) within each half",
            "buy_quote": "sum(taker_buy_quote) within each half",
            "sell_base": "sum(volume-taker_buy_base) within each half",
            "sell_quote": "sum(quote_asset_volume-taker_buy_quote) within each half",
            "separation": "log((buy_quote/buy_base)/(sell_quote/sell_base)), strict nonzero in each half",
            "directional_acceleration": (
                "both half separations have the same strict sign and second-half absolute "
                "separation is at least first-half absolute separation"
            ),
            "first_half_magnitude_rank": (
                "strict-prior midrank of absolute first-half separation against at most 270 prior "
                "source-valid decisions; minimum 180; current excluded; rank at least 0.60"
            ),
            "btc_variation": "sum squared log(close/open) over exact 1440 BTC one-minute rows [D-24h,D)",
            "variation_rank": (
                "strict-prior midrank against at most 270 prior source-valid decisions; minimum "
                "180; current excluded; rank at least 0.65"
            ),
            "trigger": "false-to-true onset of all frozen conditions against the previous source-valid decision",
            "source_valid": (
                "unique exact minute grid; finite coherent positive OHLC; finite nonnegative total "
                "and taker volumes; taker fields not above totals; positive buy and residual-sell "
                "base and quote totals in both halves; no imputation"
            ),
        },
        "clock": {
            "decision": "exact 02:00, 10:00, and 18:00 UTC after each completed eight-hour block",
            "entry": "exact BTCUSDT open five elapsed minutes after decision",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        "policy": {
            "block_minutes": 480,
            "half_minutes": 240,
            "history_decisions": 270,
            "minimum_history_decisions": 180,
            "first_half_magnitude_rank_min": 0.60,
            "minimum_acceleration_ratio": 1.0,
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
        "source_support_gates": {
            "minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8},
            "minority_side_share_min": 0.2,
            "max_month_share": 0.45,
        },
        "novelty_gates": {
            "exact_entry_jaccard_max": 0.1,
            "candidate_near_6h_share_max": 0.35,
            "occupied_5m_bar_jaccard_max": 0.25,
            "absolute_signed_exposure_pearson_max": 0.35,
            "must_pass_before_economics": True,
        },
        "economic_gates": {
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_min": 3.0,
            "strict_mdd_max_pct": 15.0,
            "mean_gross_underlying_min_bp": 20.0,
            "weekly_signflip_one_sided_p_max": 0.1,
            "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5,
            "each_calendar_half_positive": True,
            "stop_on_first_failure": True,
            "accounting": (
                "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, "
                "every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "post_stage_volatility_audit": {
            "prerequisite": "unchanged candidate passes train, test, eval, and final",
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "identical_clock_forced_long_residual_positive": True,
        },
        "diagnostic_controls": {
            "names": [
                "no_acceleration_requirement",
                "no_variation_gate",
                "flow_imbalance_acceleration_instead_of_execution_separation",
                "one_block_stale_features",
                "direction_flip",
                "forced_long",
            ],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "table": "bars_binance",
            "symbol": "BTCUSDT",
            "interval": "1m",
            "columns": [
                "ts", "open", "high", "low", "close", "volume", "quote_asset_volume",
                "taker_buy_base", "taker_buy_quote",
            ],
            "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "read_only": True,
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "repository_collision_search_opened": True,
            "exact_disjoint_half_execution_separation_acceleration_candidate_found": False,
            "prior_avfcr_lfic_outcomes_known": True,
            "prior_event_sets_reused": False,
            "prior_controls_promoted": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "selection_basis": "independent temporal acceleration of aggressor execution-price separation",
        },
        "stopping_rule": (
            "terminal first-failure sequence: source support, Gross9 novelty, train/test/eval/final "
            "strict economics, then RV20 q90 audit; no block, half, VWAP formula, rank, "
            "acceleration, variation, onset, side, hold, clock, subset, threshold, source, or "
            "control repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(); payload = build()
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if args.output.exists() and args.output.read_bytes() != encoded:
        raise RuntimeError(f"refusing to overwrite mismatched preregistration: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_bytes(encoded)
    print(json.dumps({"policy_id": POLICY_ID, "output": str(args.output), "sha256": hashlib.sha256(encoded).hexdigest()}))


if __name__ == "__main__":
    main()
