"""Outcome-blind preregistration for HVVWMMR-8."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any

POLICY_ID = "HVVWMMR-8"
DEFAULT_OUTPUT = Path("results/high_volatility_volume_weighted_median_migration_relay_preregistration_2026-08-10.json")

def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_volume_weighted_median_migration_relay_v1",
        "policy_id": POLICY_ID, "as_of_date": "2026-08-10",
        "outcomes_opened": False, "source_incidence_opened": False,
        "gross9_rows_opened": False, "singleton": True,
        "mechanism": {
            "claim": (
                "During volatile BTC trading, migration of the quote-volume-weighted median one-minute "
                "bar value price from the first four hours of a completed block to the second four "
                "hours identifies a durable shift in the auction's accepted-value center. Follow an "
                "upper-tail value-center migration for eight elapsed hours."
            ),
            "side": "strict sign of log(second-half weighted median / first-half weighted median)",
            "why_distinct": (
                "Close-VWAP candidates use endpoint displacement from one block mean; dominant-volume "
                "candidates select one bar; return-volume covariance measures directional participation; "
                "histogram classifiers fit labeled models. HVVWMMR compares two deterministic weighted "
                "medians of bar value prices and uses no endpoint direction, fitted labels, prior event, or control."
            ),
            "why_suited_to_volatile_regimes": "prior-24-hour BTC realized variation must rank in its causal upper 35%",
            "why_low_gross9_overlap_is_plausible": "sparse accepted-value migrations at fixed eight-hour boundaries are absent from Gross9",
        },
        "features": {
            "decision_grid": "exact 00:00/08:00/16:00 UTC boundaries",
            "block": "480 exact unique coherent BTCUSDT one-minute rows [D-8h,D); no imputation",
            "bar_value_price": "quote_asset_volume/volume for every minute; both volumes finite and strictly positive",
            "halves": "first 240 and final 240 minute rows, fixed by timestamp",
            "weighted_median": "sort each half by bar_value_price ascending and choose the lowest price whose cumulative quote_asset_volume is >= half its strictly positive total",
            "value_migration": "log(second-half weighted median/first-half weighted median), strict nonzero",
            "btc_realized_variation": "sqrt(sum squared exact one-minute log(close/open) returns over [D-24h,D))",
            "causal_ranks": "strict-prior midranks of absolute value_migration and variation over at most 270 source-valid boundaries, minimum 180, current excluded",
            "eligibility": "absolute migration rank>=0.75 and variation rank>=0.65",
        },
        "clock": {
            "decision": "completed fixed eight-hour boundary", "entry": "exact BTCUSDT D+5m open",
            "hold": "8 elapsed hours", "reservation": "global half-open; exit first on an equal open",
            "split_crossing_action": "skip", "gross_exposure": 0.5,
            "funding": "not a signal input; exact realized funding cash only after novelty passes",
            "oi_premium_rv20": "not signal inputs; RV20 q90 only after all economic stages pass",
        },
        "policy": {
            "block_minutes": 480, "half_minutes": 240, "history_boundaries": 270,
            "minimum_history_boundaries": 180, "migration_rank_min": 0.75,
            "variation_rank_min": 0.65, "entry_delay_minutes": 5, "hold_hours": 8,
            "leverage": 0.5, "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        "stages": {
            "train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
        },
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.20, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.10, "candidate_near_6h_share_max": 0.35, "occupied_5m_bar_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {
            "absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0,
            "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0,
            "weekly_signflip_one_sided_p_max": 0.10, "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True,
            "stop_on_first_failure": True, "future_can_rank_repair_or_reselect": False,
            "accounting": "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR",
        },
        "post_stage_volatility_audit": {"prerequisite": "unchanged candidate passes train, test, eval, and final", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "diagnostic_controls": {"names": ["no_migration_rank_gate", "no_volatility_gate", "arithmetic_vwap_migration", "one_boundary_stale_migration", "direction_flip"], "diagnostic_controls_cannot_be_promoted": True},
        "source_plan": {
            "btc": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "high", "low", "close", "volume", "quote_asset_volume"], "window": ["2023-04-01T00:00:00Z", "2026-08-01T00:00:00Z"], "read_only_after_preregistration": True},
            "funding_rate_values": "sealed during source support", "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_vwap_volume_histogram_family_outcomes_known": True,
            "prior_event_sets_or_controls_reused": False,
            "exact_weighted_median_migration_candidate_found_in_repository": False,
            "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False, "candidate_count": 1, "grid": False,
            "repair_of_prior_candidate": False, "promoted_prior_control": False,
            "selection_basis": "independent two-half accepted-value-center migration mechanism",
        },
        "stopping_rule": "Terminal first-failure sequence: source support, Gross9 novelty, strict economics; no median definition, half, rank, side, hold, clock, subset, or control repair.",
    }
    return {**core, "manifest_hash": canonical_hash(core)}

def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core) or payload != build():
        raise RuntimeError("HVVWMMR preregistration drift")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    result = build(); validate(result); args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n"); print(args.output)
