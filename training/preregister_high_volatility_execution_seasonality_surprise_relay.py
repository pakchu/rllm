"""Outcome-blind preregistration for HVESSR-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVESSR-8"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_execution_seasonality_surprise_relay_preregistration_2026-08-13.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_execution_seasonality_surprise_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-13",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "During elevated BTC variation, an unusually large completed execution-count "
                "surprise relative to the same UTC block in the prior eight weeks identifies "
                "seasonality-adjusted participation sponsorship. Follow the completed price "
                "direction for eight hours at the fresh surprise-tail onset."
            ),
            "side": "strict sign of the completed eight-hour BTC return",
            "why_distinct": (
                "Absolute execution count ignores clock seasonality; HVECE normalizes endpoint "
                "displacement by square-root count; HVTCDAR compares direction-conditioned Fano "
                "ratios; backloading and count-memory candidates use within-block timing. HVESSR "
                "compares aggregate count only with eight strictly prior same-weekday/same-UTC-slot "
                "blocks and reuses no prior event or control."
            ),
            "volatile_market_target": (
                "both completed realized variation and seasonality-adjusted execution surprise "
                "must enter independent causal upper tails"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "same-weekday execution-surprise onsets on three daily blocks are absent from Gross9"
            ),
        },
        "features": {
            "decision_grid": "exact 00:00/08:00/16:00 UTC boundaries D",
            "block": "480 exact coherent BTCUSDT bars_binance one-minute rows [D-8h,D)",
            "execution_count": (
                "sum integer nonnegative number_of_trades over 480 rows, strict positive"
            ),
            "seasonal_reference": (
                "median execution_count of exactly D-7d through D-56d at the identical UTC block; "
                "all eight exact source-valid references required; current excluded"
            ),
            "execution_surprise": (
                "log(execution_count/seasonal_reference), finite; tail statistic is its absolute value"
            ),
            "completed_return": "log(last close/first open), finite strict nonzero",
            "realized_variation": (
                "sum squared log(close/open) minute returns, finite strict positive"
            ),
            "surprise_rank": (
                "strict-prior midrank of abs(execution_surprise) over at most 270 valid blocks, "
                "minimum 180, current excluded; rank>=0.80"
            ),
            "variation_rank": (
                "strict-prior midrank over at most 270 valid blocks, minimum 180, current excluded; "
                "rank>=0.65"
            ),
            "eligible_state": "surprise and variation gates pass",
            "onset": (
                "eligible now and immediately previous exact source-valid decision block ineligible"
            ),
            "no_imputation": True,
        },
        "clock": {
            "entry": "D+5m BTCUSDT open",
            "side": "sign completed_return",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not an input; exact settlements only after novelty",
        },
        "policy": {
            "seasonal_weeks": 8,
            "prior_blocks": 270,
            "minimum_prior_blocks": 180,
            "surprise_rank_min": 0.80,
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
            "minority_side_share_min": 0.20,
            "max_month_share": 0.45,
        },
        "novelty_gates": {
            "exact_entry_jaccard_max": 0.10,
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
            "weekly_signflip_one_sided_p_max": 0.10,
            "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5,
            "each_calendar_half_positive": True,
            "stop_on_first_failure": True,
            "accounting": (
                "fixed quantity, exact funding, 6bp/10bp per notional side, held 5m favorable "
                "then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "post_stage_volatility_audit": {
            "prerequisite": "unchanged all-stage pass",
            "rv20_q90_entry_filter": False,
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "identical_clock_forced_long_residual_positive": True,
        },
        "diagnostic_controls": {
            "names": [
                "no_surprise_tail",
                "no_variation_gate",
                "absolute_count_tail",
                "one_block_stale_geometry",
                "direction_flip",
                "forced_long",
            ],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "bars": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close", "number_of_trades"],
                "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source and novelty pass",
        },
        "research_boundary": {
            "prior_count_level_elasticity_dispersion_backloading_and_memory_outcomes_known": True,
            "repository_same_weekday_execution_surprise_candidate_found": False,
            "prior_event_sets_or_controls_reused": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent seasonality-adjusted execution participation shock",
        },
        "stopping_rule": (
            "terminal first failure; no seasonal reference, rank, onset, side, clock, hold, subset, "
            "threshold, or control repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVESSR preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build()
    validate(payload)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(args.output)
