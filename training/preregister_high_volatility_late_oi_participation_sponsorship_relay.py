"""Outcome-blind preregistration for HVLOIPSR-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVLOIPSR-8"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_late_oi_participation_sponsorship_relay_"
    "preregistration_2026-08-10.json"
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
        "protocol_version": "high_volatility_late_oi_participation_sponsorship_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-10",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "In a high-variation completed eight-hour BTC auction, unusually large gross "
                "open-interest replacement that becomes concentrated in the final two hours "
                "represents late leveraged participation rather than an old position stock. "
                "Follow the completed BTC direction for eight elapsed hours."
            ),
            "side": "strict sign of the completed eight-hour BTC return",
            "why_distinct": (
                "HVLOIPSR measures when gross absolute OI changes arrive inside a fixed completed "
                "block. HVOILSR measured OI-change lead correlation with next-bar returns; GOICR "
                "measured net cancellation relative to gross OI churn; purge, expansion, and "
                "divergence candidates used endpoint OI direction. This candidate uses neither "
                "OI direction nor OI-price correlation, funding, premium, flow, liquidation, "
                "cross-assets, a fitted model, prior event rows, or a prior control."
            ),
            "why_suited_to_volatile_regimes": (
                "both completed BTC variation and gross OI activity must occupy causal upper regimes"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "three fixed UTC clocks conditioned on within-block OI-arrival concentration are "
                "absent from Gross9 primitives"
            ),
        },
        "features": {
            "decision_grid": "exact 00:00/08:00/16:00 UTC boundaries",
            "price_block": (
                "96 exact coherent BTCUSDT five-minute bars aggregated from 480 unique one-minute "
                "rows in [D-8h,D)"
            ),
            "oi_path": (
                "97 exact positive BTCUSDT period=5m raw observations at D-8h,D-7h55m,...,D; "
                "the observation stamped D must be available by the D+5m entry; no snapping or imputation"
            ),
            "oi_changes": "96 values d_i=log(OI_i/OI_(i-1))",
            "gross_oi_activity": "sum(abs(d_i)) across all 96 completed changes; strict positive",
            "late_oi_activity_share": (
                "sum(abs(d_i)) for the final 24 changes ending in (D-2h,D] divided by gross OI activity"
            ),
            "realized_variation": (
                "sqrt(sum squared five-minute log(close/open) returns) across the 96 completed bars"
            ),
            "completed_return": "log(last five-minute close/first five-minute open), strict nonzero",
            "causal_ranks": (
                "strict-prior midranks of gross OI activity, late OI activity share, and realized "
                "variation over at most 270 earlier source-valid blocks, minimum 180; current excluded"
            ),
            "eligibility": (
                "gross-OI-activity rank>=0.60, late-share rank>=0.75, and variation rank>=0.65"
            ),
            "onset": (
                "eligible now and immediately prior exact source-valid block ineligible; a missing "
                "prior opportunity cannot trigger"
            ),
            "no_imputation": True,
        },
        "clock": {
            "decision": "completed eight-hour boundary D",
            "feature_availability": (
                "all price rows end before D and the final raw OI observation has timestamp<=D and "
                "is opened only before the exact D+5m entry"
            ),
            "entry": "exact BTCUSDT perpetual D+5m open",
            "side": "sign of completed eight-hour return",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        "policy": {
            "price_bars": 96,
            "oi_points": 97,
            "oi_changes": 96,
            "late_oi_changes": 24,
            "history_blocks": 270,
            "minimum_history_blocks": 180,
            "gross_oi_activity_rank_min": 0.60,
            "late_oi_activity_share_rank_min": 0.75,
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
                "fixed quantity, exact funding, 6bp base and 10bp stress per notional side, every "
                "held 5m favorable then adverse, global HWM, full-calendar CAGR"
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
            "names": [
                "no_late_oi_activity_share_gate",
                "no_gross_oi_activity_gate",
                "no_variation_gate",
                "one_block_stale_features",
                "direction_flip",
                "forced_long",
            ],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "oi": {
                "table": "open_interest_binance",
                "symbol": "BTCUSDT",
                "period": "5m",
                "columns": ["ts", "sum_open_interest"],
                "read_after_preregistration": True,
            },
            "bars": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close"],
                "read_after_preregistration": True,
            },
            "window": ["2023-04-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "database_metadata_only_opened_before_preregistration": True,
            "prior_oi_family_outcomes_known": True,
            "repository_fixed_block_late_oi_participation_candidate_found": False,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_ranks_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "within-block timing of gross leveraged participation",
        },
        "stopping_rule": (
            "Terminal first-failure sequence: source support, Gross9 novelty, strict economics; "
            "no formula, rank, side, hold, clock, subset, comparator, or control repair."
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVLOIPSR preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(args.output)
