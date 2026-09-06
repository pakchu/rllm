"""Outcome-blind preregistration for HVTER-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVTER-8"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_ticket_elasticity_sponsorship_relay_preregistration_2026-08-10.json"
)


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_ticket_elasticity_sponsorship_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-10",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": "When minute quote turnover rises more than proportionally with reported execution count throughout a completed volatile BTC block, active periods are being sponsored by larger tickets rather than only more small prints. At a new upper-tail elasticity state, follow the completed block direction for eight elapsed hours.",
            "side": "strict sign of the completed eight-hour return, requiring the final two-hour return to agree",
            "why_distinct": "HVTER estimates the block-wide log elasticity of quote turnover to reported execution count. Small-ticket exhaustion uses one aggregate count-to-turnover ratio, directional ticket asymmetry conditions median ticket size on return sign, count-dispersion and count-memory candidates use executions without turnover, and HVRTCR couples turnover to price range rather than executions. No prior event set or control is reused.",
            "why_suited_to_volatile_regimes": "the causal variation tail selects volatile blocks while positive upper-tail excess elasticity identifies large-ticket sponsorship inside active minutes",
            "why_low_gross9_overlap_is_plausible": "offset three-daily onsets from execution-count-to-turnover elasticity are absent from Gross9 primitives",
        },
        "features": {
            "decision_grid": "exact 06:00/14:00/22:00 UTC boundaries",
            "block": "480 exact coherent bars_binance BTCUSDT perpetual one-minute rows [D-8h,D)",
            "count_coordinate": "log1p(number_of_trades), with finite nonnegative integer counts and strict positive sample variance",
            "turnover_coordinate": "log1p(quote_asset_volume), finite nonnegative",
            "ticket_elasticity": "OLS slope cov(count_coordinate,turnover_coordinate)/var(count_coordinate) across 480 contemporaneous minutes; finite and strictly greater than one",
            "elasticity_rank": "strict-prior midrank of ticket_elasticity over at most 270 valid blocks, minimum 180, current excluded; rank>=0.75",
            "realized_variation": "sum squared one-minute log(close/open) returns, finite strict positive",
            "variation_rank": "strict-prior midrank over at most 270 valid blocks, minimum 180, current excluded; rank>=0.65",
            "block_return": "log(last close/first open), finite strict nonzero",
            "final_two_hour_return": "log(last close/first open) over the final 120 completed minutes, finite strict nonzero and same sign as block_return",
            "eligible_state": "elasticity and variation gates pass with directional agreement",
            "onset": "eligible now and immediately prior exact source-valid block ineligible; missing prior cannot trigger",
            "no_imputation": True,
        },
        "clock": {
            "decision": "completed eight-hour boundary",
            "entry": "exact BTCUSDT D+5m open",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        "policy": {
            "history_blocks": 270,
            "minimum_history_blocks": 180,
            "elasticity_floor": 1.0,
            "elasticity_rank_min": 0.75,
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
            "occupied_5m_jaccard_max": 0.25,
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
            "accounting": "fixed quantity, exact funding, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR",
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
                "no_elasticity_tail",
                "no_variation_gate",
                "aggregate_average_ticket_tail",
                "one_boundary_stale_elasticity",
                "direction_flip",
            ],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "bars": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close", "quote_asset_volume", "number_of_trades"],
                "window": ["2023-04-01T00:00:00Z", "2026-08-01T00:00:00Z"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_ticket_count_turnover_range_and_path_outcomes_known": True,
            "repository_ticket_elasticity_candidate_found": False,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_rank_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "large-ticket sponsorship through count-to-turnover elasticity",
        },
        "stopping_rule": "Terminal first failure; no elasticity formula, floor, rank, direction, hold, clock, subset, or control repair.",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core) or payload != build():
        raise RuntimeError("HVTER preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)
