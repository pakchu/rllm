"""Outcome-blind preregistration for HVRTCR-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVRTCR-8"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_range_turnover_coupling_relay_preregistration_2026-08-10.json"
)


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_range_turnover_coupling_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-10",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": "When five-minute range energy and quote turnover become unusually tightly coupled throughout a completed high-variation BTC block, the auction is converting participation into price discovery rather than generating isolated low-liquidity excursions. Follow the completed block direction for eight elapsed hours.",
            "side": "strict sign of the completed eight-hour return",
            "why_distinct": "HVRTCR correlates unsigned Parkinson range energy with log quote turnover across all constituent auctions. HVRPPR signs and aggregates their product, HVVRCR correlates signed returns with volume, RBEFR divides aggregate range variance by body variance, and HVDVBR selects one maximum-volume bar. No prior event set, control, fitted outcome, funding, OI, premium, or cross-asset input is reused.",
            "why_suited_to_volatile_regimes": "the causal variation tail selects volatile blocks while an upper-tail range-turnover coupling state requires broad participation in the realized intrabar travel",
            "why_low_gross9_overlap_is_plausible": "offset three-daily range-turnover-coupling onsets with eight-hour reservation are absent from Gross9 primitives",
        },
        "features": {
            "decision_grid": "exact 04:00/12:00/20:00 UTC boundaries",
            "block": "96 exact coherent BTCUSDT perpetual five-minute aggregates from 480 unique one-minute rows [D-8h,D)",
            "bar_turnover": "sum quote_asset_volume in each group, finite nonnegative",
            "turnover_coordinate": "log1p(bar_turnover)",
            "range_energy": "log(five-minute high/five-minute low)^2/(4*log(2)), finite nonnegative",
            "range_turnover_coupling": "sample Pearson correlation of range_energy and turnover_coordinate across 96 contemporaneous bars; both sample standard deviations strict positive; correlation strict positive",
            "coupling_rank": "strict-prior midrank over at most 270 valid blocks, minimum 180, current excluded; rank>=0.75",
            "realized_variation": "sqrt(sum squared five-minute close/open log returns), strict positive",
            "variation_rank": "strict-prior midrank over at most 270 valid blocks, minimum 180, current excluded; rank>=0.65",
            "block_return": "log(last close/first open), strict nonzero",
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
            "coupling_rank_min": 0.75,
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
                "no_coupling_tail",
                "no_variation_gate",
                "range_only_direction",
                "one_boundary_stale_coupling",
                "direction_flip",
            ],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "bars": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close", "quote_asset_volume"],
                "window": ["2023-04-01T00:00:00Z", "2026-08-01T00:00:00Z"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_range_participation_return_volume_range_body_and_dominant_volume_outcomes_known": True,
            "repository_unsigned_range_turnover_coupling_candidate_found": False,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_rank_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "participation-to-range conversion mechanism",
        },
        "stopping_rule": "Terminal first failure; no range formula, coupling, rank, direction, hold, clock, subset, or control repair.",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core) or payload != build():
        raise RuntimeError("HVRTCR preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)
