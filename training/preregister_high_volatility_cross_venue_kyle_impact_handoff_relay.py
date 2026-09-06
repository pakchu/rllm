"""Outcome-blind preregistration for HVCKIHR-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVCKIHR-8"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_cross_venue_kyle_impact_handoff_relay_preregistration_2026-08-10.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_cross_venue_kyle_impact_handoff_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-10",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "When the same signed aggressive-flow fraction produces materially greater positive "
                "price response in Binance BTC spot than in its perpetual during a high-variation "
                "completed block, unlevered cash price discovery is leading derivatives. Follow the "
                "common spot/perpetual flow direction for eight elapsed hours."
            ),
            "side": "common strict sign of aggregate spot and perpetual signed aggressive quote flow",
            "why_distinct": (
                "HVCKIHR compares two deterministic through-origin flow-response slopes. Existing spot "
                "leadership candidates compare flow magnitude, volume participation, return magnitude, "
                "or error correction; BTC impact candidates use one venue or return per turnover. The "
                "candidate uses no outcome fit, prior event set, funding, OI, premium, or control."
            ),
            "why_suited_to_volatile_regimes": (
                "the causal variation tail restricts activation to volatile blocks while spot impact "
                "leadership identifies the venue bearing marginal information risk"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "three fixed UTC cross-venue impact-handoff onsets are absent from Gross9 primitives"
            ),
        },
        "features": {
            "decision_grid": "exact 00:00/08:00/16:00 UTC boundaries",
            "block": "480 exact aligned coherent one-minute BTCUSDT rows [D-8h,D) from spot and perpetual",
            "five_minute_aggregation": "96 exact left-labeled groups of five consecutive source rows per venue",
            "bar_flow": "x=(2*taker_buy_quote-quote_asset_volume)/quote_asset_volume; denominator strict positive",
            "bar_return": "r=log(five-minute close/five-minute open)",
            "venue_impact": "beta=sum(x*r)/sum(x^2), requiring strict-positive finite denominator and beta",
            "impact_handoff": "log(beta_spot/beta_perpetual), strict positive",
            "flow_consensus": "aggregate signed aggressive quote flow is strict nonzero and has one sign across both venues",
            "handoff_rank": "strict-prior midrank over at most 270 valid blocks, minimum 180, current excluded; rank>=0.80",
            "realized_variation": "sqrt(sum squared perpetual five-minute bar returns), strict positive",
            "variation_rank": "strict-prior midrank over at most 270 valid blocks, minimum 180, current excluded; rank>=0.65",
            "onset": "eligible now and immediately prior exact valid block ineligible; missing prior cannot trigger",
            "no_imputation": True,
        },
        "clock": {
            "decision": "completed eight-hour boundary",
            "entry": "exact BTCUSDT perpetual D+5m open",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        "policy": {
            "history_blocks": 270,
            "minimum_history_blocks": 180,
            "handoff_rank_min": 0.80,
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
            "names": [
                "no_handoff_tail", "no_variation_gate", "perpetual_impact_dominance",
                "one_boundary_stale_handoff", "direction_flip",
            ],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "perpetual": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m"},
            "spot": {"table": "bars_binance_spot", "symbol": "BTCUSDT", "interval": "1m"},
            "columns": ["ts", "open", "high", "low", "close", "quote_asset_volume", "taker_buy_quote"],
            "window": ["2023-04-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "read_after_preregistration": True,
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_spot_flow_and_single_venue_impact_family_outcomes_known": True,
            "repository_cross_venue_through_origin_impact_ratio_candidate_found": False,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_rank_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "cross-venue marginal flow-impact handoff mechanism",
        },
        "stopping_rule": (
            "Terminal first-failure sequence: source support, Gross9 novelty, strict economics; "
            "no venue, impact formula, rank, side, hold, clock, subset, or control repair."
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core) or payload != build():
        raise RuntimeError("HVCKIHR preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)
