"""Outcome-blind preregistration for HVSCTBA-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVSCTBA-8"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_spot_conditioned_transmission_beta_asymmetry_relay_"
    "preregistration_2026-08-13.json"
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
        "protocol_version": "high_volatility_spot_conditioned_transmission_beta_asymmetry_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-13",
        "singleton": True,
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "mechanism": {
            "claim": (
                "Within a completed high-variation BTC block, unequal perpetual return "
                "transmission from positive versus negative spot-return minutes reveals "
                "directional derivatives amplification. Extreme downside amplification "
                "is continued short and extreme upside amplification is continued long."
            ),
            "side": (
                "negative strict sign of log(down-spot transmission beta / up-spot "
                "transmission beta)"
            ),
            "why_distinct": (
                "HVPIAR compares one-venue return per quote turnover by return sign. "
                "HVCKIHR compares unconditional through-origin aggressive-flow impact "
                "between venues. HVSPER estimates error-correction adjustment loadings. "
                "HVSCTBA instead estimates two spot-conditioned cross-venue return "
                "transmission slopes and uses neither turnover, flow, basis, nor levels."
            ),
            "why_suited_to_volatile_regimes": (
                "directional transmission asymmetry is admitted only in the causal upper "
                "tail of completed perpetual path variation"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "Gross9 has no spot-sign-conditioned perpetual transmission primitive"
            ),
        },
        "features": {
            "decision_grid": "exact 00:00, 08:00, and 16:00 UTC boundaries D",
            "block": (
                "480 exact aligned coherent one-minute BTCUSDT spot and perpetual rows "
                "in [D-8h,D), with no missing or imputed row"
            ),
            "minute_return": (
                "close-to-close log return within the aligned block; the first return uses "
                "the exact close at D-8h-1m from each venue"
            ),
            "up_spot_beta": (
                "sum(r_spot*r_perpetual)/sum(r_spot^2) over strict-positive spot-return "
                "minutes"
            ),
            "down_spot_beta": (
                "sum(r_spot*r_perpetual)/sum(r_spot^2) over strict-negative spot-return "
                "minutes"
            ),
            "support": (
                "at least 120 strict-positive and 120 strict-negative spot-return minutes; "
                "both denominators and both transmission betas finite and strictly positive"
            ),
            "transmission_asymmetry": (
                "log(down_spot_beta/up_spot_beta), strict nonzero"
            ),
            "asymmetry_rank": (
                "strict-prior midrank of abs(transmission_asymmetry) over at most 270 "
                "source-valid blocks, minimum 180, current excluded; rank>=0.80"
            ),
            "variation": "sqrt(sum squared completed perpetual minute returns)",
            "variation_rank": (
                "strict-prior midrank over at most 270 source-valid blocks, minimum 180, "
                "current excluded; rank>=0.65"
            ),
            "onset": (
                "the full asymmetry and variation rule is eligible now and was ineligible "
                "at the immediately prior exact source-valid boundary; missing prior cannot trigger"
            ),
            "no_imputation": True,
        },
        "clock": {
            "decision": "completed eight-hour boundary D",
            "entry": "exact BTCUSDT perpetual D+5m open",
            "hold": "8 elapsed hours",
            "side": "negative transmission-asymmetry sign",
            "reservation": "natural global nonoverlap; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty",
        },
        "policy": {
            "block_minutes": 480,
            "minimum_minutes_each_spot_sign": 120,
            "history_blocks": 270,
            "minimum_history_blocks": 180,
            "asymmetry_rank_min": 0.80,
            "variation_rank_min": 0.65,
            "onset_required": True,
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
                "fixed quantity, exact funding, 6bp base and 10bp stress per notional "
                "side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "post_stage_volatility_audit": {
            "prerequisite": "unchanged candidate passes all stages",
            "rv20_q90_entry_filter": False,
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "identical_clock_forced_long_residual_positive": True,
        },
        "diagnostic_controls": {
            "definitions": {
                "no_asymmetry_tail": "transmission side plus variation and onset without tail rank",
                "no_variation_gate": "asymmetry tail and onset only",
                "no_onset": "all tail-qualified high-variation blocks",
                "one_block_stale_asymmetry": "prior block asymmetry with current variation",
                "direction_flip": "positive primary asymmetry sign",
                "same_clock_forced_long": "side +1 on primary clock",
            },
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "perpetual": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m"},
            "spot": {"table": "bars_binance_spot", "symbol": "BTCUSDT", "interval": "1m"},
            "columns": ["ts", "open", "high", "low", "close"],
            "window": ["2023-03-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "read_after_preregistration_commit": True,
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_spot_perpetual_and_impact_family_outcomes_known": True,
            "repository_spot_sign_conditioned_transmission_candidate_found": False,
            "prior_event_sets_or_controls_reused": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent cross-venue conditional-transmission mechanism",
        },
        "stopping_rule": (
            "terminal first failure; no venue, return, beta, sign support, rank, variation, "
            "onset, side, clock, hold, subset, threshold, comparator, or control repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVSCTBA preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(args.output)
