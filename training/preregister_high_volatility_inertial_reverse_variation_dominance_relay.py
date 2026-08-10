"""Outcome-blind preregistration for HVIRVD-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVIRVD-8"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_inertial_reverse_variation_dominance_relay_"
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
        "protocol_version": "high_volatility_inertial_reverse_variation_dominance_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-10",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "A completed high-variation BTC path can be decomposed into squared return mass "
                "whose sign agrees with the immediately preceding return (inertial variation) and "
                "mass whose sign reverses (reverse variation). When one component causally dominates, "
                "follow the completed path under inertial dominance and fade it under reverse dominance "
                "for eight elapsed hours."
            ),
            "side": (
                "strict sign of the completed eight-hour return multiplied by the strict sign of "
                "inertial variation minus reverse variation"
            ),
            "why_distinct": (
                "HVIRVD partitions squared five-minute return mass by agreement with the immediately "
                "previous return sign. Serial autocorrelation sums signed return products; variance "
                "ratio compares sampling scales; sign-run candidates count run geometry; semivariance "
                "partitions returns by sign against zero. HVIRVD uses no volume, flow, funding, OI, "
                "cross-assets, prior event rows, fitted outcomes, or prior controls."
            ),
            "why_suited_to_volatile_regimes": (
                "completed realized variation must occupy its causal upper regime, while absolute "
                "inertial-versus-reverse dominance must independently occupy its causal upper quartile"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "three fixed UTC sign-transition variation onsets are absent from Gross9 primitives"
            ),
        },
        "research_basis": {
            "primary_reference": (
                "Luo, Tao, and Zou (2022), A new measure of realized volatility: Inertial and "
                "reverse realized semivariance, Finance Research Letters 47, 102658"
            ),
            "doi": "10.1016/j.frl.2021.102658",
            "outcome_blind_use": (
                "The paper defines inertial and reverse realized variation from consecutive return "
                "sign agreement and reports return-predictive content; no repository incidence or "
                "post-entry outcome was used to choose this singleton rule."
            ),
        },
        "features": {
            "decision_grid": "exact 00:00/08:00/16:00 UTC boundaries",
            "block": "97 exact coherent BTCUSDT five-minute closes in [D-8h,D]",
            "five_minute_returns": "96 values r_i=log(close_i/close_(i-1))",
            "transition_pairs": "95 consecutive pairs (r_(i-1),r_i); zero products are excluded",
            "inertial_variation": "sum r_i squared where r_i*r_(i-1)>0",
            "reverse_variation": "sum r_i squared where r_i*r_(i-1)<0",
            "dominance": "inertial variation minus reverse variation, strict nonzero",
            "dominance_share": (
                "abs(dominance)/(inertial variation+reverse variation), strict positive denominator"
            ),
            "completed_return": "log(last close/first close), finite strict nonzero",
            "realized_variation": "sqrt(sum of all 96 r_i squared), finite strict positive",
            "causal_ranks": (
                "strict-prior midranks of dominance share and realized variation over at most 270 "
                "earlier source-valid blocks, minimum 180; current excluded"
            ),
            "eligibility": "dominance-share rank>=0.75 and variation rank>=0.65",
            "onset": (
                "eligible now and immediately prior exact source-valid block ineligible; missing prior "
                "opportunity cannot trigger"
            ),
            "no_imputation": True,
        },
        "clock": {
            "decision": "completed eight-hour boundary D",
            "entry": "exact BTCUSDT perpetual D+5m open",
            "side": "sign(completed return)*sign(inertial variation-reverse variation)",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        "policy": {
            "close_observations": 97,
            "return_observations": 96,
            "transition_observations": 95,
            "history_blocks": 270,
            "minimum_history_blocks": 180,
            "dominance_share_rank_min": 0.75,
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
                "no_dominance_share_gate",
                "no_variation_gate",
                "one_block_stale_features",
                "dominance_interpretation_flip",
                "direction_flip",
                "forced_long",
            ],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "bars": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "5m",
                "columns": ["ts", "open", "high", "low", "close"],
                "window": ["2023-04-01T00:00:00Z", "2026-08-01T00:00:00Z"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_price_transition_family_outcomes_known": True,
            "repository_inertial_reverse_realized_variation_candidate_found": False,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_ranks_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "published inertial-versus-reverse realized-variation decomposition",
        },
        "stopping_rule": (
            "Terminal first-failure sequence: source support, Gross9 novelty, strict economics; no "
            "formula, rank, side, hold, clock, subset, comparator, or control repair."
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVIRVD preregistration drift")


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
