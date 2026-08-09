"""Outcome-blind preregistration for HVSAR-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVSAR-12"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_serial_autocorrelation_relay_preregistration_2026-08-09.json"
)


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_serial_autocorrelation_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "In an unusually volatile completed twelve-hour BTC path, an extreme positive "
                "lag-one correlation of five-minute returns indicates persistent order splitting, "
                "whereas an extreme negative correlation indicates alternating liquidity repair. "
                "The same completed displacement should therefore continue in the positive state "
                "and reverse in the negative state over the next twelve hours."
            ),
            "side": (
                "sign of the completed twelve-hour return multiplied by the strict sign of its "
                "lag-one five-minute return autocorrelation"
            ),
            "why_distinct": (
                "HVVRR measures fine-versus-coarse variance non-additivity and always fades; HVSER "
                "measures the unconditional sign distribution and always follows; HVCRMR learns from "
                "previous post-event responses. HVSAR uses only within-path ordered return products "
                "and lets their contemporaneously completed serial-dependence sign choose continuation "
                "or reversal. It reuses no terminal event set, control, threshold, or outcome."
            ),
            "why_suited_to_volatile_regimes": (
                "both completed realized variation and absolute serial dependence must be in causal "
                "upper tails"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "sparse twelve-hour serial-dependence tail clocks are absent from Gross9 primitives"
            ),
        },
        "features": {
            "decision_grid": "exact 00:00 and 12:00 UTC boundaries",
            "path": (
                "144 exact consecutive completed five-minute close-to-close log returns in "
                "[decision-12h,decision), requiring 145 contiguous finite positive closes"
            ),
            "completed_return": "sum of the 144 returns; strict nonzero",
            "realized_variation": "sqrt(sum of squared 144 returns)",
            "lag_one_autocorrelation": (
                "Pearson correlation between returns[1:] and returns[:-1]; both vectors must have "
                "strictly positive sample standard deviation"
            ),
            "causal_ranks": (
                "strict-prior midranks of realized variation and absolute lag-one autocorrelation "
                "over at most 270 earlier source-valid boundaries, minimum 252; current excluded"
            ),
            "eligibility": (
                "realized-variation rank>=0.70 and absolute-autocorrelation rank>=0.75; "
                "completed return and autocorrelation are strict nonzero"
            ),
            "no_imputation": True,
            "grid": False,
        },
        "clock": {
            "decision": "exact boundary after the entire twelve-hour path is complete",
            "entry": "exact BTCUSDT decision+5m open",
            "side": "sign(completed_return)*sign(lag_one_autocorrelation)",
            "hold": "12 elapsed hours",
            "reservation": "fixed opportunities are naturally half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "funding_oi_premium_implied_vol_rv20": (
                "not signal inputs; exact funding only after novelty; RV20 q90 only after all "
                "economic stages pass"
            ),
        },
        "policy": {
            "return_bars": 144,
            "prior_boundaries": 270,
            "prior_min_boundaries": 252,
            "variation_rank_min": 0.70,
            "absolute_autocorrelation_rank_min": 0.75,
            "entry_delay_minutes": 5,
            "hold_hours": 12,
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
                "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, "
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
                "no_variation_gate",
                "no_autocorrelation_tail_gate",
                "one_boundary_stale_features",
                "fixed_momentum",
                "fixed_reversal",
                "direction_flip",
            ],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        "source_plan": {
            "historical": "hash-bound completed BTCUSDT five-minute cache through 2026-06-01",
            "live_extension": "read-only Postgres completed bars through 2026-08-01",
            "columns": ["date", "open", "high", "low", "close"],
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_path_family_outcomes_known": True,
            "prior_candidate_events_or_controls_reused": False,
            "prior_candidate_outcomes_used_to_set_hvsar_rule": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "ordered-return serial-dependence mechanism",
        },
        "stopping_rule": (
            "Terminal first-failure sequence: source support, Gross9 novelty, strict economics; "
            "no path, rank, sign, hold, clock, subset, threshold, or control repair."
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVSAR preregistration hash mismatch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
