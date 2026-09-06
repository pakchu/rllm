"""Outcome-blind preregistration for HVPPLA-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVPPLA-8"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_premium_to_price_lead_asymmetry_relay_preregistration_2026-08-10.json"
)


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_premium_to_price_lead_asymmetry_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-10",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "Within a completed volatile eight-hour BTC block, a premium-index change that "
                "predictively rank-aligns with the next minute's BTC return more strongly than BTC "
                "returns align with the next premium change identifies derivatives-led information "
                "transmission. When completed premium displacement and BTC displacement agree, "
                "follow their common direction for the next eight hours."
            ),
            "side": "common strict sign of premium-index displacement and BTC block return",
            "why_distinct": (
                "HVPCR uses only opposite-sign eight-hour premium and BTC displacement; HVPASR uses "
                "unsigned daily premium total variation; HVPFPR uses final-hour mean premium pressure; "
                "HVFPSC uses actual funding values and a fitted classifier. HVPPLA's primary object is "
                "the directional one-minute premium-to-next-price lead advantage over the reverse lag."
            ),
            "why_suited_to_volatile_regimes": (
                "the completed prior-24-hour BTC realized variation must rank in its causal upper 35%"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "sparse cross-lag transmission tails at three settlement boundaries are absent from "
                "Gross9 primitives"
            ),
        },
        "features": {
            "decision_grid": "exact 00:00/08:00/16:00 UTC boundaries",
            "joined_block": (
                "480 exact timestamp-aligned unique completed BTCUSDT perpetual and premium-index "
                "one-minute OHLC rows in [D-8h,D); positive coherent BTC OHLC, finite coherent signed "
                "premium OHLC; no imputation"
            ),
            "btc_minute_return": "r_i=log(BTC close_i/BTC open_i)",
            "premium_change": "dp_i=premium close_i-premium close_(i-1)",
            "premium_lead": (
                "Spearman(dp_i,r_(i+1)) for i=1..478 using average-tied ranks; 478 pairs; finite "
                "positive result required"
            ),
            "price_lead": (
                "Spearman(r_i,dp_(i+1)) for i=1..478 using average-tied ranks; 478 pairs; finite"
            ),
            "lead_advantage": "premium_lead-price_lead; finite",
            "premium_displacement": "last premium close-first premium close; strict nonzero",
            "btc_return": "log(last BTC close/first BTC open); strict nonzero",
            "direction_alignment": "strict signs of premium_displacement and btc_return agree",
            "btc_realized_variation": (
                "sqrt(sum squared exact BTC one-minute log(close/open) returns over [D-24h,D))"
            ),
            "causal_ranks": (
                "strict-prior midranks over at most 270 prior source-valid boundaries, minimum 180, "
                "current excluded"
            ),
            "eligibility": (
                "premium_lead>0, lead_advantage_rank>=0.75, absolute premium-displacement "
                "rank>=0.60, BTC realized-variation rank>=0.65, and direction alignment"
            ),
            "actual_funding_value": "not read or used",
        },
        "clock": {
            "decision": "exact completed eight-hour boundary after every required source minute",
            "entry": "exact BTCUSDT D+5m open",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on an equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact realized funding cash only after novelty passes",
            "oi_rv20": "not signal inputs; RV20 q90 only after all economic stages pass",
        },
        "policy": {
            "block_minutes": 480,
            "lead_pairs": 478,
            "history_boundaries": 270,
            "minimum_history_boundaries": 180,
            "lead_advantage_rank_min": 0.75,
            "premium_displacement_rank_min": 0.60,
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
            "future_can_rank_repair_or_reselect": False,
            "accounting": (
                "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, "
                "every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "post_stage_volatility_audit": {
            "prerequisite": "unchanged candidate passes train, test, eval, and final",
            "rv20_q90_entry_filter": False,
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "identical_clock_forced_long_residual_positive": True,
        },
        "diagnostic_controls": {
            "names": [
                "no_lead_advantage_gate",
                "no_premium_displacement_tail",
                "no_volatility_gate",
                "one_block_stale_premium",
                "direction_flip",
            ],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        "source_plan": {
            "btc": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close"],
            },
            "premium": {
                "table": "bars_binance_premium",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close"],
            },
            "window": ["2023-04-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "read_only_after_preregistration": True,
            "funding_rate_values": "sealed during source support",
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_premium_family_outcomes_known": True,
            "prior_hvpfpr_source_incidence_known": True,
            "prior_event_sets_or_controls_reused": False,
            "exact_cross_lag_candidate_found_in_repository": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent premium-to-next-price lead-asymmetry mechanism",
        },
        "stopping_rule": (
            "Terminal first-failure sequence: source support, Gross9 novelty, strict economics; "
            "no lag, formula, rank, side, hold, clock, subset, or control repair."
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core) or payload != build():
        raise RuntimeError("HVPPLA preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)
