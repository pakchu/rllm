"""Outcome-blind preregistration for HVOIPSCR-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

POLICY_ID = "HVOIPSCR-8"
DEFAULT_OUTPUT = Path("results/high_volatility_oi_price_sign_concordance_relay_preregistration_2026-08-12.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_oi_price_sign_concordance_relay_v1",
        "policy_id": POLICY_ID, "as_of_date": "2026-08-12", "singleton": True,
        "outcomes_opened": False, "source_incidence_opened": False, "gross9_rows_opened": False,
        "mechanism": {
            "claim": (
                "During a volatile completed BTC auction, repeated same-direction five-minute price "
                "and open-interest changes indicate that leveraged inventory is being created and "
                "removed in synchrony with price discovery rather than merely changing at the block "
                "endpoint. At a fresh upper-tail sign-concordance state with substantial gross OI "
                "activity, follow the completed price displacement for eight elapsed hours."
            ),
            "side": "strict sign of the completed eight-hour BTC return",
            "why_distinct": (
                "HVOIPCSR correlates absolute OI-change and absolute price-return magnitudes; HVOILSR "
                "correlates signed OI changes with later returns; GOICR compares gross OI churn with "
                "net OI displacement; expansion, purge, divergence and reconciliation candidates use "
                "endpoint OI changes. HVOIPSCR discards magnitudes inside its primary statistic and "
                "counts contemporaneous bar-by-bar sign agreement. It uses no lead, endpoint OI sign "
                "gate, funding, premium, flow, fitted outcome, prior event set, or promoted control."
            ),
            "why_suited_to_volatile_regimes": (
                "both completed BTC variation and gross OI activity must occupy causal upper regimes"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "offset three-daily OI/price sign-concordance onsets are absent from Gross9 primitives"
            ),
        },
        "features": {
            "decision_grid": "exact 02:00, 10:00 and 18:00 UTC boundaries D",
            "price_path": "96 exact coherent five-minute BTCUSDT bars from 480 unique one-minute rows [D-8h,D)",
            "oi_path": "97 exact positive BTCUSDT period=5m observations from D-8h through D; timestamp<=D; no imputation",
            "price_return": "96 values r_i=log(five-minute close/open), finite",
            "oi_change": "96 values d_i=log(OI_i/OI_(i-1)), finite",
            "sign_pair": "sign(r_i)*sign(d_i); a zero in either member contributes zero",
            "sign_concordance": "count(sign_pair=+1)/count(sign_pair!=0), requiring at least 72 nonzero pairs",
            "gross_oi_activity": "sum(abs(d_i)), finite strict positive",
            "block_return": "log(last completed close/first completed open), finite strict nonzero",
            "realized_variation": "sqrt(sum squared r_i), finite strict positive",
            "causal_ranks": (
                "separate strict-prior midranks of sign concordance, gross OI activity and realized "
                "variation over at most 270 earlier jointly source-valid decisions, minimum 180, current excluded"
            ),
            "eligibility": "concordance rank>=0.80, gross-OI-activity rank>=0.60, variation rank>=0.65",
            "onset": "eligible now and immediately preceding exact source-valid boundary ineligible; missing prior cannot trigger",
            "no_imputation": True,
        },
        "clock": {
            "decision": "D after all completed price and OI inputs are available",
            "entry": "exact BTCUSDT perpetual D+5m open", "side": "sign(block_return)",
            "hold": "8 elapsed hours", "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip", "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
            "rv20": "q90 audit only after unchanged all-stage pass",
        },
        "policy": {
            "block_minutes": 480, "price_bars": 96, "oi_points": 97, "minimum_nonzero_pairs": 72,
            "history_decisions": 270, "minimum_history_decisions": 180,
            "concordance_rank_min": 0.80, "gross_oi_activity_rank_min": 0.60, "variation_rank_min": 0.65,
            "entry_delay_minutes": 5, "hold_hours": 8, "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006, "stress_cost_per_notional_side": 0.001,
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
            "absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0,
            "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.10,
            "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5,
            "each_calendar_half_positive": True, "stop_on_first_failure": True,
            "accounting": "fixed quantity, exact funding, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR",
        },
        "post_stage_volatility_audit": {"prerequisite": "unchanged train/test/eval/final pass", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "diagnostic_controls": {"names": ["no_concordance_gate", "no_gross_oi_activity_gate", "no_variation_gate", "one_block_stale_features", "direction_flip", "forced_long"], "cannot_be_promoted": True},
        "source_plan": {
            "bars": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "high", "low", "close"], "read_after_preregistration": True},
            "oi": {"table": "open_interest_binance", "symbol": "BTCUSDT", "period": "5m", "columns": ["ts", "sum_open_interest"], "read_after_preregistration": True},
            "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"], "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_oi_family_outcomes_known": True, "repository_oi_price_sign_concordance_candidate_found": False,
            "nearby_magnitude_coactivity_lead_churn_and_endpoint_candidates_known": True,
            "prior_event_sets_reused": False, "prior_outcomes_used_to_set_formula_ranks_side_hold_or_clock": False,
            "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False,
            "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False,
            "selection_basis": "independent contemporaneous sign-coupling of leveraged inventory and price discovery",
        },
        "stopping_rule": "terminal first-failure sequence: source support, Gross9 novelty, train/test/eval/final strict economics, then RV20 q90 audit; no aggregation, pair treatment, rank, side, hold, clock, subset, threshold, source, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core) or payload != build():
        raise RuntimeError("HVOIPSCR preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    payload = build(); validate(payload); args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
