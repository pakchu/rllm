"""Outcome-blind preregistration for HVDFICR-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVDFICR-12"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_daily_flow_impact_capacity_reversal_preregistration_2026-08-13.json"
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
        "protocol_version": "high_volatility_daily_flow_impact_capacity_reversal_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-13",
        "singleton": True,
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "mechanism": {
            "claim": (
                "When a completed BTC day exhibits unusually high contemporaneous price impact "
                "per unit of normalized aggressive flow, market-maker inventory capacity is "
                "scarce. In an already high-variation regime, the current impact channel can "
                "dominate delayed information and the completed aggregate flow should reverse "
                "over the next twelve hours."
            ),
            "side": "negative strict sign of completed-day aggregate aggressive quote flow",
            "external_basis": {
                "paper": "Heterogeneous Return Predictability from Order Flow",
                "author": "Meichen Qian",
                "ssrn": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6875187",
                "version": "2026-06-03",
                "reported_result": (
                    "order-flow return predictability can become negative when market makers are "
                    "very unwilling to provide liquidity, with high volatility strengthening the reversal"
                ),
                "adaptation": (
                    "positive through-origin hourly flow-impact beta is used only as a transparent "
                    "daily inventory-capacity proxy; no coefficient or threshold is fitted to BTC outcomes"
                ),
            },
            "why_distinct": (
                "HVFIC compares high- versus low-flow bars inside offset eight-hour blocks and follows "
                "same-sign flow. HVLIR follows hourly price displacement after turnover-normalized impact. "
                "HVPST estimates next-bar signed-turnover reversal gamma and fades price. HVDFICR instead "
                "uses one full UTC day of 24 hourly normalized-flow/return pairs, ranks the single positive "
                "contemporaneous impact beta as a market-maker capacity state, and fades aggregate flow. "
                "It reuses no prior event set or diagnostic control."
            ),
            "why_suited_to_volatile_regimes": (
                "completed daily realized variation must occupy its causal upper 35 percent"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "at most one sparse 00:05 UTC capacity-tail onset per day is absent from Gross9 primitives"
            ),
        },
        "features": {
            "decision_grid": "every exact 00:00 UTC day boundary D",
            "source_day": (
                "1,440 exact coherent BTCUSDT perpetual one-minute rows [D-24h,D), partitioned "
                "into 24 exact nonoverlapping hourly buckets"
            ),
            "hour_return": "log(hour last close/hour first open), finite strict nonzero",
            "hour_normalized_flow": (
                "sum(2*taker_buy_quote-quote_asset_volume)/sum(quote_asset_volume), with positive "
                "denominator and finite strict nonzero value"
            ),
            "impact_beta": (
                "sum(hour_normalized_flow*hour_return)/sum(hour_normalized_flow^2) through the origin "
                "over all 24 hours; finite strict positive"
            ),
            "aggregate_flow": (
                "sum daily signed aggressive quote divided by sum daily quote turnover; finite strict nonzero"
            ),
            "realized_variation": (
                "sqrt(sum squared one-minute open-to-close log returns) over the completed day; "
                "finite strict positive"
            ),
            "causal_ranks": (
                "strict-prior midranks of impact beta and realized variation over at most 180 earlier "
                "source-valid days, minimum 120, current excluded"
            ),
            "capacity_tail": "impact-beta rank>=0.75",
            "variation_gate": "realized-variation rank>=0.65",
            "onset": (
                "full eligible state true now and false on the immediately preceding exact source-valid "
                "day; missing prior cannot trigger"
            ),
            "no_imputation": True,
        },
        "clock": {
            "feature_available": "D after the completed UTC source day",
            "entry": "exact BTCUSDT perpetual D+5m open",
            "side": "negative aggregate-flow sign",
            "hold": "12 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
            "rv20": "q90 only after unchanged all-stage pass",
        },
        "policy": {
            "hourly_buckets": 24,
            "history_days": 180,
            "minimum_history_days": 120,
            "impact_rank_min": 0.75,
            "variation_rank_min": 0.65,
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
                "fixed quantity, exact funding, 6bp and 10bp per notional side, held 5m favorable "
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
                "no_impact_tail",
                "no_variation_gate",
                "negative_beta_state",
                "one_day_stale_capacity",
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
                "columns": [
                    "ts", "open", "high", "low", "close", "quote_asset_volume", "taker_buy_quote",
                ],
                "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "external_order_flow_capacity_basis_read": True,
            "repository_daily_flow_impact_capacity_reversal_found": False,
            "prior_impact_and_daily_flow_candidate_outcomes_known": True,
            "prior_event_sets_reused": False,
            "prior_diagnostic_control_promoted": False,
            "prior_outcomes_used_to_set_formula_rank_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "selection_basis": "paper-predicted order-flow reversal under scarce market-maker capacity",
        },
        "stopping_rule": (
            "terminal first failure; no source, hourly partition, beta, rank, variation, onset, side, "
            "delay, hold, clock, subset, threshold, comparator, or control repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVDFICR preregistration drift")
    if value["outcomes_opened"] or value["source_incidence_opened"] or value["gross9_rows_opened"]:
        raise RuntimeError("HVDFICR evidence boundary drift")


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
