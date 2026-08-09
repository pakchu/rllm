"""Outcome-blind preregistration for HVDOER-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

POLICY_ID = "HVDOER-12"
DEFAULT_OUTPUT = Path("results/high_volatility_daily_online_expert_relay_preregistration_2026-08-09.json")
MARKET = Path("data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz")
MARKET_SHA = "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_daily_online_expert_relay_v1",
        "policy_id": POLICY_ID, "as_of_date": "2026-08-09", "singleton": True,
        "current_candidate_outcomes_opened": False, "candidate_incidence_opened": False,
        "gross9_rows_opened": False,
        "mechanism": {
            "claim": (
                "High-volatility BTC alternates between trend propagation and shock reversal. A daily "
                "causal selector that scores frozen 6-hour and 24-hour momentum and reversal experts "
                "using only their latest already-completed twelve-hour counterfactual labels can adapt "
                "direction without future-stage fitting. The currently best expert should govern the "
                "next twelve elapsed hours."
            ),
            "side": "side emitted by the highest mean-log-payoff frozen expert over the latest 60 mature daily labels",
            "why_distinct": (
                "HVCRMR averages responses of eligible twelve-hour shocks and failed novelty. The 2026-07 "
                "online search reused six outcome-screened legacy sleeves and a grid. HVDOER has four "
                "formulaic raw-price experts, one fixed daily clock, one fixed memory, no grid, no sleeve "
                "events, and no selection-window ranking."
            ),
            "volatile_market_target": "current completed 24-hour realized-variation causal rank at least 0.65",
            "why_low_gross9_overlap_is_plausible": "one 03:05 UTC adaptive daily clock is absent from Gross9",
        },
        "features": {
            "decision": "exact daily 03:00 UTC after the 02:55 five-minute bar completes",
            "returns": {
                "r6": "log(last completed 5m close / completed close exactly 6h earlier)",
                "r24": "log(last completed 5m close / completed close exactly 24h earlier)",
            },
            "experts_in_tie_order": ["momentum_6h", "reversal_6h", "momentum_24h", "reversal_24h"],
            "expert_sides": {
                "momentum_6h": "sign(r6)", "reversal_6h": "-sign(r6)",
                "momentum_24h": "sign(r24)", "reversal_24h": "-sign(r24)",
            },
            "counterfactual_label": (
                "expert side at prior decision times log(open at prior entry+12h / open at prior entry); "
                "labels enter memory only after the exit open is observed"
            ),
            "online_score": (
                "arithmetic mean of each expert's latest at most 60 mature labels, minimum 30; current "
                "decision and all unmatured labels excluded; largest score wins, frozen expert order breaks ties"
            ),
            "variation": "sqrt(sum squared 288 exact completed 5m close-to-close returns over prior 24h)",
            "variation_rank": "strict-prior midrank over at most 252 valid daily decisions, minimum 126, current excluded; rank>=0.65",
            "no_imputation": True,
        },
        "causal_label_authorization": {
            "allowed_before_incidence": (
                "only counterfactual expert labels whose fixed twelve-hour exit open is strictly before "
                "or equal to the current decision"
            ),
            "current_candidate_postentry_return_forbidden": True,
            "funding_pnl_cagr_mdd_forbidden": True,
            "future_stage_refit_or_reset": False,
        },
        "clock": {
            "entry": "exact daily 03:05 UTC open", "hold": "12 elapsed hours",
            "reservation": "daily clocks are nonoverlapping; global half-open, exit first on equal open",
            "split_crossing_action": "skip", "gross_exposure": 0.5,
            "funding_oi_premium_rv20": "not signal inputs; exact funding after novelty; RV20 q90 only after all economics pass",
        },
        "policy": {
            "expert_memory_days": 60, "minimum_mature_labels": 30,
            "variation_prior_days": 252, "variation_minimum_days": 126,
            "variation_rank_min": 0.65, "entry_delay_minutes": 5, "hold_hours": 12,
            "leverage": 0.5, "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        "stages": {
            "train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
        },
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.2, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.1, "candidate_near_6h_share_max": 0.35, "occupied_5m_bar_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {
            "absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0,
            "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0,
            "weekly_signflip_one_sided_p_max": 0.1, "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True,
            "stop_on_first_failure": True,
            "accounting": "fixed quantity, exact funding, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR",
        },
        "post_stage_volatility_audit": {"prerequisite": "unchanged candidate passes train, test, eval, final", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "diagnostic_controls": {"names": ["no_variation_gate", "fixed_momentum_6h", "fixed_momentum_24h", "one_day_stale_winner", "direction_flip", "same_clock_forced_long"], "cannot_be_promoted": True},
        "source_plan": {
            "historical": {"path": str(MARKET), "sha256": MARKET_SHA},
            "live_extension": "read-only Postgres BTCUSDT bars_binance 1m through 2026-08-01 with exact overlap parity",
            "execution_prices": "sealed except causally mature expert-label opens until source and novelty pass",
        },
        "research_boundary": {
            "prior_online_and_fixed_direction_outcomes_known": True,
            "exact_hvdoer_incidence_or_current_trade_outcomes_known": False,
            "candidate_incidence_opened": False, "current_candidate_postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False, "candidate_count": 1, "grid": False,
            "repair_of_prior_candidate": False, "promoted_prior_control": False,
            "selection_basis": "independent causal online regime adaptation over formulaic raw-price experts",
        },
        "stopping_rule": "terminal first failure; no clock, expert, memory, label, rank, volatility, side, hold, subset, comparator, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVDOER preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(); result = build(); validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
