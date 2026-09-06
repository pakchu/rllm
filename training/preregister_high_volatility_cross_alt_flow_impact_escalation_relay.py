"""Outcome-blind preregistration for HVCAFIE-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVCAFIE-8"
SLUG = "high_volatility_cross_alt_flow_impact_escalation_relay"
ALTS = ("ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT")
DEFAULT_OUTPUT = Path(f"results/{SLUG}_preregistration_2026-08-13.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": f"{SLUG}_v1",
        "policy_id": POLICY_ID,
        "slug": SLUG,
        "as_of_date": "2026-08-13",
        "singleton": True,
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "mechanism": {
            "claim": "At three fixed daily decisions, at least three liquid alt perpetuals showing same-direction aggressive quote flow and price response in both halves of a completed eight-hour block, with strictly rising return-per-flow impact in the second half, identify broad liquidity depletion and accelerating price discovery. During elevated BTC variation, BTC follows the common alt flow-impact direction for eight hours.",
            "side": "strict common direction of the selected alts' two-half aggressive flow and price response",
            "why_distinct": "BTC temporal-impact escalation divides absolute return by quote turnover without aggressive-flow direction. Cross-alt flow-price transmission maps first-half flow into second-half return without comparing price impact across halves. HVCAFIE requires within-alt same-direction aggressive flow and price response in both halves, computes the second-to-first return-per-absolute-flow-imbalance ratio, selects each alt's own causal escalation tail, and transfers only broad cross-alt agreement. It uses no BTC return direction, funding, OI, premium, fitted outcome, reused event set, promoted control, or post-entry data.",
            "why_suited_to_volatile_regimes": "BTC completed twenty-four-hour realized variation must occupy its causal upper 35 percent while the selected alts exhibit impact acceleration above one.",
            "why_low_gross9_overlap_is_plausible": "03:05, 11:05 and 19:05 UTC entries conditioned on cross-alt two-half aggressive-flow impact escalation are absent from Gross9 primitive clocks.",
        },
        "features": {
            "bars": "96 exact aligned coherent five-minute aggregates from bars_binance one-minute rows per alt over [D-8h,D), with finite positive OHLC and quote_asset_volume and finite taker_buy_quote in [0,quote_asset_volume]; no imputation",
            "decision_times": "every calendar day at exact 03:00, 11:00 and 19:00 UTC",
            "halves": "first 48 and final 48 completed five-minute bars",
            "half_flow_imbalance": "(2*sum(taker_buy_quote)-sum(quote_asset_volume))/sum(quote_asset_volume), finite strict nonzero with positive denominator",
            "half_return": "log(last half close/first half open), finite strict nonzero",
            "directional_response": "both half flow imbalances have one strict common sign and each half return has that same sign",
            "half_impact": "abs(half_return)/abs(half_flow_imbalance), finite strict positive",
            "impact_escalation": "second_half_impact/first_half_impact, finite and strictly greater than one",
            "escalation_rank": "for each alt independently, strict-prior midrank of impact escalation over at most 120 prior jointly source-valid scheduled decisions, minimum 60, current excluded; selected when rank>=0.65",
            "score": "sum(log(impact_escalation)*direction over selected alts)/sum(log(impact_escalation) over selected alts)",
            "consensus_gate": "at least three selected alts, abs(score)>=0.60, and at least three selected directions equal score sign",
            "btc_realized_variation": "sqrt(sum squared exact BTC five-minute log(close/open) returns over [D-24h,D))",
            "variation_rank": "strict-prior midrank over 20 prior jointly source-valid scheduled decisions, minimum 15, current excluded; rank>=0.65",
            "no_imputation": True,
        },
        "clock": {
            "decision": "exact daily 03:00, 11:00 or 19:00 UTC",
            "entry": "exact BTCUSDT perpetual decision+5m open",
            "side": "score sign",
            "hold": "8 elapsed hours",
            "reservation": "global chronological half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not signal input; exact settlements only after novelty",
        },
        "policy": {
            "block_bars": 96, "half_bars": 48, "escalation_prior_decisions": 120, "escalation_minimum_decisions": 60, "escalation_rank_min": 0.65,
            "minimum_selected_alts": 3, "score_absolute_min": 0.60, "minimum_agreeing_alts": 3,
            "variation_bars": 288, "variation_prior_decisions": 20, "variation_minimum_decisions": 15, "variation_rank_min": 0.65,
            "decision_hours_utc": [3, 11, 19], "decision_minute": 0, "entry_delay_minutes": 5, "hold_hours": 8,
            "leverage": 0.5, "base_cost_per_notional_side": 0.0006, "stress_cost_per_notional_side": 0.001,
        },
        "stages": {
            "train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
        },
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.2, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.1, "candidate_near_6h_share_max": 0.35, "occupied_5m_bar_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.1, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "accounting": "fixed quantity, exact funding, 6bp/10bp per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "post_stage_volatility_audit": {"prerequisite": "unchanged all-stage pass", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "diagnostic_controls": {"names": ["no_variation_gate", "no_escalation_tail", "two_alt_consensus", "no_rising_impact", "one_decision_stale_consensus", "direction_flip", "forced_long"], "cannot_be_promoted": True},
        "source_plan": {"market": {"table": "bars_binance", "symbols": ["BTCUSDT", *ALTS], "interval": "1m", "columns": ["ts", "symbol", "open", "high", "low", "close", "quote_asset_volume", "taker_buy_quote"], "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"]}, "read_after_preregistration_commit": True, "execution_prices": "sealed until source and novelty pass"},
        "research_boundary": {"related_btc_impact_and_cross_alt_flow_transmission_outcomes_known": True, "repository_exact_cross_alt_two_half_flow_impact_escalation_found": False, "prior_event_sets_or_controls_reused": False, "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False, "selection_basis": "independent cross-alt aggressive-flow impact-escalation mechanism selected by outcome-blind primitive-gap audit"},
        "stopping_rule": "terminal first failure; no universe, block, half, flow, impact formula, history, rank, consensus, variation gate, direction, clock, hold, subset, comparator, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVCAFIE prereg drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build()
    validate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if args.output.exists() and args.output.read_bytes() != encoded:
        raise RuntimeError(f"refusing overwrite {args.output}")
    args.output.write_bytes(encoded)
    print(args.output)
