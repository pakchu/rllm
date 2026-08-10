"""Outcome-blind preregistration for HVALPR-24."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVALPR-24"
DEFAULT_OUTPUT = Path("results/high_volatility_amihud_liquidity_premium_relay_preregistration_2026-08-11.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    candidate = copy.deepcopy(template.build())
    candidate.pop("manifest_hash")
    candidate.update(
        protocol_version="high_volatility_amihud_liquidity_premium_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-11",
        mechanism={
            "claim": "A positive daily innovation in Amihud illiquidity raises the compensation required for holding BTC, while an unusually negative innovation lowers it. In elevated realized variation, trade with the liquidity premium for the next day: long after an upper-tail illiquidity innovation and short after a lower-tail innovation.",
            "side": "long for innovation rank>=0.80; short for innovation rank<=0.20",
            "why_distinct": "HVPIAR compares upside versus downside contemporaneous price impact and trades toward the fragile side; temporal/Kyle candidates compare impact across halves or venues. HVALPR uses a directionless full-day Amihud state innovation relative to its strictly prior median and maps that state to a next-day liquidity premium. It uses no completed return direction, flow, funding, OI, price anchor, fitted outcome, prior event, or promoted control.",
            "why_suited_to_volatile_regimes": "the completed daily realized-variation rank must be in its causal upper 35%",
            "why_low_gross9_overlap_is_plausible": "sparse daily two-sided illiquidity-tail clocks are absent from Gross9 primitives",
        },
        external_basis={
            "paper": "Beyond the Hype: A Multi-Layer Machine Learning Framework for Cryptocurrency Return Forecasting",
            "author": "Waseem Khoso",
            "written": "2026-02-28",
            "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6320138",
            "reported_relation": "formal hypothesis tests report that lower liquidity predicts higher future cryptocurrency returns",
            "untested_adaptation": "the fixed prior-median Amihud innovation, two-sided ranks, high-variation gate, Binance perpetual execution, and one-day hold are tested as a transparent BTC adaptation rather than a replication of the paper's random forest",
        },
        features={
            "decision_grid": "every calendar day at exact 00:00 UTC",
            "window": "288 exact coherent five-minute groups from 1,440 BTCUSDT one-minute rows [D-24h,D)",
            "five_minute_return": "log(group close/group open), finite",
            "five_minute_quote_turnover": "sum quote_asset_volume, finite strict positive in every group",
            "daily_amihud": "mean(abs(five-minute return)/five-minute quote turnover) across 288 groups, finite strict positive",
            "reference": "median log daily Amihud over thirty exact source-valid days ending strictly before current day",
            "innovation": "log(current daily Amihud)-strictly-prior thirty-day median log Amihud, finite strict nonzero",
            "realized_variation": "sqrt(sum squared five-minute returns), finite strict positive",
            "prior_ranks": "strict-prior midranks of innovation and variation over at most 180 earlier source-valid days, minimum 120, current excluded",
            "liquidity_tail": "innovation rank>=0.80 or <=0.20",
            "variation_gate": "realized-variation rank>=0.65",
            "source_validity": "exact distinct minute grid, finite positive coherent OHLC, finite positive quote turnover, no imputation",
            "no_imputation": True,
        },
        clock={
            "feature_available": "00:00 UTC after the completed daily window",
            "entry": "exact BTCUSDT 00:05 UTC open",
            "hold": "24 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        policy={
            "reference_days": 30,
            "history_days": 180,
            "minimum_history_days": 120,
            "innovation_rank_long_min": 0.80,
            "innovation_rank_short_max": 0.20,
            "variation_rank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 24,
            "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        diagnostic_controls={
            "names": ["no_liquidity_tail", "no_variation_gate", "raw_illiquidity_rank", "one_day_stale_innovation", "direction_flip", "forced_long"],
            "cannot_be_promoted": True,
        },
        source_plan={
            "bars": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close", "quote_asset_volume"],
                "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        research_boundary={
            "external_liquidity_premium_basis_read": True,
            "repository_amihud_liquidity_premium_candidate_found": False,
            "prior_price_impact_outcomes_known": True,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "paper-reported liquidity premium expressed as a fixed transparent Amihud innovation state",
        },
        stopping_rule="Terminal first failure; no Amihud aggregation, median window, rank history, tail, variation, side, hold, clock, subset, threshold, or control repair.",
    )
    return {**candidate, "manifest_hash": canonical_hash(candidate)}


def validate(candidate: dict[str, Any]) -> None:
    core = {key: value for key, value in candidate.items() if key != "manifest_hash"}
    if candidate.get("manifest_hash") != canonical_hash(core) or candidate != build():
        raise RuntimeError("HVALPR preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build()
    validate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(args.output)
