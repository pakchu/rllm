"""Outcome-blind preregistration for HVRAVI-24."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVRAVI-24"
DEFAULT_OUTPUT = Path("results/high_volatility_range_action_verification_ignition_relay_preregistration_2026-08-11.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    contract = copy.deepcopy(template.build()); contract.pop("manifest_hash")
    contract.update(
        protocol_version="high_volatility_range_action_verification_ignition_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-11",
        mechanism={
            "claim": "During elevated realized variation, the canonical Chande Range Action Verification Index first crossing above its 3% trend threshold on completed four-hour BTC auctions identifies a newly verified multi-day trend; follow the sign of the seven-versus-sixty-five-period simple-average spread for twenty-four hours.",
            "side": "long when SMA(7)>SMA(65) and absolute RAVI crosses strictly above 3%; short when SMA(7)<SMA(65) at the same crossing",
            "why_distinct": "RAVI measures the absolute percentage separation of fixed 7/65 simple close averages and trades only a fresh 3% trend ignition. MACD compares exponential speeds and a signal line; VHF uses range versus travel; ADX uses true-range directional movement; ordinary price/average crosses do not require verified separation. HVRAVI uses no volume, flow, OI, funding, fitted outcome, reused event, repair, or promoted control.",
            "why_suited_to_volatile_regimes": "completed trailing twenty-four-hour realized variation must independently rank in its causal upper 35%, focusing verified trend ignition on July-like auctions",
            "why_low_gross9_overlap_is_plausible": "sparse four-hour 7/65 simple-average separation ignitions are absent from Gross9 primitives",
        },
        external_basis={
            "origin": "Tushar Chande, Range Action Verification Index, Beyond Technical Analysis",
            "definition_source": "https://www.neuroshell.com/manuals/ais2/ravi.htm",
            "fixed_definition": "absolute value of 100*(SMA(close,7)-SMA(close,65))/SMA(close,65)",
            "fixed_interpretation": "the original 3 percent threshold separates trending from sideways markets",
            "selection_use": "published 7/65 simple-average periods, absolute percentage formula, 3% threshold, and trend interpretation only; no incidence or outcomes",
        },
        features={
            "decision_grid": "every exact four-hour UTC boundary",
            "source_bar": "exact aggregation of 240 coherent BTCUSDT one-minute rows [T-4h,T)",
            "fast_average": "simple mean of the latest seven valid completed four-hour closes",
            "slow_average": "simple mean of the latest sixty-five valid completed four-hour closes",
            "signed_spread_pct": "100*(fast_average-slow_average)/slow_average, finite with positive denominator",
            "ravi": "absolute signed_spread_pct",
            "ignition": "current RAVI strictly above 3 after immediately prior valid RAVI at or below 3",
            "direction": "strict sign of signed_spread_pct",
            "variation": "sqrt(sum squared completed five-minute open-to-close log returns over [T-24h,T)), finite strict positive",
            "variation_rank": "strict-prior midrank over at most 180 earlier valid decisions, minimum 120, current excluded; rank>=0.65",
            "no_imputation": True,
        },
        clock={
            "feature_available": "four-hour boundary after completed source bar",
            "entry": "exact BTCUSDT open five elapsed minutes later",
            "hold": "24 elapsed hours",
            "reservation": "global half-open first-eligible reservation; exit first on equal open",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        policy={
            "fast_periods": 7,
            "slow_periods": 65,
            "trend_threshold_pct": 3.0,
            "variation_hours": 24,
            "variation_history_decisions": 180,
            "minimum_variation_history_decisions": 120,
            "variation_rank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 24,
            "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        diagnostic_controls={
            "names": ["no_variation_gate", "trend_release", "one_bar_stale_ignition", "direction_flip", "forced_long"],
            "cannot_be_promoted": True,
        },
        source_plan={
            "bars": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "high", "low", "close"], "window": ["2023-04-01T00:00:00Z", "2026-08-01T00:00:00Z"], "read_after_preregistration": True},
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        research_boundary={
            "canonical_ravi_definition_read": True,
            "repository_ravi_candidate_found": False,
            "adjacent_macd_vhf_adx_outcomes_known": True,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "canonical RAVI(7,65) 3% trend ignition under the requested high-variation regime",
        },
        stopping_rule="Terminal first failure; no average type, period, formula, 3% threshold, ignition, variation, side, hold, clock, subset, or control repair.",
    )
    return {**contract, "manifest_hash": canonical_hash(contract)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build(): raise RuntimeError("HVRAVI preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    payload = build(); validate(payload); args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"); print(args.output)
