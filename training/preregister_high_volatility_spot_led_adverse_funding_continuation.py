"""Outcome-blind preregistration for HVSLAF-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from training import preregister_high_volatility_cross_structure_action_vote as hvcav


POLICY_ID = "HVSLAF-8"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_spot_led_adverse_funding_continuation_"
    "preregistration_2026-08-16.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    contract = hvcav.build()
    core = {
        "protocol_version": "high_volatility_spot_led_adverse_funding_continuation_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-16",
        "singleton": True,
        "exploratory_discovery": True,
        "fresh_confirmatory_evidence": False,
        "source_incidence_opened": False,
        "outcomes_opened": False,
        "gross9_rows_opened": False,
        "candidate_family": [POLICY_ID],
        "candidate_family_size": 1,
        "mechanism": {
            "claim": (
                "At an actual funding settlement after a high-variation eight-hour cycle, "
                "same-direction spot and perpetual returns with larger absolute spot displacement "
                "and same-direction spot aggressive flow identify cash-led price discovery. When "
                "the settled funding sign is adverse to that direction, derivatives remain "
                "under-positioned; continue in the cash-led direction for eight hours."
            ),
            "side": "common strict sign of completed spot return, perpetual return, and spot aggressive quote flow",
            "why_distinct": (
                "HVSLAF uses the conjunction of spot displacement leadership, spot aggressive-flow "
                "confirmation, and opposite-sign actual settled funding. It uses no OI, alt, FX, "
                "post-settlement confirmation, fitted outcome, or promoted control."
            ),
            "why_suited_to_volatile_regimes": "completed BTC perpetual variation must be in its causal upper 40 percent",
            "why_low_gross9_overlap_is_plausible": (
                "the actual-settlement clock plus spot/perpetual magnitude ordering, spot flow, and "
                "adverse funding conjunction is absent from Gross9 primitives"
            ),
        },
        "features": {
            "decision": "actual BTCUSDT funding settlement S",
            "cycle": "480 exact coherent one-minute Binance spot and perpetual BTCUSDT rows [S-8h,S)",
            "perpetual_return": "sum of 480 minute ln(perpetual close/open) returns, finite strict nonzero",
            "spot_return": "sum of 480 minute ln(spot close/open) returns, finite strict nonzero",
            "spot_flow_share": (
                "sum(2*spot taker_buy_quote-spot quote_asset_volume)/sum(spot quote_asset_volume), "
                "finite strict nonzero"
            ),
            "cash_leadership": (
                "spot and perpetual returns have one strict sign, spot flow has that sign, and "
                "abs(spot_return)>abs(perpetual_return)"
            ),
            "funding_adversity": "finite nonzero funding_rate[S] has the strict opposite sign to spot_return",
            "realized_variation": "sqrt(sum squared one-minute perpetual returns), strict positive",
            "variation_rank": (
                "strict-prior midrank over at most 270 source-valid settlement cycles; current excluded; "
                "minimum 180; rank>=0.60"
            ),
            "eligible": "exact source-valid cycle, cash leadership, adverse settled funding, and variation rank passes",
            "side": "strict sign of spot_return",
            "no_imputation": True,
        },
        "clock": {
            "decisions": "actual BTCUSDT funding settlements, normally 00/08/16 UTC",
            "entry": "exact BTCUSDT perpetual S+5m open",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal entry",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "trigger settlement precedes entry; exact held settlements only after source and Gross9 pass",
        },
        "policy": {
            "history_cycles": 270,
            "minimum_history_cycles": 180,
            "path_minutes": 480,
            "variation_rank_min": 0.60,
            "entry_delay_minutes": 5,
            "hold_hours": 8,
            "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.0010,
        },
        "stages": contract["stages"],
        "source_support_gates": contract["source_support_gates"],
        "gross9_novelty_gates": contract["gross9_novelty_gates"],
        "economic_gates": contract["economic_gates"],
        "source_plan": {
            "funding": "Postgres funding_rates_binance BTCUSDT actual timestamps/rates",
            "perpetual": "Postgres bars_binance BTCUSDT exact coherent 1m OHLC",
            "spot": (
                "Postgres bars_binance_spot BTCUSDT exact coherent 1m OHLC, quote_asset_volume, "
                "and taker_buy_quote"
            ),
            "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "read_after_preregistration": True,
            "execution_prices": "sealed until source support and Gross9 pass",
        },
        "research_boundary": {
            "prior_spot_perpetual_OI_state_outcomes_known": True,
            "prior_funding_and_delayed_cash_source_failure_known": True,
            "prior_funding_level_family_source_failures_known": True,
            "exact_spot_led_adverse_funding_flow_conjunction_incidence_or_outcomes_known": False,
            "user_authorized_multi_condition_composition": True,
            "prior_outcomes_informed_candidate_family": True,
            "prior_outcomes_used_to_set_rank_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "composition_is_exploratory_not_confirmatory": True,
            "promoted_prior_control": False,
            "classification": "exploratory discovery; not fresh confirmatory evidence",
        },
        "stopping_rule": (
            "terminal first failure; no settlement, source, leadership, flow, funding, variation, "
            "side, clock, hold, subset, threshold, comparator, or control repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: Mapping[str, Any]) -> None:
    if value != build():
        raise RuntimeError("HVSLAF-8 preregistration drift")
    contract = hvcav.build()
    for key in ("stages", "source_support_gates", "gross9_novelty_gates", "economic_gates"):
        if value[key] != contract[key]:
            raise RuntimeError(f"HVSLAF-8 {key} drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = build()
    validate(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
