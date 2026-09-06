"""Outcome-blind preregistration for HVWRTR-8."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template


POLICY_ID = "HVWRTR-8"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_wick_response_transfer_relay_preregistration_2026-08-10.json"
)


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = copy.deepcopy(template.build())
    core.pop("manifest_hash")
    core.update(
        protocol_version="high_volatility_wick_response_transfer_relay_v1",
        policy_id=POLICY_ID,
        mechanism={
            "claim": "Signed rejection wicks reveal whether failed intrabar excursions are being absorbed or overrun. Estimate that response law on the completed first seven hours, then apply it to independent final-hour wick pressure during elevated BTC variation.",
            "side": "strict sign of first-seven-hour wick-to-next-body response times final-hour signed wick pressure",
            "why_distinct": "HVWIR directly fades one aggregate dominant wick; HVSRP relates absolute returns to next returns; HVTLRR transfers turnover response and confirms block direction; HVIABR counts contemporaneous close locations. HVWRTR instead estimates an ordered signed-wick response law and applies it to a disjoint final-hour pressure sample without block-return confirmation.",
            "why_suited_to_volatile_regimes": "the causal variation tail selects volatile paths while a response-tail gate selects stable rejection transmission",
            "why_low_gross9_overlap_is_plausible": "hourly onsets from lagged wick-response transfer are absent from Gross9 primitives",
        },
        features={
            "decision_grid": "every exact UTC hour D",
            "block": "96 exact coherent five-minute aggregates from 480 unique bars_binance BTCUSDT one-minute rows [D-8h,D)",
            "upper_wick": "log(high/max(open,close)) for each five-minute bar",
            "lower_wick": "log(min(open,close)/low) for each five-minute bar",
            "signed_wick": "lower_wick-upper_wick",
            "bar_return": "log(close/open)",
            "response": "sample Pearson correlation of signed_wick[t] with bar_return[t+1] for t=0..82; both sample variances positive and response strict nonzero",
            "separation": "bar 83 to 84 transition unused between response estimation and pressure application",
            "final_hour_pressure": "sum signed_wick[t] for t=84..95; finite strict nonzero",
            "realized_variation": "sqrt(sum squared one-minute log(close/open) returns), finite strict positive",
            "response_rank": "strict-prior midrank of abs(response) over at most 2160 earlier source-valid hourly observations, minimum 1440, current excluded; rank>=0.80",
            "variation_rank": "same strict-prior 2160/1440 rule; rank>=0.65",
            "eligible_state": "response and variation gates pass with response and final-hour pressure strict nonzero",
            "onset": "eligible now and immediately preceding exact source-valid hour ineligible; missing prior cannot trigger",
            "no_imputation": True,
        },
        clock={
            "decision": "every exact UTC hour after the completed eight-hour block",
            "entry": "exact BTCUSDT D+5m open",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        policy={
            "history_hours": 2160,
            "minimum_history_hours": 1440,
            "response_rank_min": 0.80,
            "variation_rank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 8,
            "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        diagnostic_controls={
            "names": ["no_response_tail", "no_variation_gate", "late_wick_pressure_only", "one_hour_stale_transfer", "direction_flip", "forced_long"],
            "cannot_be_promoted": True,
        },
        source_plan={
            "bars": {
                "table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close"],
                "window": ["2023-04-01T00:00:00Z", "2026-08-01T00:00:00Z"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        research_boundary={
            "prior_wick_return_response_and_recent_candidate_outcomes_known": True,
            "repository_ordered_wick_response_transfer_candidate_found": False,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_rank_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "ordered rejection-wick response transfer with disjoint application window",
        },
        stopping_rule="Terminal first failure; no wick formula, response window, rank, side, hold, clock, subset, or control repair.",
    )
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core) or payload != build():
        raise RuntimeError("HVWRTR preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(); validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)
