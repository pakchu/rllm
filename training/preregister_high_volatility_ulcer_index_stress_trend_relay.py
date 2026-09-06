"""Outcome-blind preregistration for HVUI-12."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template


POLICY_ID = "HVUI-12"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_ulcer_index_stress_trend_relay_preregistration_2026-08-11.json"
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
    contract = copy.deepcopy(template.build())
    contract.pop("manifest_hash")
    contract.update(
        protocol_version="high_volatility_ulcer_index_stress_trend_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-11",
        mechanism={
            "claim": "When completed trailing realized variation and the canonical Ulcer Index are both unusually high, the market is moving inside a deep, persistent drawdown-stress state rather than an ordinary volatile fluctuation; follow the latest completed four-hour direction for twelve hours.",
            "side": "strict sign of the completed four-hour close/open log return",
            "why_distinct": "Ulcer Index is the root mean square of percentage drawdowns from a rolling closing high. It measures drawdown depth and persistence jointly, not duration alone, directional drawup/drawdown asymmetry, endpoint return, OHLC range, variance acceleration, path passage, volume, flow, OI, funding, a fitted outcome, reused event set, or promoted control.",
            "why_suited_to_volatile_regimes": "completed trailing twenty-four-hour realized variation must rank in its causal upper 35%, and Ulcer Index must independently rank in its upper 30%",
            "why_low_gross9_overlap_is_plausible": "four-hour rolling RMS drawdown-stress states are absent from Gross9 primitives",
        },
        external_basis={
            "origin": "Peter Martin canonical Ulcer Index convention",
            "fixed_definition": "fourteen-period rolling closing high; percentage drawdown from that high; square; fourteen-period mean; square root",
            "selection_use": "published fourteen-period RMS drawdown definition, terminal bar direction, and elevated-stress interpretation only; no incidence or outcomes",
        },
        features={
            "decision_grid": "every exact four-hour UTC boundary",
            "source_bar": "exact aggregation of 240 coherent BTCUSDT one-minute rows [T-4h,T)",
            "rolling_high": "highest completed four-hour close over the current and prior thirteen valid consecutive bars",
            "percentage_drawdown": "100*(completed close/rolling_high-1), finite and nonpositive",
            "ulcer_index": "sqrt(mean of the current and prior thirteen squared percentage_drawdown values), requiring the full nested warmup without gaps",
            "ulcer_rank": "strict-prior midrank over at most 180 earlier valid decisions, minimum 120, current excluded; rank>=0.70",
            "terminal_return": "log(completed four-hour close/open), finite strict nonzero",
            "variation": "sqrt(sum squared completed one-minute open-to-close log returns over [T-24h,T)), finite strict positive",
            "variation_rank": "strict-prior midrank over at most 180 earlier valid decisions, minimum 120, current excluded; rank>=0.65",
            "eligible": "source valid, both frozen rank gates pass, and terminal return is strict nonzero",
            "no_imputation": True,
        },
        clock={
            "feature_available": "four-hour boundary after all feature inputs complete",
            "entry": "exact BTCUSDT perpetual open five elapsed minutes later",
            "side": "terminal_return sign",
            "hold": "12 elapsed hours",
            "reservation": "global half-open first-eligible reservation; exit first on equal open",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        policy={
            "source_bar_minutes": 240,
            "ulcer_periods": 14,
            "history_decisions": 180,
            "minimum_history_decisions": 120,
            "ulcer_rank_min": 0.70,
            "variation_hours": 24,
            "variation_rank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 12,
            "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        diagnostic_controls={
            "names": [
                "no_ulcer_gate",
                "no_variation_gate",
                "raw_drawdown_stress",
                "one_bar_stale_state",
                "direction_flip",
                "forced_long",
            ],
            "cannot_be_promoted": True,
        },
        source_plan={
            "bars": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close"],
                "window": ["2020-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        research_boundary={
            "canonical_ulcer_index_definition_read": True,
            "repository_ulcer_index_candidate_found": False,
            "prior_drawdown_duration_and_asymmetry_outcomes_known": True,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_rank_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "canonical Ulcer Index(14) RMS drawdown stress under the requested high-variation regime",
        },
        stopping_rule="Terminal first failure; no period, drawdown definition, rank, variation gate, side, hold, clock, subset, control, or other repair.",
    )
    return {**contract, "manifest_hash": canonical_hash(contract)}


def validate(contract: dict[str, Any]) -> None:
    core = {key: value for key, value in contract.items() if key != "manifest_hash"}
    if contract.get("manifest_hash") != canonical_hash(core) or contract != build():
        raise RuntimeError("HVUI-12 preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = build()
    validate(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(args.output)
