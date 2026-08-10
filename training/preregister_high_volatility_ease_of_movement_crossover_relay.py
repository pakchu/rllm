"""Outcome-blind preregistration for HVEOM-24."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVEOM-24"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_ease_of_movement_crossover_relay_preregistration_2026-08-11.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    contract = copy.deepcopy(template.build())
    contract.pop("manifest_hash")
    contract.update(
        protocol_version="high_volatility_ease_of_movement_crossover_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-11",
        mechanism={
            "claim": "During elevated realized variation, a canonical fourteen-period Ease of Movement zero crossover on completed four-hour BTC auctions identifies a change in whether price midpoint travel is being achieved with relatively little volume; follow the newly signed mobility side for twenty-four hours.",
            "side": "long when the completed 14-period EOM crosses strictly above zero; short on the strict reverse crossing",
            "why_distinct": "Range-only candidates ignore participation and volume-flow candidates use signed aggressor flow or turnover levels. HVEOM combines completed midpoint displacement, completed high-low range, and unsigned base volume in the canonical mobility ratio; it uses no taker side, OI, funding, fitted outcome, reused event set, or promoted control.",
            "why_suited_to_volatile_regimes": "completed trailing twenty-four-hour realized variation must rank in its causal upper 35%",
            "why_low_gross9_overlap_is_plausible": "sparse four-hour unsigned volume-normalized midpoint-mobility sign changes are absent from Gross9 primitives",
        },
        external_basis={
            "origin": "Richard W. Arms Jr., canonical Ease of Movement convention",
            "fixed_definition": "one-period midpoint movement times high-low range divided by base volume, followed by a 14 completed-period simple average; trade strict zero crossover",
            "selection_use": "published 14-period smoothing and zero-line interpretation only; positive scale constants omitted because zero crossings are invariant; no incidence or outcomes",
        },
        features={
            "decision_grid": "every exact four-hour UTC boundary",
            "source_bar": "exact aggregation of 240 coherent BTCUSDT one-minute rows [T-4h,T), including summed base volume",
            "midpoint_move": "current completed (high+low)/2 minus immediately prior completed midpoint",
            "raw_eom": "midpoint_move*(high-low)/summed base volume, all denominator and range terms strict positive",
            "smoothed_eom": "simple mean of latest 14 valid consecutive completed raw EOM values",
            "crossover": "current finite nonzero smoothed EOM has strict opposite sign to immediately prior finite nonzero value",
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
            "eom_periods": 14,
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
            "names": ["no_variation_gate", "one_bar_stale_crossover", "direction_flip", "midpoint_only_crossover"],
            "cannot_be_promoted": True,
        },
        source_plan={
            "bars": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close", "volume"],
                "window": ["2020-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        research_boundary={
            "canonical_ease_of_movement_definition_read": True,
            "repository_ease_of_movement_candidate_found": False,
            "prior_range_and_volume_candidate_outcomes_known": True,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "canonical Ease of Movement zero crossover under the requested high-variation regime",
        },
        stopping_rule="Terminal first failure; no EOM period, volume field, crossover, variation, side, hold, clock, subset, control, or other repair.",
    )
    return {**contract, "manifest_hash": canonical_hash(contract)}


def validate(contract: dict[str, Any]) -> None:
    core = {key: value for key, value in contract.items() if key != "manifest_hash"}
    if contract.get("manifest_hash") != canonical_hash(core) or contract != build():
        raise RuntimeError("HVEOM preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = build()
    validate(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(args.output)
