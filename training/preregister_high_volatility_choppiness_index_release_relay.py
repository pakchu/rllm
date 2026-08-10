"""Outcome-blind preregistration for HVCIR-24."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVCIR-24"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_choppiness_index_release_relay_preregistration_2026-08-11.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    contract = copy.deepcopy(template.build())
    contract.pop("manifest_hash")
    contract.update(
        protocol_version="high_volatility_choppiness_index_release_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-11",
        mechanism={
            "claim": "During elevated realized variation, a canonical fourteen-period Choppiness Index crossing below 38.2 on completed four-hour BTC auctions marks release from range-like price action; follow the sign of the completed fourteen-bar displacement for twenty-four hours.",
            "side": "long when the release bar close is strictly above the close fourteen completed bars earlier; short when strictly below",
            "why_distinct": "Breakout candidates trade price crossing a range boundary, ADX trades directional-movement strength, and Aroon trades extrema recency. HVCIR uses the logarithmic ratio of summed true range to the completed-window high-low span as a regime-release event, with direction supplied only by fixed-horizon displacement; it uses no boundary crossing, OI, flow, volume, funding, fitted outcome, prior event set, or promoted control.",
            "why_suited_to_volatile_regimes": "completed trailing twenty-four-hour realized variation must rank in its causal upper 35%",
            "why_low_gross9_overlap_is_plausible": "sparse four-hour path-efficiency regime releases are absent from Gross9 primitives",
        },
        external_basis={
            "origin": "canonical Choppiness Index convention attributed to E. W. Dreiss",
            "fixed_definition": "14 completed periods; 100*log10(sum(TR,14)/(highest high-lowest low,14))/log10(14); release on strict crossing below 38.2",
            "selection_use": "standard published period, lower Fibonacci threshold, and trend-following interpretation only; no incidence or outcomes",
        },
        features={
            "decision_grid": "every exact four-hour UTC boundary",
            "source_bar": "exact aggregation of 240 coherent BTCUSDT one-minute rows [T-4h,T)",
            "true_range": "max(high-low, abs(high-prior completed close), abs(low-prior completed close))",
            "choppiness_index": "100*log10(sum latest 14 true ranges/(latest 14 highest high-lowest low))/log10(14), finite with strict positive range",
            "release": "current completed-bar Choppiness Index <38.2 and immediately prior completed-bar value >=38.2",
            "direction": "sign(current completed close minus completed close fourteen bars earlier), strict nonzero",
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
            "choppiness_periods": 14,
            "release_threshold": 38.2,
            "direction_lag_bars": 14,
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
            "names": ["no_variation_gate", "one_bar_stale_release", "direction_flip", "range_midpoint_direction"],
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
            "canonical_choppiness_definition_read": True,
            "repository_choppiness_candidate_found": False,
            "prior_breakout_adx_aroon_outcomes_known": True,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "canonical path-efficiency release under the requested high-variation regime",
        },
        stopping_rule="Terminal first failure; no Choppiness period, threshold, direction, variation, side, hold, clock, subset, control, or other repair.",
    )
    return {**contract, "manifest_hash": canonical_hash(contract)}


def validate(contract: dict[str, Any]) -> None:
    core = {key: value for key, value in contract.items() if key != "manifest_hash"}
    if contract.get("manifest_hash") != canonical_hash(core) or contract != build():
        raise RuntimeError("HVCIR preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = build()
    validate(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(args.output)
