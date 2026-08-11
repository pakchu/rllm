"""Outcome-blind preregistration for HVRQC-8."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVRQC-8"
DEFAULT_OUTPUT = Path("results/high_volatility_realized_quarticity_concentration_relay_preregistration_2026-08-11.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    contract = copy.deepcopy(template.build())
    contract.pop("manifest_hash")
    contract.update(
        protocol_version="high_volatility_realized_quarticity_concentration_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-11",
        mechanism={
            "claim": "During elevated realized variation, a fresh upper-tail onset in scale-free realized-quarticity concentration means the completed eight-hour auction's variance was carried by a small number of disproportionately large shocks rather than diffuse noise; follow the completed block displacement for eight hours.",
            "side": "strict sign of the completed eight-hour close-to-close return at a fresh quarticity-concentration upper-tail onset",
            "why_distinct": "HVRQC aggregates every completed five-minute return to the fourth power and normalizes by squared realized variation. Dominant-shock candidates select one argmax return; realized skew uses signed cubes; semivariance separates signed squares; variance concentration compares fixed time partitions. HVRQC uses no argmax identity, sign inside the concentration statistic, fixed subwindow, volume, flow, OI, funding, fitted outcome, reused event, repair, or promoted control.",
            "why_suited_to_volatile_regimes": "completed eight-hour realized variation must independently rank in its causal upper 35%, while quarticity concentration must newly enter its upper quartile, targeting July-like discontinuous risk transfer",
            "why_low_gross9_overlap_is_plausible": "sparse scale-free fourth-moment concentration onsets on fixed eight-hour clocks are absent from Gross9 primitives",
        },
        external_basis={
            "origin": "realized-quarticity fourth-power-variation construction",
            "fixed_definition": "for 96 completed five-minute log returns, concentration = 96*sum(r^4)/(sum(r^2)^2), finite with positive denominator; unity is the equal-magnitude lower bound",
            "selection_use": "the fourth-power variation construction and scale normalization only; no published directional-return claim, incidence, or outcome is imported",
        },
        features={
            "decision_grid": "every exact eight-hour UTC boundary",
            "source_bars": "exact aggregation of coherent BTCUSDT one-minute rows into five-minute closes",
            "returns": "96 consecutive completed five-minute close-to-close log returns ending at T, requiring 97 closes",
            "variation": "sqrt(sum of squared 96 returns), finite strict positive",
            "quarticity_concentration": "96*sum of fourth powers divided by squared sum of squares, finite",
            "variation_rank": "strict-prior midrank over at most 180 earlier valid eight-hour decisions, minimum 120, current excluded; rank>=0.65",
            "concentration_rank": "strict-prior midrank over the same causal history; rank>=0.75",
            "event": "current concentration rank>=0.75 after the immediately prior valid decision was below 0.75, with current variation rank>=0.65",
            "direction": "strict completed trailing eight-hour close-to-close return",
            "no_imputation": True,
        },
        clock={
            "feature_available": "eight-hour boundary after all source closes",
            "entry": "exact BTCUSDT open five elapsed minutes later",
            "side": "strict trailing eight-hour displacement sign",
            "hold": "8 elapsed hours",
            "reservation": "global half-open first-eligible reservation; exit first on equal open",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        policy={
            "bar_minutes": 5,
            "sample_returns": 96,
            "decision_hours": 8,
            "variation_history_decisions": 180,
            "minimum_history_decisions": 120,
            "variation_rank_min": 0.65,
            "concentration_rank_min": 0.75,
            "entry_delay_minutes": 5,
            "hold_hours": 8,
            "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        diagnostic_controls={
            "names": ["no_variation_gate", "diffuse_variation", "one_decision_stale_onset", "direction_flip", "forced_long"],
            "cannot_be_promoted": True,
        },
        source_plan={
            "bars": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close"],
                "window": ["2023-05-25T00:00:00Z", "2026-08-01T00:00:00Z"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        research_boundary={
            "realized_quarticity_definition_read": True,
            "repository_exact_quarticity_candidate_found": False,
            "adjacent_dominant_shock_skew_semivariance_candidates_known": True,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "fixed normalized fourth-power concentration onset under the requested high-variation regime",
        },
        stopping_rule="Terminal first failure; no bar size, sample, moment power, normalization, ranks, onset, variation, side, hold, clock, subset, threshold, or control repair.",
    )
    return {**contract, "manifest_hash": canonical_hash(contract)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVRQC preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build()
    validate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
