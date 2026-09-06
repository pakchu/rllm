"""Outcome-blind preregistration for HVCVAROIR-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_cross_structure_action_vote as contract

POLICY_ID = "HVCVAROIR-8"
DEFAULT_OUTPUT = Path("results/high_volatility_variance_acceleration_oi_regime_router_preregistration_2026-08-18.json")
COMPONENTS = {
    "expansion_continuation": {
        "preregistration": {"path": "results/high_volatility_causal_variance_acceleration_inventory_relay_preregistration_2026-08-16.json", "sha256": "8cf94eac46378a61a737f3ef857e07b3e728cb150ce3a24b6ae10f0e9839ac64"},
        "support": {"path": "results/high_volatility_causal_variance_acceleration_inventory_relay_support_2026-08-16.json", "sha256": "b45760985d3c9094333c6d4c58400c24488cbe0d8a48fb235a8118306442f649"},
        "clock": {"path": "data/high_volatility_causal_variance_acceleration_inventory_relay_clocks_2020_2026.csv.gz", "sha256": "86f7e4f3b7624396b5a39be132b3c1630087648fbabf2d38a6604863e21b507c"},
    },
    "contraction_reversal": {
        "preregistration": {"path": "results/high_volatility_causal_variance_acceleration_oi_contraction_reversal_preregistration_2026-08-16.json", "sha256": "125f52d7d28a1185b19b088fea590345159a2aaa210874ce9721801d3401e955"},
        "support": {"path": "results/high_volatility_causal_variance_acceleration_oi_contraction_reversal_support_2026-08-16.json", "sha256": "9e1a0acd693940bbcbdd8165ea3544b0355a527682c6dd776294fd083116f439"},
        "clock": {"path": "data/high_volatility_causal_variance_acceleration_oi_contraction_reversal_clocks_2020_2026.csv.gz", "sha256": "9478c34a0edab6acd32b7cf2eac4ebb7fadc8a3b28c6f7e0fa79da9902363af4"},
    },
}

def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def build() -> dict[str, Any]:
    gates = contract.build()
    core = {
        "protocol_version": "high_volatility_variance_acceleration_oi_regime_router_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-18",
        "singleton": True,
        "candidate_family": [POLICY_ID],
        "candidate_family_size": 1,
        "source_incidence_opened": False,
        "outcomes_opened": False,
        "gross9_rows_opened": False,
        "component_artifacts": COMPONENTS,
        "mechanism": {
            "claim": "Under the same frozen upper-tail variance-acceleration state, expanding OI represents fresh inventory that sponsors continuation while contracting OI represents liquidation exhaustion that sponsors reversal.",
            "routing": "OI expansion emits the frozen expansion-continuation side; OI contraction emits the frozen contraction-reversal side; zero OI change is cash",
            "why_volatile": "both branches already require variance-acceleration rank and realized-variation rank at least 0.60",
            "why_distinct": "one mutually exclusive OI-sign router over two source-only frozen branches; no threshold, outcome fit, option data, funding signal, or control promotion",
        },
        "construction": {
            "operator": "disjoint union by strict OI-change sign",
            "expansion_branch": "immutable HVCVARIR-8 clock and side",
            "contraction_branch": "immutable HVCVAROICR-8 clock and side",
            "duplicate_decision": "hard failure",
            "entry": "immutable D+5m",
            "hold": "8 elapsed hours",
            "additional_or_tuned_thresholds": "none",
            "weights": "none",
            "alternatives": "none",
        },
        "stages": gates["stages"],
        "source_support_gates": gates["source_support_gates"],
        "gross9_novelty_gates": gates["gross9_novelty_gates"],
        "economic_gates": gates["economic_gates"],
        "research_boundary": {
            "expansion_incidence_and_train_failure_known": True,
            "contraction_incidence_and_source_side_balance_failure_known": True,
            "contraction_postentry_outcomes_known": False,
            "combined_incidence_opened": False,
            "combined_outcomes_opened": False,
            "classification": "exploratory discovery; not fresh confirmatory evidence",
            "repair_of_component_candidate": False,
            "selection_basis": "predefined inventory sign maps variance acceleration to continuation versus reversal",
        },
        "stopping_rule": "source, Gross9, train/test/eval/final; terminal first failure; no branch, OI sign, rank, side, entry, hold, subset, threshold, weight, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}

def validate(value: dict[str, Any]) -> None:
    if value != build():
        raise RuntimeError("HVCVAROIR-8 preregistration drift")
    for artifacts in COMPONENTS.values():
        for artifact in artifacts.values():
            if sha256(artifact["path"]) != artifact["sha256"]:
                raise RuntimeError("HVCVAROIR-8 component drift")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    value = build(); validate(value); args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"); print(args.output)
