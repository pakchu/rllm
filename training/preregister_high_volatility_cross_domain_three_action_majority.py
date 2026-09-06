"""Outcome-blind preregistration for the singleton HVCM3-8 policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_cross_structure_action_vote as hvcav


POLICY_ID = "HVCM3-8"
ACTION_IDS = ("HVECE-8", "HVIVDR-8", "HVSTBR-8")
COMPONENT_IDS = ACTION_IDS
CANDIDATE_FAMILY = (POLICY_ID,)
DEFAULT_OUTPUT = Path(
    "results/high_volatility_cross_domain_three_action_majority_preregistration_2026-08-16.json"
)

COMPONENT_ARTIFACTS: dict[str, dict[str, dict[str, str]]] = {
    "HVECE-8": {
        "preregistration": {
            "path": "results/high_volatility_execution_count_elasticity_relay_preregistration_2026-08-12.json",
            "sha256": "62094724e9d4bcb58ec3f8582cb3ce0e296debbbbfc04bb04dae2332b5ad7d47",
        },
        "support": {
            "path": "results/high_volatility_execution_count_elasticity_relay_support_2026-08-12.json",
            "sha256": "e493f0c25654b00b0353f629bcc7495b849175557bfefd452f91674955d24e91",
        },
        "gross9": {
            "path": "results/high_volatility_execution_count_elasticity_relay_gross9_novelty_2026-08-12.json",
            "sha256": "b5435f22ef269e5a8323ea42e1697ac986aa43f9aa4c1172eb324a57afd74006",
        },
        "clock": {
            "path": "data/high_volatility_execution_count_elasticity_relay_clocks_2023_2026.csv.gz",
            "sha256": "5f280288026c9df24a1821a28209b8e8b0f4fa36c1ca209429c0f652a6e5ee48",
        },
    },
    "HVSTBR-8": {
        "preregistration": {
            "path": "results/high_volatility_standardized_tail_breadth_relay_preregistration_2026-08-10.json",
            "sha256": "6d70486179ac1ac8b35c53fbff02a580e526207216f286c746ef7659c45b69ca",
        },
        "support": {
            "path": "results/high_volatility_standardized_tail_breadth_relay_support_2026-08-10.json",
            "sha256": "07ae089dfb634bb7e5092def89f5536897d1610b26f9afc79bb54321d62b8fc1",
        },
        "gross9": {
            "path": "results/high_volatility_standardized_tail_breadth_relay_gross9_novelty_2026-08-10.json",
            "sha256": "168ad6d0120618184aaded4cd9932597e6220c620613c0acd5db4561922f6f40",
        },
        "clock": {
            "path": "data/high_volatility_standardized_tail_breadth_relay_clocks_2023_2026.csv.gz",
            "sha256": "05fbef229d0b8c2d0e14e89fb3f935718c7cefb7128de7fc1d240b9849eb10f9",
        },
    },
    "HVIVDR-8": {
        "preregistration": {
            "path": "results/high_volatility_intraday_variance_dispersion_relay_preregistration_2026-08-12.json",
            "sha256": "7193b15406e0d88266e61fe81131993350e17a2b7a1ee448b20b56f1abd23328",
        },
        "support": {
            "path": "results/high_volatility_intraday_variance_dispersion_relay_support_2026-08-12.json",
            "sha256": "8475fbf443ee133fa757dae5440b5a7dee82fbbcfd2c20d8315b77ecae5f7649",
        },
        "gross9": {
            "path": "results/high_volatility_intraday_variance_dispersion_relay_gross9_novelty_2026-08-12.json",
            "sha256": "ab501c086480be32733f2b43358dacbab33f4650e3f408b4f910070cd5ec88bb",
        },
        "clock": {
            "path": "data/high_volatility_intraday_variance_dispersion_relay_clocks_2023_2026.csv.gz",
            "sha256": "05d37053dde59552140cc26ccd841a9a942b32d90c14e05cf1fd179b52486aa6",
        },
    },
}

EXPECTED_COMPONENT_SCALARS: dict[str, dict[str, dict[str, Any]]] = {
    component: {
        "preregistration": {
            "policy_id": component,
            "singleton": True,
            "gross9_rows_opened": False,
        },
        "support": {
            "policy_id": component,
            "support_passed": True,
            "advance_to_gross9_novelty": True,
            "advance_to_economic_outcomes": False,
            "gross9_rows_opened": False,
        },
        "gross9": {
            "policy_id": component,
            "source_support_passed": True,
            "every_gross9_sleeve_passed": True,
            "gross9_novelty_status": "passed",
            "advance_to_economic_outcomes": True,
        },
    }
    for component in COMPONENT_IDS
}


def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def three_action_majority_side(first: int, second: int, third: int) -> int:
    """Return a strict majority among at least two active frozen actions."""
    sides = (first, second, third)
    if any(type(side) is not int or side not in (-1, 0, 1) for side in sides):
        raise ValueError("HVCM3-8 component sides must be exact -1, 0, or +1")
    active = [side for side in sides if side != 0]
    if len(active) < 2:
        return 0
    total = sum(active)
    if total == 0:
        return 0
    return 1 if total > 0 else -1


def _read_json_object(path: str | Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"HVCM3-8 expected JSON object: {path}")
    return value


def build() -> dict[str, Any]:
    hvcav_contract = hvcav.build()
    core = {
        "protocol_version": "high_volatility_cross_domain_three_action_majority_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-16",
        "exploratory_discovery": True,
        "fresh_confirmatory_evidence": False,
        "combined_incidence_opened": False,
        "combined_outcomes_opened": False,
        "action_ids": list(ACTION_IDS),
        "component_ids": list(COMPONENT_IDS),
        "candidate_family": list(CANDIDATE_FAMILY),
        "candidate_family_size": 1,
        "single_candidate_only": True,
        "familywise_multiplicity": "none; independent singleton family; no Bonferroni adjustment",
        "component_artifacts": COMPONENT_ARTIFACTS,
        "component_gate_status": {
            component: {
                "source_support_passed": True,
                "gross9_novelty_passed": True,
                "primary_clock_immutable": True,
            }
            for component in COMPONENT_IDS
        },
        "construction": {
            "operator": "high-volatility cross-domain three-action strict majority",
            "decision_join": "exact same-D join of the three untouched action clocks on the common 00/08/16 UTC grid",
            "timestamp_tolerance": "none",
            "active_action": "an action contributes one vote only when its frozen clock is active at exact D",
            "quorum": "at least two active actions",
            "vote_rule": "strict nonzero majority of active +1/-1 sides",
            "tie_or_no_quorum": "cash",
            "emitted_side": "the strict majority side",
            "entry": "decision_time + 5 minutes",
            "hold": "8 elapsed hours",
            "weights": "none",
            "priority": "none",
            "alternatives": "none",
            "additional_or_tuned_thresholds": "none",
            "component_formula_threshold_clock_mutability": "immutable",
            "independent_from_HVCAV_8": True,
            "HVCAV_subset_or_quorum_repair": "none",
        },
        "clock": {
            "decisions": "exact 00:00, 08:00 and 16:00 UTC",
            "entry": "exact decision D+5m entry",
            "hold": "8 elapsed hours",
            "component_clocks": "untouched exact frozen primary clocks",
            "timestamp_tolerance": "none",
            "funding": "not a signal input; exact held settlements only after source and Gross9 pass",
        },
        "stages": hvcav_contract["stages"],
        "source_support_gates": hvcav_contract["source_support_gates"],
        "gross9_novelty_gates": hvcav_contract["gross9_novelty_gates"],
        "economic_gates": hvcav_contract["economic_gates"],
        "research_boundary": {
            "all_three_component_prior_outcomes_known": True,
            "prior_HVCAV_incidence_known": True,
            "prior_HVCAV_outcomes_known": True,
            "known_component_or_HVCAV_evidence_used_to_alter_components": False,
            "known_HVCAV_evidence_used_for_subset_or_quorum_repair": False,
            "combined_incidence_opened": False,
            "combined_postentry_returns_or_pnl_opened": False,
            "economics_artifacts_read_during_validation": False,
            "classification": "exploratory discovery; not fresh confirmatory evidence",
        },
        "stopping_rule": "Evaluate the single frozen candidate through source support, Gross9 novelty, and train/test/eval/final economics; stop on first failure with no substitution, threshold, weight, priority, alternative, component, direction, entry, side, hold, clock, subset, quorum, veto, confirmation, or HVCAV repair.",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: Mapping[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVCM3-8 preregistration drift")
    if tuple(value.get("candidate_family", ())) != CANDIDATE_FAMILY:
        raise RuntimeError("HVCM3-8 candidate family drift")
    if value.get("component_artifacts") != COMPONENT_ARTIFACTS:
        raise RuntimeError("HVCM3-8 component artifact bindings drift")

    hvcav_contract = hvcav.build()
    for gate in (
        "stages",
        "source_support_gates",
        "gross9_novelty_gates",
        "economic_gates",
    ):
        if value.get(gate) != hvcav_contract[gate]:
            raise RuntimeError(f"HVCM3-8 {gate} drift from HVCAV-8")

    for component, artifacts in COMPONENT_ARTIFACTS.items():
        for artifact_type, artifact in artifacts.items():
            if sha256_file(artifact["path"]) != artifact["sha256"]:
                raise RuntimeError(
                    f"HVCM3-8 {component} {artifact_type} artifact drift"
                )
            if artifact_type == "clock":
                continue
            artifact_value = _read_json_object(artifact["path"])
            expected_scalars = EXPECTED_COMPONENT_SCALARS[component][artifact_type]
            if any(artifact_value.get(key) != expected for key, expected in expected_scalars.items()):
                raise RuntimeError(
                    f"HVCM3-8 {component} {artifact_type} pass scalar drift"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(args.output)
