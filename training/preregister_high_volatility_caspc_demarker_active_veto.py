"""Outcome-blind preregistration for the singleton HVCDEMV-8 policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_cross_structure_action_vote as hvcav


POLICY_ID = "HVCDEMV-8"
ACTION_IDS = ("HVCASPC-8", "HVDEM-24")
COMPONENT_IDS = ACTION_IDS
CANDIDATE_FAMILY = (POLICY_ID,)
DEFAULT_OUTPUT = Path(
    "results/high_volatility_caspc_demarker_active_veto_preregistration_2026-08-16.json"
)

COMPONENT_ARTIFACTS: dict[str, dict[str, dict[str, str]]] = {
    "HVDEM-24": {
        "preregistration": {"path": "results/high_volatility_demarker_extreme_reentry_relay_preregistration_2026-08-11.json", "sha256": "a66976c7a542de9865b417d3eef82f79c2e19eebb11e5c66339036db19e84dae"},
        "support": {"path": "results/high_volatility_demarker_extreme_reentry_relay_support_2026-08-11.json", "sha256": "bf74e61e59073c2f5e03012523652eff86191abbb99c66946185678b41c33e70"},
        "gross9": {"path": "results/high_volatility_demarker_extreme_reentry_relay_gross9_novelty_2026-08-11.json", "sha256": "1f173f9d3caa56813b9a98020a2092a0f869718ff15feafe73103f56808583b7"},
        "clock": {"path": "data/high_volatility_demarker_extreme_reentry_relay_clocks_2023_2026.csv.gz", "sha256": "c701ebfeaee9dda74dbf9f8eed3d0c1fd29dd0ab3473c65d3bfa1409da1c5368"},
    },
    "HVCASPC-8": {
        "preregistration": {"path": "results/high_volatility_cross_alt_serial_persistence_consensus_relay_preregistration_2026-08-13.json", "sha256": "be77dfa8d83dfcb54171da4cf3263c57336fc629dd8f7b63ff8e8b33a860784a"},
        "support": {"path": "results/high_volatility_cross_alt_serial_persistence_consensus_relay_support_2026-08-13.json", "sha256": "53f401d5ab9a499838c128b7023b4e420925455ca7e858e5a4db5c6e4f83d52e"},
        "gross9": {"path": "results/high_volatility_cross_alt_serial_persistence_consensus_relay_gross9_novelty_2026-08-13.json", "sha256": "61c7fe10f3b331528315e8d7625aff7c8fb8ff1c82b6c553aa34f19ad89832f3"},
        "clock": {"path": "data/high_volatility_cross_alt_serial_persistence_consensus_relay_clocks_2023_2026.csv.gz", "sha256": "bffadb70bea5d96fc779c641dbe3a1f50ba94fdf76a4e950d778a32a2b64b085"},
    },
}
COMPONENT_ARTIFACTS = {component: COMPONENT_ARTIFACTS[component] for component in COMPONENT_IDS}

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


def active_state_veto_side(primary: int, confirmer: int) -> int:
    """Keep the primary side unless an active confirmer is opposite."""
    sides = (primary, confirmer)
    if any(type(side) is not int or side not in (-1, 0, 1) for side in sides):
        raise ValueError("HVCDEMV-8 component sides must be exact -1, 0, or +1")
    if primary == 0 or confirmer == -primary:
        return 0
    return primary


def _read_json_object(path: str | Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"HVCDEMV-8 expected JSON object: {path}")
    return value


def build() -> dict[str, Any]:
    hvcav_contract = hvcav.build()
    core = {
        "protocol_version": "high_volatility_caspc_demarker_active_veto_v1",
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
            "operator": "high-volatility oscillator-family active-position opposite-side veto",
            "decision_join": "each HVCASPC-8 entry E joined to the unique frozen HVDEM-24 position satisfying veto_entry<=E<veto_exit, if any",
            "timestamp_tolerance": "none",
            "primary_action": "HVCASPC-8",
            "veto_action": "HVDEM-24",
            "routing_rule": "emit each frozen HVCASPC-8 side unless a causally entered HVDEM-24 position is active at the HVELRSI entry and has the opposite frozen side",
            "conflict_rule": "opposite active HVDEM-24 vetoes to cash; same-side or no active HVDEM-24 leaves HVCASPC-8 unchanged",
            "emitted_side": "the frozen HVCASPC-8 side when not vetoed; otherwise cash",
            "entry": "feature_available_time + 5 minutes",
            "hold": "8 elapsed hours",
            "weights": "none",
            "priority": "none; HVCASPC-8 is the fixed primary and HVDEM-24 is only an opposite-side veto",
            "alternatives": "none",
            "additional_or_tuned_thresholds": "none",
            "component_formula_threshold_clock_mutability": "immutable",
            "independent_from_HVCAV_8": True,
            "HVCAV_subset_or_quorum_repair": "none",
        },
        "clock": {
            "decisions": "exact 03:00, 11:00 and 19:00 UTC primary boundaries",
            "entry": "exact feature-available D+5m entry",
            "hold": "8 elapsed hours",
            "component_clocks": "untouched exact frozen primary clocks",
            "timestamp_tolerance": "none",
            "funding": "not a signal input; exact held settlements only after source and Gross9 pass",
        },
        "stages": hvcav_contract["stages"],
        "source_support_gates": hvcav_contract["source_support_gates"],
        "operator_activation_gate": {
            "minimum_opposite_vetoes_train": 1,
            "minimum_opposite_vetoes_full": 3,
            "must_pass_before_gross9": True,
        },
        "gross9_novelty_gates": hvcav_contract["gross9_novelty_gates"],
        "economic_gates": hvcav_contract["economic_gates"],
        "research_boundary": {
            "all_component_prior_outcomes_known": True,
            "component_selection_used_prior_train_outcomes": True,
            "component_selection_basis": "HVDEM-24 and HVCASPC-8 independently passed all frozen train economics gates; HVCASPC-8 is primary and the causally active-position opposite veto is fixed before combined incidence or outcomes",
            "component_test_eval_final_outcomes_used": False,
            "prior_HVCAV_incidence_known": True,
            "prior_HVCAV_outcomes_known": True,
            "archived_component_later_stage_artifacts_exist": True,
            "archived_component_later_stage_outcomes_used_for_selection": False,
            "known_component_or_HVCAV_evidence_used_to_alter_components": True,
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
        raise RuntimeError("HVCDEMV-8 preregistration drift")
    if tuple(value.get("candidate_family", ())) != CANDIDATE_FAMILY:
        raise RuntimeError("HVCDEMV-8 candidate family drift")
    if value.get("component_artifacts") != COMPONENT_ARTIFACTS:
        raise RuntimeError("HVCDEMV-8 component artifact bindings drift")

    hvcav_contract = hvcav.build()
    for gate in (
        "stages",
        "source_support_gates",
        "gross9_novelty_gates",
        "economic_gates",
    ):
        if value.get(gate) != hvcav_contract[gate]:
            raise RuntimeError(f"HVCDEMV-8 {gate} drift from HVCAV-8")

    for component, artifacts in COMPONENT_ARTIFACTS.items():
        for artifact_type, artifact in artifacts.items():
            if sha256_file(artifact["path"]) != artifact["sha256"]:
                raise RuntimeError(
                    f"HVCDEMV-8 {component} {artifact_type} artifact drift"
                )
            if artifact_type == "clock":
                continue
            artifact_value = _read_json_object(artifact["path"])
            expected_scalars = EXPECTED_COMPONENT_SCALARS[component][artifact_type]
            if any(artifact_value.get(key) != expected for key, expected in expected_scalars.items()):
                raise RuntimeError(
                    f"HVCDEMV-8 {component} {artifact_type} pass scalar drift"
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
