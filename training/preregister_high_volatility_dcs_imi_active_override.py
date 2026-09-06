"""Outcome-blind preregistration for the singleton HVDIMIO-8 policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_cross_structure_action_vote as hvcav


POLICY_ID = "HVDIMIO-8"
ACTION_IDS = ("HVDCS-8", "HVIMI-24")
COMPONENT_IDS = ACTION_IDS
CANDIDATE_FAMILY = (POLICY_ID,)
DEFAULT_OUTPUT = Path(
    "results/high_volatility_dcs_imi_active_override_preregistration_2026-08-16.json"
)

COMPONENT_ARTIFACTS: dict[str, dict[str, dict[str, str]]] = {
    "HVIMI-24": {
        "preregistration": {"path": "results/high_volatility_intraday_momentum_index_reentry_relay_preregistration_2026-08-11.json", "sha256": "f6446d3e6585fe56eaaab1ff6897e6e2659c3ba2c6bff255dd423cfee58fb1f7"},
        "support": {"path": "results/high_volatility_intraday_momentum_index_reentry_relay_support_2026-08-11.json", "sha256": "a96128d56f7c125dca5c0c8c865b7cd8b728423a9aa64e88d1434a5b2d85c47c"},
        "gross9": {"path": "results/high_volatility_intraday_momentum_index_reentry_relay_gross9_novelty_2026-08-11.json", "sha256": "791de02675ba9fc98c234cb65f6ffc5ee53e70d70adeaaa64665dcc419105883"},
        "clock": {"path": "data/high_volatility_intraday_momentum_index_reentry_relay_clocks_2023_2026.csv.gz", "sha256": "133f92a81a6f665a5a895eb1adbcd700b5a7a7736470edc8244b7ce750519cc2"},
    },
    "HVDCS-8": {
        "preregistration": {"path": "results/high_volatility_directional_change_scarcity_relay_preregistration_2026-08-10.json", "sha256": "746594b628f0ee377de6847479d8a84bf58ae4f1a06ab2cb23dc30986b89e325"},
        "support": {"path": "results/high_volatility_directional_change_scarcity_relay_support_2026-08-10.json", "sha256": "f0dc6e4a24792caba7a1967280158d8aed55de27a0c297146b2ed20b0c94d8dc"},
        "gross9": {"path": "results/high_volatility_directional_change_scarcity_relay_gross9_novelty_2026-08-10.json", "sha256": "0aa717adc52752a6c10cea752f9482a12b4eaf27ff12699999dfd48ac9e90232"},
        "clock": {"path": "data/high_volatility_directional_change_scarcity_relay_clocks_2023_2026.csv.gz", "sha256": "055d86ba9bcb9f619a3362a242febceb28d2f1b07ebef74d5d70b7393dc542fc"},
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


def active_state_override_side(primary: int, confirmer: int) -> int:
    """Use the active longer-horizon side when it opposes the primary."""
    sides = (primary, confirmer)
    if any(type(side) is not int or side not in (-1, 0, 1) for side in sides):
        raise ValueError("HVDIMIO-8 component sides must be exact -1, 0, or +1")
    if primary == 0:
        return 0
    if confirmer == -primary:
        return confirmer
    return primary


def _read_json_object(path: str | Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"HVDIMIO-8 expected JSON object: {path}")
    return value


def build() -> dict[str, Any]:
    hvcav_contract = hvcav.build()
    core = {
        "protocol_version": "high_volatility_dcs_imi_active_override_v1",
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
            "operator": "high-volatility oscillator-family active-position opposite-side override",
            "decision_join": "each HVDCS-8 entry E joined to the unique frozen HVIMI-24 position satisfying state_entry<=E<state_exit, if any",
            "timestamp_tolerance": "none",
            "primary_action": "HVDCS-8",
            "override_action": "HVIMI-24",
            "routing_rule": "emit each frozen HVDCS-8 side unless a causally entered HVIMI-24 position is active at the HVDCS entry and has the opposite frozen side, then emit HVIMI-24",
            "conflict_rule": "opposite active HVIMI-24 overrides the primary direction; same-side or no active HVIMI-24 leaves HVDCS-8 unchanged",
            "emitted_side": "the frozen HVDCS-8 side unless opposite active HVIMI-24 overrides it",
            "entry": "feature_available_time + 5 minutes",
            "hold": "8 elapsed hours",
            "weights": "none",
            "priority": "none; HVDCS-8 is the fixed primary and HVIMI-24 is only an opposite-side override",
            "alternatives": "none",
            "additional_or_tuned_thresholds": "none",
            "component_formula_threshold_clock_mutability": "immutable",
            "independent_from_HVCAV_8": True,
            "HVCAV_subset_or_quorum_repair": "none",
        },
        "clock": {
            "decisions": "exact HVDCS-8 boundaries 00/08/16 UTC",
            "entry": "exact feature-available D+5m entry",
            "hold": "8 elapsed hours",
            "component_clocks": "untouched exact frozen primary clocks",
            "timestamp_tolerance": "none",
            "funding": "not a signal input; exact held settlements only after source and Gross9 pass",
        },
        "stages": hvcav_contract["stages"],
        "source_support_gates": hvcav_contract["source_support_gates"],
        "operator_activation_gate": {
            "minimum_opposite_overrides_train": 1,
            "minimum_opposite_overrides_full": 3,
            "must_pass_before_gross9": True,
        },
        "gross9_novelty_gates": hvcav_contract["gross9_novelty_gates"],
        "economic_gates": hvcav_contract["economic_gates"],
        "research_boundary": {
            "all_component_prior_outcomes_known": True,
            "component_selection_used_prior_train_outcomes": True,
            "component_selection_basis": "HVIMI-24 and HVDCS-8 independently passed all frozen train economics gates; HVDCS-8 is primary and the causally active-position opposite override is fixed before combined incidence or outcomes",
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
        raise RuntimeError("HVDIMIO-8 preregistration drift")
    if tuple(value.get("candidate_family", ())) != CANDIDATE_FAMILY:
        raise RuntimeError("HVDIMIO-8 candidate family drift")
    if value.get("component_artifacts") != COMPONENT_ARTIFACTS:
        raise RuntimeError("HVDIMIO-8 component artifact bindings drift")

    hvcav_contract = hvcav.build()
    for gate in (
        "stages",
        "source_support_gates",
        "gross9_novelty_gates",
        "economic_gates",
    ):
        if value.get(gate) != hvcav_contract[gate]:
            raise RuntimeError(f"HVDIMIO-8 {gate} drift from HVCAV-8")

    for component, artifacts in COMPONENT_ARTIFACTS.items():
        for artifact_type, artifact in artifacts.items():
            if sha256_file(artifact["path"]) != artifact["sha256"]:
                raise RuntimeError(
                    f"HVDIMIO-8 {component} {artifact_type} artifact drift"
                )
            if artifact_type == "clock":
                continue
            artifact_value = _read_json_object(artifact["path"])
            expected_scalars = EXPECTED_COMPONENT_SCALARS[component][artifact_type]
            if any(artifact_value.get(key) != expected for key, expected in expected_scalars.items()):
                raise RuntimeError(
                    f"HVDIMIO-8 {component} {artifact_type} pass scalar drift"
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
