"""Outcome-blind preregistration for the singleton HVCPO-8 policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_cross_structure_action_vote as hvcav


POLICY_ID = "HVCPO-8"
ACTION_IDS = ("HVMRSR-8", "HVSARR-8")
COMPONENT_IDS = ACTION_IDS
CANDIDATE_FAMILY = (POLICY_ID,)
DEFAULT_OUTPUT = Path(
    "results/high_volatility_complementary_persistence_opposition_override_preregistration_2026-08-16.json"
)

COMPONENT_ARTIFACTS: dict[str, dict[str, dict[str, str]]] = {
    "HVMRSR-8": {
        "preregistration": {
            "path": "results/high_volatility_median_return_shift_relay_preregistration_2026-08-10.json",
            "sha256": "29c87e3b08c500c2ffa0371c94849a5d84dcbf99ae0eb3f9df25922c3393863b",
        },
        "support": {
            "path": "results/high_volatility_median_return_shift_relay_support_2026-08-10.json",
            "sha256": "41f8754ad340d062cb550151d2017c3567b12661671573fc7f9307129e091b46",
        },
        "gross9": {
            "path": "results/high_volatility_median_return_shift_relay_gross9_novelty_2026-08-10.json",
            "sha256": "83fb5e444b8d19519c833a3f13021bb7ad89189bd5ab1a97d8403ba96130bdcf",
        },
        "clock": {
            "path": "data/high_volatility_median_return_shift_relay_clocks_2023_2026.csv.gz",
            "sha256": "71adf3d170a68a26928d6d4600d8fb46046b7adddb64f464948355df41f98d5c",
        },
    },
    "HVRQA-8": {
        "preregistration": {
            "path": "results/high_volatility_recurrence_determinism_relay_preregistration_2026-08-12.json",
            "sha256": "fb1ebcf09a0b9fb12861e6ed03ae8706b131773cd0a69905ca647200e7787339",
        },
        "support": {
            "path": "results/high_volatility_recurrence_determinism_relay_support_2026-08-12.json",
            "sha256": "43b7488f922c67a3256033408b5de2f51ed82c824e2848a23baa4799efd7a866",
        },
        "gross9": {
            "path": "results/high_volatility_recurrence_determinism_relay_gross9_novelty_2026-08-12.json",
            "sha256": "209c5a694342f4cbd63ed3257c2559eedbba1bbb60565eee98368e63bcd4aada",
        },
        "clock": {
            "path": "data/high_volatility_recurrence_determinism_relay_clocks_2023_2026.csv.gz",
            "sha256": "c81323fe967557989d13e309e9919029bdc5cf9bdea749d5a8551ad4e43260dc",
        },
    },
    "HVSARR-8": {
        "preregistration": {
            "path": "results/high_volatility_speculative_alt_rotation_relay_preregistration_2026-08-12.json",
            "sha256": "9c014fafa912faaa26edd783b9e767396c4ec8eafaf3cca94571ffded8813c0c",
        },
        "support": {
            "path": "results/high_volatility_speculative_alt_rotation_relay_support_2026-08-12.json",
            "sha256": "ad9830bbc54e7512a593a0b5803f425015aadad26d0aeb88f837961648fea41e",
        },
        "gross9": {
            "path": "results/high_volatility_speculative_alt_rotation_relay_gross9_novelty_2026-08-12.json",
            "sha256": "0fd22a847e0fd81a2de294f241571e22abe4864f720b3acd0b6b98410d1ee27a",
        },
        "clock": {
            "path": "data/high_volatility_speculative_alt_rotation_relay_clocks_2023_2026.csv.gz",
            "sha256": "f69e809595fb55dff5ffe4f04e1f00e3804d85d7909ae9cfc47a574256771a10",
        },
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


def override_side(primary: int, confirmer: int) -> int:
    """Use the opposite confirmer when it conflicts with the primary."""
    sides = (primary, confirmer)
    if any(type(side) is not int or side not in (-1, 0, 1) for side in sides):
        raise ValueError("HVCPO-8 component sides must be exact -1, 0, or +1")
    if primary == 0:
        return 0
    if confirmer == -primary:
        return confirmer
    return primary


def _read_json_object(path: str | Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"HVCPO-8 expected JSON object: {path}")
    return value


def build() -> dict[str, Any]:
    hvcav_contract = hvcav.build()
    core = {
        "protocol_version": "high_volatility_complementary_persistence_opposition_override_v1",
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
            "operator": "high-volatility complementary-persistence opposition override",
            "decision_join": "exact same-D join of the two untouched action clocks on the common 01/09/17 UTC grid",
            "timestamp_tolerance": "none",
            "primary_action": "HVMRSR-8",
            "override_action": "HVSARR-8",
            "routing_rule": "consider only active HVMRSR-8 decisions; emit its frozen side unless HVSARR-8 is active at exact D with the opposite frozen side, in which case emit HVSARR-8",
            "conflict_rule": "opposite HVSARR-8 replaces the primary side; same-side or inactive HVSARR-8 leaves HVMRSR-8 unchanged",
            "emitted_side": "the frozen HVMRSR-8 side except at exact opposite conflicts, where the frozen HVSARR-8 side is emitted",
            "entry": "decision_time + 5 minutes",
            "hold": "8 elapsed hours",
            "weights": "none",
            "priority": "none; HVMRSR-8 defines incidence and HVSARR-8 only overrides exact opposite conflicts",
            "alternatives": "none",
            "additional_or_tuned_thresholds": "none",
            "component_formula_threshold_clock_mutability": "immutable",
            "independent_from_HVCAV_8": True,
            "HVCAV_subset_or_quorum_repair": "none",
        },
        "clock": {
            "decisions": "exact 01:00, 09:00 and 17:00 UTC",
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
            "all_component_prior_outcomes_known": True,
            "component_selection_used_prior_train_outcomes": True,
            "component_selection_basis": "HVMRSR-8 passed every aggregate train gate except first-half positivity; HVSARR-8 had both train halves positive on the identical 01/09/17 UTC grid; the opposition override is frozen from those aggregate train facts",
            "component_test_eval_final_outcomes_used": False,
            "prior_HVCAV_incidence_known": True,
            "prior_HVCAV_outcomes_known": True,
            "prior_HVCPR_pair_incidence_known": True,
            "prior_HVCPR_outcomes_opened": False,
            "prior_HVCPSR_source_incidence_known": True,
            "prior_HVCPSR_gross9_novelty_known": True,
            "prior_HVCPSR_outcomes_opened": False,
            "prior_HVCPV_source_and_gross9_known": True,
            "prior_HVCPV_train_outcomes_known": True,
            "prior_HVCPV_terminal_failure_known": "failed only exact calendar-half positivity; no later outcomes opened",
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
        raise RuntimeError("HVCPO-8 preregistration drift")
    if tuple(value.get("candidate_family", ())) != CANDIDATE_FAMILY:
        raise RuntimeError("HVCPO-8 candidate family drift")
    if value.get("component_artifacts") != COMPONENT_ARTIFACTS:
        raise RuntimeError("HVCPO-8 component artifact bindings drift")

    hvcav_contract = hvcav.build()
    for gate in (
        "stages",
        "source_support_gates",
        "gross9_novelty_gates",
        "economic_gates",
    ):
        if value.get(gate) != hvcav_contract[gate]:
            raise RuntimeError(f"HVCPO-8 {gate} drift from HVCAV-8")

    for component, artifacts in COMPONENT_ARTIFACTS.items():
        for artifact_type, artifact in artifacts.items():
            if sha256_file(artifact["path"]) != artifact["sha256"]:
                raise RuntimeError(
                    f"HVCPO-8 {component} {artifact_type} artifact drift"
                )
            if artifact_type == "clock":
                continue
            artifact_value = _read_json_object(artifact["path"])
            expected_scalars = EXPECTED_COMPONENT_SCALARS[component][artifact_type]
            if any(artifact_value.get(key) != expected for key, expected in expected_scalars.items()):
                raise RuntimeError(
                    f"HVCPO-8 {component} {artifact_type} pass scalar drift"
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
