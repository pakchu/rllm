"""Outcome-blind preregistration for the singleton HVCPCV-8 policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_cross_structure_action_vote as hvcav


POLICY_ID = "HVCPCV-8"
PRIMARY_ID = "HVOIPCSR-8"
CONFIRMER_IDS = ("HVRBRAR-8", "HVLTTC-8")
COMPONENT_IDS = (PRIMARY_ID, *CONFIRMER_IDS)
CANDIDATE_FAMILY = (POLICY_ID,)
DEFAULT_OUTPUT = Path(
    "results/high_volatility_cross_domain_primary_confirmation_veto_preregistration_2026-08-16.json"
)

COMPONENT_ARTIFACTS: dict[str, dict[str, dict[str, str]]] = {
    "HVOIPCSR-8": {
        "preregistration": {
            "path": "results/high_volatility_oi_price_coactivity_sponsorship_relay_preregistration_2026-08-10.json",
            "sha256": "7fd82290e7f98c82d59af3df684094b548869ed79047c7fe3c9db1ae16f99e1a",
        },
        "support": {
            "path": "results/high_volatility_oi_price_coactivity_sponsorship_relay_support_2026-08-10.json",
            "sha256": "771024bb0e286e7db4d7955345b66db5a63d45d65fc95f1f0ddd43642d2724db",
        },
        "gross9": {
            "path": "results/high_volatility_oi_price_coactivity_sponsorship_relay_gross9_novelty_2026-08-10.json",
            "sha256": "45927fb0328b5ce48de677f9a383da5ce77f25b2438b76d9086bfc7f04c6617f",
        },
        "clock": {
            "path": "data/high_volatility_oi_price_coactivity_sponsorship_relay_clocks_2023_2026.csv.gz",
            "sha256": "36b6b7baed774eb8cf1ce17224bc37108f9685265ff6650416a4e6775bcf23f4",
        },
    },
    "HVRBRAR-8": {
        "preregistration": {
            "path": "results/high_volatility_block_range_breakout_retest_acceptance_relay_preregistration_2026-08-10.json",
            "sha256": "49f68b06fa9e5e3334f671fc8dd8862af6c888263773f575a1722ada567d2ce1",
        },
        "support": {
            "path": "results/high_volatility_block_range_breakout_retest_acceptance_relay_support_2026-08-10.json",
            "sha256": "64e7abc1b7283cbe3b7e4dad010594f028b021e711ff25ba65eba57a8c2fad67",
        },
        "gross9": {
            "path": "results/high_volatility_block_range_breakout_retest_acceptance_relay_gross9_novelty_2026-08-10.json",
            "sha256": "021614d0e8b1b72e872c18e7c51df3faa5cc70c5805d6d315464656ca5e39a96",
        },
        "clock": {
            "path": "data/high_volatility_block_range_breakout_retest_acceptance_relay_clocks_2023_2026.csv.gz",
            "sha256": "77b16b08f928883c67a408304f5fbb694ced3c378534986efe9020f90cc1fc98",
        },
    },
    "HVLTTC-8": {
        "preregistration": {
            "path": "results/high_volatility_large_ticket_temporal_clustering_relay_preregistration_2026-08-13.json",
            "sha256": "f5da0987ff2d7f8ec0081c3eff806c0b46c44ebd93fb06ec4497163c4b7565f1",
        },
        "support": {
            "path": "results/high_volatility_large_ticket_temporal_clustering_relay_support_2026-08-13.json",
            "sha256": "52271ac02bfef0429155947ea5c05a67aff47cdac9ebb3727015203a537300ce",
        },
        "gross9": {
            "path": "results/high_volatility_large_ticket_temporal_clustering_relay_gross9_novelty_2026-08-13.json",
            "sha256": "d7701615cd82ddeae675a3bec63590ba3ab2402157476d691e8994e0ffae288e",
        },
        "clock": {
            "path": "data/high_volatility_large_ticket_temporal_clustering_relay_clocks_2023_2026.csv.gz",
            "sha256": "7529b389ab5a34a361f16b93b68bd865bd5d430532c3fccea2fc344bd79239d5",
        },
    },
}

EXPECTED_COMPONENT_SCALARS: dict[str, dict[str, dict[str, Any]]] = {
    component: {
        "preregistration": {
            "policy_id": component,
            "singleton": True,
            "outcomes_opened": False,
            "source_incidence_opened": False,
            "gross9_rows_opened": False,
        },
        "support": {
            "policy_id": component,
            "support_passed": True,
            "advance_to_gross9_novelty": True,
            "advance_to_economic_outcomes": False,
            "postentry_return_pnl_execution_price_opened": False,
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


def primary_confirmation_veto_side(
    primary_side: int, confirmer_sides: Sequence[int]
) -> int:
    """Return the frozen primary side, with zero denoting inactive/cash."""
    sides = (primary_side, *confirmer_sides)
    if any(type(side) is not int or side not in (-1, 0, 1) for side in sides):
        raise ValueError("HVCPCV-8 component sides must be exact -1, 0, or +1")
    if len(confirmer_sides) != len(CONFIRMER_IDS):
        raise ValueError("HVCPCV-8 requires exactly two confirmer sides")
    if primary_side == 0:
        return 0
    active_confirmers = [side for side in confirmer_sides if side != 0]
    if primary_side not in active_confirmers:
        return 0
    if any(side == -primary_side for side in active_confirmers):
        return 0
    return primary_side


def _read_json_object(path: str | Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"HVCPCV-8 expected JSON object: {path}")
    return value


def build() -> dict[str, Any]:
    hvcav_contract = hvcav.build()
    core = {
        "protocol_version": "high_volatility_cross_domain_primary_confirmation_veto_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-16",
        "exploratory_discovery": True,
        "fresh_confirmatory_evidence": False,
        "combined_incidence_opened": False,
        "combined_outcomes_opened": False,
        "primary_id": PRIMARY_ID,
        "confirmer_ids": list(CONFIRMER_IDS),
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
            "operator": "high-volatility cross-domain primary confirmation veto",
            "decision_join": "exact equality at 00:00, 08:00, or 16:00 UTC across the three untouched component clocks",
            "timestamp_tolerance": "none",
            "primary_action": "HVOIPCSR-8 exact frozen decision action and side",
            "active_component": "a component is active only when its frozen clock has a non-cash action at that exact decision",
            "confirmation_rule": "emit the primary side only when the primary is active and at least one active confirmer has that exact same side",
            "veto_rule": "cash when any active confirmer has the side opposite the primary",
            "inactive_primary": "cash",
            "no_active_confirmer": "cash",
            "no_same_side_confirmer": "cash",
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
        raise RuntimeError("HVCPCV-8 preregistration drift")
    if tuple(value.get("candidate_family", ())) != CANDIDATE_FAMILY:
        raise RuntimeError("HVCPCV-8 candidate family drift")
    if value.get("component_artifacts") != COMPONENT_ARTIFACTS:
        raise RuntimeError("HVCPCV-8 component artifact bindings drift")

    hvcav_contract = hvcav.build()
    for gate in (
        "stages",
        "source_support_gates",
        "gross9_novelty_gates",
        "economic_gates",
    ):
        if value.get(gate) != hvcav_contract[gate]:
            raise RuntimeError(f"HVCPCV-8 {gate} drift from HVCAV-8")

    for component, artifacts in COMPONENT_ARTIFACTS.items():
        for artifact_type, artifact in artifacts.items():
            if sha256_file(artifact["path"]) != artifact["sha256"]:
                raise RuntimeError(
                    f"HVCPCV-8 {component} {artifact_type} artifact drift"
                )
            if artifact_type == "clock":
                continue
            artifact_value = _read_json_object(artifact["path"])
            expected_scalars = EXPECTED_COMPONENT_SCALARS[component][artifact_type]
            if any(artifact_value.get(key) != expected for key, expected in expected_scalars.items()):
                raise RuntimeError(
                    f"HVCPCV-8 {component} {artifact_type} pass scalar drift"
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
