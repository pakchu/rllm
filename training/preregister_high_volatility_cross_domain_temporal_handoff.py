"""Outcome-blind preregistration for the singleton HVTCH-8 policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_cross_structure_action_vote as hvcav


POLICY_ID = "HVTCH-8"
MEMORY_ID = "HVTAMC-8"
TOPOLOGY_ID = "HVITR-8"
VETO_ID = "HVRBRAR-8"
COMPONENT_IDS = (MEMORY_ID, TOPOLOGY_ID, VETO_ID)
CANDIDATE_FAMILY = (POLICY_ID,)
DEFAULT_OUTPUT = Path(
    "results/high_volatility_cross_domain_temporal_handoff_preregistration_2026-08-16.json"
)

COMPONENT_ARTIFACTS: dict[str, dict[str, dict[str, str]]] = {
    "HVTAMC-8": {
        "preregistration": {
            "path": "results/high_volatility_trade_arrival_memory_continuation_preregistration_2026-08-10.json",
            "sha256": "1b4e121fed923d600aa27721fce289c960f76049aa839c0b908b5d6c4f4eaae7",
        },
        "support": {
            "path": "results/high_volatility_trade_arrival_memory_continuation_support_2026-08-10.json",
            "sha256": "2f588ed548f4083ac834193163acb60f9ee71f9ce74acadbb744f5d87ccf3c05",
        },
        "gross9": {
            "path": "results/high_volatility_trade_arrival_memory_continuation_gross9_novelty_2026-08-10.json",
            "sha256": "01c222d91f2f303b5f4b750a7ff3a63d115a11dc50c7af76c0c1aead84d1bbb5",
        },
        "clock": {
            "path": "data/high_volatility_trade_arrival_memory_continuation_clocks_2023_2026.csv.gz",
            "sha256": "c09ee3a0a38a5044c915e3a8fa304de724e9c8dcd8aa6961c32de21e7d519440",
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
    "HVITR-8": {
        "preregistration": {
            "path": "results/high_volatility_intrinsic_topology_ridge_relay_preregistration_2026-08-10.json",
            "sha256": "a5f81522864782a8b88a286abf33039e35995f9776518c9379ae44fdda488c7c",
        },
        "support": {
            "path": "results/high_volatility_intrinsic_topology_ridge_relay_support_2026-08-10.json",
            "sha256": "677efab584ed159019e8c0cf2396cb6ae440aaa98d4be0ac62dd86b51eb9ce48",
        },
        "gross9": {
            "path": "results/high_volatility_intrinsic_topology_ridge_relay_gross9_novelty_2026-08-10.json",
            "sha256": "3001c955f196b158f6417fdeed88c8f1dbba3e5b6f50d095e839807e9ef4473e",
        },
        "clock": {
            "path": "data/high_volatility_intrinsic_topology_ridge_relay_clocks_2023_2026.csv.gz",
            "sha256": "f5c166f1b0c7826147fa004e4d36a5fe57d1fa3414e6a69d77dc21d371ba496e",
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


def temporal_handoff_side(
    previous_memory_side: int, current_topology_side: int, current_veto_side: int
) -> int:
    """Return current topology after lagged agreement and current non-opposition."""
    sides = (previous_memory_side, current_topology_side, current_veto_side)
    if any(type(side) is not int or side not in (-1, 0, 1) for side in sides):
        raise ValueError("HVTCH-8 component sides must be exact -1, 0, or +1")
    if previous_memory_side == 0 or current_topology_side == 0:
        return 0
    if previous_memory_side != current_topology_side:
        return 0
    if current_veto_side == -current_topology_side:
        return 0
    return current_topology_side


def _read_json_object(path: str | Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"HVTCH-8 expected JSON object: {path}")
    return value


def build() -> dict[str, Any]:
    hvcav_contract = hvcav.build()
    core = {
        "protocol_version": "high_volatility_cross_domain_temporal_handoff_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-16",
        "exploratory_discovery": True,
        "fresh_confirmatory_evidence": False,
        "combined_incidence_opened": False,
        "combined_outcomes_opened": False,
        "memory_id": MEMORY_ID,
        "topology_id": TOPOLOGY_ID,
        "veto_id": VETO_ID,
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
            "operator": "high-volatility cross-domain temporal confirmation handoff",
            "decision_join": "HVTAMC-8 at D-8h joined to HVITR-8 and optional HVRBRAR-8 at exact D on the common 00/08/16 UTC grid",
            "timestamp_tolerance": "none",
            "lagged_memory": "HVTAMC-8 must be active at exactly D-8 elapsed hours",
            "current_receiver": "HVITR-8 must be active at D with the same side as lagged HVTAMC-8",
            "current_veto": "cash only when HVRBRAR-8 is active at D opposite the receiver; inactive or same-side HVRBRAR-8 does not veto",
            "emitted_side": "the frozen current HVITR-8 side",
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
        raise RuntimeError("HVTCH-8 preregistration drift")
    if tuple(value.get("candidate_family", ())) != CANDIDATE_FAMILY:
        raise RuntimeError("HVTCH-8 candidate family drift")
    if value.get("component_artifacts") != COMPONENT_ARTIFACTS:
        raise RuntimeError("HVTCH-8 component artifact bindings drift")

    hvcav_contract = hvcav.build()
    for gate in (
        "stages",
        "source_support_gates",
        "gross9_novelty_gates",
        "economic_gates",
    ):
        if value.get(gate) != hvcav_contract[gate]:
            raise RuntimeError(f"HVTCH-8 {gate} drift from HVCAV-8")

    for component, artifacts in COMPONENT_ARTIFACTS.items():
        for artifact_type, artifact in artifacts.items():
            if sha256_file(artifact["path"]) != artifact["sha256"]:
                raise RuntimeError(
                    f"HVTCH-8 {component} {artifact_type} artifact drift"
                )
            if artifact_type == "clock":
                continue
            artifact_value = _read_json_object(artifact["path"])
            expected_scalars = EXPECTED_COMPONENT_SCALARS[component][artifact_type]
            if any(artifact_value.get(key) != expected for key, expected in expected_scalars.items()):
                raise RuntimeError(
                    f"HVTCH-8 {component} {artifact_type} pass scalar drift"
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
