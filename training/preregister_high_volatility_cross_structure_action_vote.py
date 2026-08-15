"""Outcome-blind preregistration for the single HVCAV-8 action vote."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_cross_structure_priority_router as hvcspr
from training import preregister_high_volatility_state_ordered_filter as hvsof


POLICY_ID = "HVCAV-8"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_cross_structure_action_vote_preregistration_2026-08-16.json"
)
ACTION_ORDER = ("CARSC-8", "HVCMMI-8", "HVIABR-8", "HVTFR-8", "HVSVF-8")
ELIGIBILITY_ID = "HVTCCR-8"
CANDIDATE_FAMILY = (POLICY_ID,)
FAMILY_SIZE = 1
WEEKLY_SIGNFLIP_P_MAX = 0.10

ACTION_ARTIFACTS: dict[str, dict[str, dict[str, str]]] = {
    action: (
        hvcspr.ACTION_ARTIFACTS[action]
        if action in hvcspr.ACTION_ARTIFACTS
        else hvsof.ACTION_ARTIFACTS[action]
    )
    for action in ACTION_ORDER
}
ELIGIBILITY_ARTIFACTS = {ELIGIBILITY_ID: hvsof.ELIGIBILITY_ARTIFACTS[ELIGIBILITY_ID]}


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


def action_vote_side(active_sides: Sequence[int]) -> int:
    """Return the frozen strict-majority side, with zero denoting cash."""
    if any(type(side) is not int or side not in (-1, 1) for side in active_sides):
        raise ValueError("HVCAV-8 active action sides must be exact +1 or -1")
    if len(active_sides) < 2:
        return 0
    signed_vote = sum(active_sides)
    if signed_vote > 0:
        return 1
    if signed_vote < 0:
        return -1
    return 0


def build() -> dict[str, Any]:
    prior = hvsof.build()
    eligibility_condition = dict(prior["eligibility_conditions"][ELIGIBILITY_ID])
    economic_gates = dict(prior["economic_gates"])
    economic_gates["weekly_signflip_one_sided_p_max"] = WEEKLY_SIGNFLIP_P_MAX
    core = {
        "protocol_version": "high_volatility_cross_structure_action_vote_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-16",
        "exploratory_discovery": True,
        "fresh_confirmatory_evidence": False,
        "consensus_incidence_opened": False,
        "consensus_outcomes_opened": False,
        "action_order": list(ACTION_ORDER),
        "eligibility_id": ELIGIBILITY_ID,
        "candidate_family": list(CANDIDATE_FAMILY),
        "candidate_family_size": FAMILY_SIZE,
        "single_candidate_only": True,
        "action_artifacts": ACTION_ARTIFACTS,
        "eligibility_artifacts": ELIGIBILITY_ARTIFACTS,
        "component_gate_status": {
            "actions": {
                action: {
                    "source_support_passed": True,
                    "gross9_novelty_passed": True,
                    "primary_clock_immutable": True,
                }
                for action in ACTION_ORDER
            },
            "eligibility_source": {
                ELIGIBILITY_ID: {
                    "source_support_passed": True,
                    "gross9_novelty_passed": True,
                    "source_state_panel_immutable": True,
                }
            },
        },
        "construction": {
            "operator": "HVTCCR-gated strict unweighted action-side majority vote",
            "decision_join": "exact equality among action decision_time and HVTCCR state-panel decision_time",
            "timestamp_tolerance": "none",
            "eligibility_rule": "eligible only when HVTCCR source_valid is true and concentration_rank >= 0.80 at that same decision",
            "active_action": "an action contributes exactly one vote only when its frozen clock is active at that exact decision",
            "active_side_domain": [-1, 1],
            "quorum": "at least two active actions",
            "vote_rule": "strict nonzero majority of the active +1/-1 action sides",
            "side": "the strict majority side",
            "tie": "cash",
            "no_quorum": "cash",
            "entry": "decision_time + 5 minutes",
            "hold": "8 elapsed hours",
            "nonoverlap": "natural deterministic half-open holds; consecutive eligible decisions abut and never overlap",
            "weights": "none; every active action contributes exactly one vote",
            "priority": "none",
            "alternatives": "none",
            "additional_or_tuned_thresholds": "none",
            "action_formula_threshold_clock_mutability": "immutable",
            "eligibility_formula_threshold_decision_mutability": "immutable",
        },
        "eligibility_conditions": {
            ELIGIBILITY_ID: eligibility_condition,
            "same_frozen_source_state_panel_as_HVSOF_8": True,
            "same_decisions": True,
            "no_side_or_action": True,
        },
        "clock": {
            "action_decisions": "exact 00:00, 08:00 and 16:00 UTC",
            "entry": "exact decision D+5m entry",
            "hold": "8 elapsed hours",
            "action_already_supplies_high_volatility_gate": True,
            "eligibility_adds_or_changes_high_volatility_gate": False,
            "funding": "not a signal input; exact held settlements only after source and Gross9 pass",
        },
        "stages": prior["stages"],
        "source_support_gates": prior["source_support_gates"],
        "gross9_novelty_gates": prior["gross9_novelty_gates"],
        "economic_gates": economic_gates,
        "research_boundary": {
            "all_action_component_outcomes_known": True,
            "all_eligibility_component_outcomes_known": True,
            "all_component_outcomes_known": True,
            "all_prior_outcomes_known": True,
            "all_prior_family_outcomes_known": True,
            "prior_HVSOF_outcomes_known": True,
            "prior_HVCSPR_outcomes_known": True,
            "prior_HVMCPAC_outcomes_known": True,
            "prior_HVMDPAC_outcomes_known": True,
            "prior_CARSC_outcomes_known": True,
            "prior_HVCMMI_outcomes_known": True,
            "prior_HVIABR_outcomes_known": True,
            "prior_HVTFR_outcomes_known": True,
            "prior_HVSVF_outcomes_known": True,
            "known_outcomes_used_to_change_frozen_components": False,
            "consensus_incidence_opened": False,
            "consensus_postentry_returns_or_pnl_opened": False,
            "classification": "exploratory discovery; not fresh confirmatory evidence",
        },
        "stopping_rule": "Evaluate the single frozen candidate through source support, Gross9 novelty, and train/test/eval/final economics; stop on first failure with no substitution, threshold, weight, priority, alternative, action, eligibility, direction, entry, side, hold, subset, or control repair.",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: Mapping[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVCAV-8 preregistration drift")
    if tuple(value.get("candidate_family", ())) != CANDIDATE_FAMILY:
        raise RuntimeError("HVCAV-8 candidate family drift")
    if value.get("action_artifacts") != ACTION_ARTIFACTS:
        raise RuntimeError("HVCAV-8 action artifact bindings drift")
    if value.get("eligibility_artifacts") != ELIGIBILITY_ARTIFACTS:
        raise RuntimeError("HVCAV-8 eligibility artifact bindings drift")
    if value.get("economic_gates", {}).get("weekly_signflip_one_sided_p_max") != 0.10:
        raise RuntimeError("HVCAV-8 weekly sign-flip gate drift")
    for artifacts in (*ACTION_ARTIFACTS.values(), *ELIGIBILITY_ARTIFACTS.values()):
        for artifact_type, artifact in artifacts.items():
            if sha256_file(artifact["path"]) != artifact["sha256"]:
                raise RuntimeError(f"HVCAV-8 {artifact_type} artifact drift")


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
