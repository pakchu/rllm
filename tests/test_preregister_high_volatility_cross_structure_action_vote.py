from __future__ import annotations

import copy

import pytest

from training import preregister_high_volatility_cross_structure_action_vote as p
from training import preregister_high_volatility_cross_structure_priority_router as hvcspr
from training import preregister_high_volatility_state_ordered_filter as hvsof


def test_manifest_fixes_one_candidate_and_exact_components() -> None:
    value = p.build()
    p.validate(value)
    assert value["policy_id"] == "HVCAV-8"
    assert value["action_order"] == [
        "CARSC-8",
        "HVCMMI-8",
        "HVIABR-8",
        "HVTFR-8",
        "HVSVF-8",
    ]
    assert value["eligibility_id"] == "HVTCCR-8"
    assert value["candidate_family"] == ["HVCAV-8"]
    assert value["candidate_family_size"] == 1
    assert value["single_candidate_only"] is True
    assert "train_only_selection" not in value
    assert "familywise_multiplicity" not in value
    assert "multiplicity" not in value


def test_vote_requires_quorum_and_strict_nonzero_majority() -> None:
    assert p.action_vote_side([]) == 0
    assert p.action_vote_side([1]) == 0
    assert p.action_vote_side([-1]) == 0
    assert p.action_vote_side([1, -1]) == 0
    assert p.action_vote_side([1, 1]) == 1
    assert p.action_vote_side([-1, -1]) == -1
    assert p.action_vote_side([1, 1, -1]) == 1
    assert p.action_vote_side([-1, -1, 1]) == -1
    assert p.action_vote_side([1, 1, -1, -1]) == 0
    with pytest.raises(ValueError, match=r"exact \+1 or -1"):
        p.action_vote_side([1, 0])
    with pytest.raises(ValueError, match=r"exact \+1 or -1"):
        p.action_vote_side([True, 1])


def test_frozen_eligibility_clock_and_natural_nonoverlap() -> None:
    value = p.build()
    assert value["eligibility_conditions"]["HVTCCR-8"] == hvsof.build()[
        "eligibility_conditions"
    ]["HVTCCR-8"]
    construction = value["construction"]
    assert construction["quorum"] == "at least two active actions"
    assert construction["vote_rule"] == (
        "strict nonzero majority of the active +1/-1 action sides"
    )
    assert construction["tie"] == "cash"
    assert construction["no_quorum"] == "cash"
    assert construction["weights"].startswith("none")
    assert construction["priority"] == "none"
    assert construction["alternatives"] == "none"
    assert construction["additional_or_tuned_thresholds"] == "none"
    assert "abut and never overlap" in construction["nonoverlap"]
    assert value["clock"] == {
        "action_decisions": "exact 00:00, 08:00 and 16:00 UTC",
        "entry": "exact decision D+5m entry",
        "hold": "8 elapsed hours",
        "action_already_supplies_high_volatility_gate": True,
        "eligibility_adds_or_changes_high_volatility_gate": False,
        "funding": "not a signal input; exact held settlements only after source and Gross9 pass",
    }


def test_stages_and_gates_are_unchanged_with_no_extra_multiplicity() -> None:
    value = p.build()
    prior = hvsof.build()
    assert value["stages"] == prior["stages"]
    assert value["source_support_gates"] == prior["source_support_gates"]
    assert value["gross9_novelty_gates"] == prior["gross9_novelty_gates"]
    assert value["economic_gates"] == prior["economic_gates"]
    assert value["economic_gates"]["weekly_signflip_one_sided_p_max"] == 0.10


def test_all_requested_artifacts_are_hash_pinned() -> None:
    value = p.build()
    assert set(value["action_artifacts"]) == set(p.ACTION_ORDER)
    assert set(value["eligibility_artifacts"]) == {"HVTCCR-8"}
    for action in ("CARSC-8", "HVCMMI-8", "HVIABR-8"):
        assert value["action_artifacts"][action] == hvcspr.ACTION_ARTIFACTS[action]
    for action in ("HVTFR-8", "HVSVF-8"):
        assert value["action_artifacts"][action] == hvsof.ACTION_ARTIFACTS[action]
    assert value["eligibility_artifacts"]["HVTCCR-8"] == hvsof.ELIGIBILITY_ARTIFACTS[
        "HVTCCR-8"
    ]
    assert all(
        set(artifacts) == {"preregistration", "support", "gross9", "clock"}
        for artifacts in value["action_artifacts"].values()
    )
    assert set(value["eligibility_artifacts"]["HVTCCR-8"]) == {
        "preregistration",
        "support",
        "gross9",
        "state_panel",
        "source_manifest",
    }
    for artifacts in (
        *value["action_artifacts"].values(),
        *value["eligibility_artifacts"].values(),
    ):
        for artifact in artifacts.values():
            assert set(artifact) == {"path", "sha256"}
            assert len(artifact["sha256"]) == 64
            assert p.sha256_file(artifact["path"]) == artifact["sha256"]


def test_known_outcomes_disclosed_but_consensus_remains_unopened() -> None:
    value = p.build()
    boundary = value["research_boundary"]
    assert boundary["all_component_outcomes_known"] is True
    assert boundary["all_prior_outcomes_known"] is True
    assert boundary["all_prior_family_outcomes_known"] is True
    assert boundary["consensus_incidence_opened"] is False
    assert boundary["consensus_postentry_returns_or_pnl_opened"] is False
    assert value["consensus_incidence_opened"] is False
    assert value["consensus_outcomes_opened"] is False
    assert value["exploratory_discovery"] is True
    assert value["fresh_confirmatory_evidence"] is False


def test_manifest_hash_detects_any_preregistered_drift() -> None:
    drifted = copy.deepcopy(p.build())
    drifted["construction"]["quorum"] = "at least three active actions"
    with pytest.raises(RuntimeError, match="preregistration drift"):
        p.validate(drifted)
