from __future__ import annotations

import copy

import pytest

from training import preregister_high_volatility_cross_structure_action_vote as hvcav
from training import preregister_high_volatility_oi_large_ticket_joint_sponsorship as p


def test_singleton_and_frozen_component_thresholds() -> None:
    value = p.build()
    p.validate(value)
    assert value["policy_id"] == "HVOILTJ-8"
    assert value["candidate_family"] == ["HVOILTJ-8"]
    assert value["policy"]["coactivity_rank_min"] == 0.75
    assert value["policy"]["gross_oi_activity_rank_min"] == 0.60
    assert value["policy"]["clustering_rank_min"] == 0.80
    assert value["policy"]["oi_variation_rank_min"] == 0.65
    assert value["policy"]["ticket_variation_rank_min"] == 0.65


def test_contract_gates_are_exact() -> None:
    value = p.build()
    prior = hvcav.build()
    for key in ("stages", "source_support_gates", "gross9_novelty_gates", "economic_gates"):
        assert value[key] == prior[key]


def test_causal_joint_onset_boundary() -> None:
    value = p.build()
    assert value["features"]["joint_onset"].startswith("joint eligible now")
    assert value["features"]["side"] == "strict sign of completed three-hour return"
    assert value["source_incidence_opened"] is False
    assert value["outcomes_opened"] is False
    assert value["gross9_rows_opened"] is False


def test_prior_component_knowledge_disclosed_without_tuning() -> None:
    boundary = p.build()["research_boundary"]
    assert boundary["prior_HVOIPCSR_source_novelty_and_train_failure_known"] is True
    assert boundary["prior_HVLTTC_source_novelty_and_train_failure_known"] is True
    assert boundary["prior_component_event_rows_reused"] is False
    assert boundary["component_thresholds_copied_without_search"] is True
    assert boundary["prior_outcomes_used_to_set_component_thresholds_side_hold_or_clock"] is False
    assert boundary["repair_of_prior_candidate"] is False


def test_manifest_drift_rejected() -> None:
    value = copy.deepcopy(p.build())
    value["policy"]["clustering_rank_min"] = 0.75
    with pytest.raises(RuntimeError, match="preregistration drift"):
        p.validate(value)
