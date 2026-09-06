from __future__ import annotations

import copy

import pytest

from training import preregister_high_volatility_cross_structure_action_vote as hvcav
from training import preregister_high_volatility_large_ticket_funding_innovation_router as p


def test_singleton_fixed_router_policy() -> None:
    value = p.build(); p.validate(value)
    assert value["policy_id"] == "HVLTFIR-8"
    assert value["candidate_family"] == ["HVLTFIR-8"]
    assert value["policy"]["clustering_rank_min"] == 0.80
    assert value["policy"]["variation_rank_min"] == 0.65
    assert value["policy"]["funding_gap_min_minutes"] == 479
    assert value["policy"]["funding_gap_max_minutes"] == 481


def test_contract_gates_are_exact() -> None:
    value = p.build(); prior = hvcav.build()
    for key in ("stages", "source_support_gates", "gross9_novelty_gates", "economic_gates"):
        assert value[key] == prior[key]


def test_causal_funding_route_and_boundary() -> None:
    value = p.build()
    assert value["features"]["side"] == "negative strict sign of funding innovation"
    assert "S1<=D" in value["features"]["latest_funding"]
    assert "no timestamp snapping" in value["features"]["funding_freshness"]
    assert value["source_incidence_opened"] is False
    assert value["outcomes_opened"] is False


def test_prior_knowledge_without_repair() -> None:
    boundary = p.build()["research_boundary"]
    assert boundary["prior_HVLTTC_source_novelty_and_train_failure_known"] is True
    assert boundary["exact_large_ticket_funding_innovation_routing_incidence_or_outcomes_known"] is False
    assert boundary["prior_outcomes_used_to_set_funding_gap_side_hold_or_clock"] is False
    assert boundary["repair_of_prior_candidate"] is False
    assert boundary["promoted_prior_control"] is False


def test_manifest_drift_rejected() -> None:
    value = copy.deepcopy(p.build()); value["policy"]["funding_gap_max_minutes"] = 482
    with pytest.raises(RuntimeError, match="preregistration drift"):
        p.validate(value)
