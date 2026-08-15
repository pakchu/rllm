from __future__ import annotations

import copy

import pytest

from training import preregister_high_volatility_adverse_funding_aggressive_flow_continuation as p
from training import preregister_high_volatility_cross_structure_action_vote as hvcav


def test_singleton_and_fixed_policy() -> None:
    value = p.build()
    p.validate(value)
    assert value["policy_id"] == "HVAFGF-8"
    assert value["candidate_family"] == ["HVAFGF-8"]
    assert value["policy"]["variation_rank_min"] == 0.60
    assert value["policy"]["hold_hours"] == 8
    assert value["policy"]["leverage"] == 0.5


def test_contract_gates_are_exact() -> None:
    value = p.build()
    prior = hvcav.build()
    for key in ("stages", "source_support_gates", "gross9_novelty_gates", "economic_gates"):
        assert value[key] == prior[key]


def test_causal_adverse_funding_flow_boundary() -> None:
    value = p.build()
    assert value["features"]["variation_rank"].startswith("strict-prior")
    assert "opposite sign" in value["features"]["funding_adversity"]
    assert value["features"]["side"] == "strict sign of flow_share"
    assert value["source_incidence_opened"] is False
    assert value["outcomes_opened"] is False
    assert value["gross9_rows_opened"] is False


def test_prior_knowledge_disclosed_without_repair() -> None:
    boundary = p.build()["research_boundary"]
    assert boundary["prior_price_flow_OI_router_gross9_failure_known"] is True
    assert boundary["prior_taker_flow_decay_train_failure_known"] is True
    assert boundary["exact_adverse_funding_price_flow_conjunction_incidence_or_outcomes_known"] is False
    assert boundary["repair_of_prior_candidate"] is False
    assert boundary["prior_outcomes_used_to_set_rank_side_hold_or_clock"] is False


def test_manifest_drift_rejected() -> None:
    value = copy.deepcopy(p.build())
    value["policy"]["hold_hours"] = 6
    with pytest.raises(RuntimeError, match="preregistration drift"):
        p.validate(value)
