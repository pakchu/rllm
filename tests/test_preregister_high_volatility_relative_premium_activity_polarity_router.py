from __future__ import annotations
import copy
import pytest
from training import preregister_high_volatility_cross_structure_action_vote as hvcav
from training import preregister_high_volatility_relative_premium_activity_polarity_router as p


def test_singleton_fixed_quiescence_policy():
    v=p.build();p.validate(v);assert v["policy_id"]=="HVPRAPR-12";assert v["candidate_family"]==["HVPRAPR-12"]
    assert v["policy"]["lower_relative_activity_rank_max"]==.25;assert v["policy"]["btc_variation_rank_min"]==.65;assert v["policy"]["hold_hours"]==12


def test_contract_gates_are_exact():
    v=p.build();prior=hvcav.build()
    for key in ("stages","source_support_gates","gross9_novelty_gates","economic_gates"):assert v[key]==prior[key]


def test_causal_boundary_and_prior_disclosure():
    v=p.build();assert v["features"]["causal_ranks"].startswith("strict-prior");assert "lower tail" in v["features"]["side"]
    assert v["source_incidence_opened"] is False and v["outcomes_opened"] is False
    b=v["research_boundary"];assert b["prior_high_premium_activity_HVPASR_train_failure_known"] is True;assert b["two_sided_relative_router_control_existed"] is False;assert b["repair_of_prior_candidate"] is False


def test_manifest_drift_rejected():
    v=copy.deepcopy(p.build());v["policy"]["upper_relative_activity_rank_min"]=.70
    with pytest.raises(RuntimeError,match="preregistration drift"):p.validate(v)
