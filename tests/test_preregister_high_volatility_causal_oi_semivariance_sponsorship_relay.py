from __future__ import annotations
import copy
import pytest
from training import preregister_high_volatility_causal_oi_semivariance_sponsorship_relay as p
from training import preregister_high_volatility_cross_structure_action_vote as hvcav

def test_singleton_mechanism_and_fixed_policy() -> None:
    value=p.build();p.validate(value)
    assert value["policy_id"]=="HVCOSR-24"
    assert value["candidate_family"]==["HVCOSR-24"]
    assert value["candidate_family_size"]==1
    assert value["policy"]=={"history_cycles":270,"minimum_history_cycles":180,"day_minutes":1440,"absolute_imbalance_rank_min":.60,"variation_rank_min":.60,"entry_delay_minutes":5,"hold_hours":24,"leverage":.5,"base_cost_per_notional_side":.0006,"stress_cost_per_notional_side":.0010}

def test_exact_hvcav_stages_and_gates() -> None:
    value=p.build(); prior=hvcav.build()
    for key in ("stages","source_support_gates","gross9_novelty_gates","economic_gates"):
        assert value[key]==prior[key]

def test_causal_semivariance_boundary() -> None:
    value=p.build()
    assert value["features"]["oi_expansion"].startswith("strict positive")
    assert value["features"]["absolute_imbalance_rank"].startswith("strict-prior")
    assert value["features"]["variation_rank"].startswith("strict-prior")
    assert "no funding" in value["features"]["eligible"]
    assert value["mechanism"]["side"].startswith("strict sign")
    assert value["source_incidence_opened"] is False
    assert value["outcomes_opened"] is False
    assert value["gross9_rows_opened"] is False

def test_prior_semivariance_failures_disclosed_without_repair() -> None:
    b=p.build()["research_boundary"]
    assert b["prior_HVSIR_source_and_gross9_failure_known"] is True
    assert b["same_mechanism_as_HVSIR"] is False
    assert b["prior_HVCASI_source_novelty_and_train_failure_known"] is True
    assert b["same_mechanism_as_HVCASI"] is False
    assert b["repair_of_prior_candidate"] is False
    assert b["prior_outcomes_used_to_set_formula_rank_side_hold_or_clock"] is False

def test_manifest_drift_rejected() -> None:
    value=copy.deepcopy(p.build());value["policy"]["variation_rank_min"]=.55
    with pytest.raises(RuntimeError,match="preregistration drift"):p.validate(value)
