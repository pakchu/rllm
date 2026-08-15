from __future__ import annotations
import copy, pytest
from training import preregister_high_volatility_causal_ticket_impact_inventory_relay as p
from training import preregister_high_volatility_cross_structure_action_vote as hvcav

def test_singleton_and_fixed_policy() -> None:
    x=p.build();p.validate(x)
    assert x["policy_id"]=="HVCTIIR-8" and x["candidate_family"]==["HVCTIIR-8"]
    assert x["policy"]=={"history_cycles":270,"minimum_history_cycles":180,"path_minutes":480,"ticket_impact_rank_min":.60,"variation_rank_min":.60,"entry_delay_minutes":5,"hold_hours":8,"leverage":.5,"base_cost_per_notional_side":.0006,"stress_cost_per_notional_side":.001}

def test_contract_and_causality() -> None:
    x=p.build(); prior=hvcav.build()
    for k in ("stages","source_support_gates","gross9_novelty_gates","economic_gates"): assert x[k]==prior[k]
    f=x["features"]; assert "log1p" in f["ticket_impact_alignment"] and "all 480" in f["ticket_impact_alignment"]
    assert f["ticket_impact_rank"].startswith("strict-prior") and f["variation_rank"].startswith("strict-prior")
    assert x["source_incidence_opened"] is False and x["outcomes_opened"] is False

def test_boundary_discloses_prior_without_repair() -> None:
    b=p.build()["research_boundary"]
    assert b["prior_HVCATAIR_source_novelty_and_train_failure_known"] is True
    assert b["same_mechanism_as_HVCATAIR"] is False and b["repair_of_prior_candidate"] is False
    assert b["prior_outcomes_used_to_set_formula_rank_side_hold_or_clock"] is False

def test_drift_rejected() -> None:
    x=copy.deepcopy(p.build());x["policy"]["ticket_impact_rank_min"]=.55
    with pytest.raises(RuntimeError,match="preregistration drift"):p.validate(x)
