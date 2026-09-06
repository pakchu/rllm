from __future__ import annotations
import copy
import pytest
from training import preregister_high_volatility_causal_vixfix_reentry_reversal as p
from training import preregister_high_volatility_cross_structure_action_vote as hvcav

def test_singleton_mechanism_and_fixed_policy() -> None:
    value=p.build();p.validate(value)
    assert value["policy_id"]=="HVCVFR-24"
    assert value["candidate_family"]==["HVCVFR-24"]
    assert value["candidate_family_size"]==1
    assert value["policy"]=={"history_decisions":270,"minimum_history_decisions":180,"prior_close_periods":22,"current_auction_minutes":1440,"variation_minutes":1440,"displacement_tail_min":.80,"variation_rank_min":.60,"entry_delay_minutes":5,"hold_hours":24,"leverage":.5,"base_cost_per_notional_side":.0006,"stress_cost_per_notional_side":.0010}

def test_exact_hvcav_stages_and_gates() -> None:
    value=p.build(); prior=hvcav.build()
    for key in ("stages","source_support_gates","gross9_novelty_gates","economic_gates"):
        assert value[key]==prior[key]

def test_causal_boundary_and_no_price_direction_condition() -> None:
    value=p.build()
    assert value["features"]["displacement_rank"].startswith("strict-prior")
    assert value["features"]["variation_rank"].startswith("strict-prior")
    assert "no OI" in value["features"]["eligible"]
    assert value["features"]["tail_reentry"].startswith("immediately previous")
    assert value["mechanism"]["side"].startswith("long after")
    assert value["source_incidence_opened"] is False
    assert value["outcomes_opened"] is False
    assert value["gross9_rows_opened"] is False

def test_prior_failure_disclosed_without_repair() -> None:
    b=p.build()["research_boundary"]
    assert b["prior_HVFADR_source_incidence_and_failure_known"] is True
    assert b["prior_HVFADR_outcomes_opened"] is False
    assert b["same_mechanism_as_HVFADR"] is False
    assert b["prior_HVCFDR_source_incidence_and_failure_known"] is True
    assert b["prior_HVCFDR_outcomes_opened"] is False
    assert b["same_mechanism_as_HVCFDR"] is False
    assert b["prior_HVCFIR_source_incidence_and_failure_known"] is True
    assert b["prior_HVCFIR_outcomes_opened"] is False
    assert b["same_mechanism_as_HVCFIR"] is False
    assert b["prior_HVCFZR_source_incidence_and_failure_known"] is True
    assert b["prior_HVCFZR_outcomes_opened"] is False
    assert b["same_mechanism_as_HVCFZR"] is False
    assert b["prior_HVCFLR_source_incidence_and_failure_known"] is True
    assert b["prior_HVCFLR_outcomes_opened"] is False
    assert b["same_mechanism_as_HVCFLR"] is False
    assert b["prior_HVCFMR_source_novelty_and_train_failure_known"] is True
    assert b["same_mechanism_as_HVCFMR"] is False
    assert b["prior_HVCFBR_source_incidence_and_failure_known"] is True
    assert b["prior_HVCFBR_outcomes_opened"] is False
    assert b["same_mechanism_as_HVCFBR"] is False
    assert b["prior_HVCFRR_source_incidence_and_failure_known"] is True
    assert b["prior_HVCFRR_outcomes_opened"] is False
    assert b["same_mechanism_as_HVCFRR"] is False
    assert b["prior_HVCORR_source_novelty_and_train_failure_known"] is True
    assert b["same_mechanism_as_HVCORR"] is False
    assert b["prior_HVDPR_source_incidence_known"] is True
    assert b["prior_HVDPR_outcomes_opened"] is False
    assert b["same_mechanism_as_HVDPR"] is False
    assert b["prior_HVCOPR_source_novelty_and_train_failure_known"] is True
    assert b["same_mechanism_as_HVCOPR"] is False
    assert b["prior_HVDUDAR_outcomes_known"] is True
    assert b["same_mechanism_as_HVDUDAR"] is False
    assert b["prior_HVCEDR_source_and_novelty_failure_known"] is True
    assert b["prior_HVCEDR_outcomes_opened"] is False
    assert b["same_mechanism_as_HVCEDR"] is False
    assert b["repair_of_prior_candidate"] is False
    assert b["prior_outcomes_used_to_set_formula_rank_side_hold_or_clock"] is False

def test_manifest_drift_rejected() -> None:
    value=copy.deepcopy(p.build());value["policy"]["variation_rank_min"]=.55
    with pytest.raises(RuntimeError,match="preregistration drift"):p.validate(value)
