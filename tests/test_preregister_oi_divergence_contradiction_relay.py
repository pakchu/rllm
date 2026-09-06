from training import preregister_oi_divergence_contradiction_relay as prereg
def test_oidcr_is_symmetric_outcome_blind_singleton():
 p=prereg.build();prereg.validate(p);assert p["policy_id"]=="OIDCR-8" and p["outcomes_opened"] is False and p["source_incidence_opened"] is False and p["singleton"] is True;assert p["research_boundary"]["symmetric_candidate_outcomes_known"] is False and p["research_boundary"]["grid"] is False
