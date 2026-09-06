from training import preregister_cboe_convexity_beta_residual_relay as prereg
def test_ccbrr_is_outcome_blind_singleton():
 r=prereg.build();assert r["policy_id"]=="CCBRR-12";assert r["outcomes_opened"] is False;assert r["source_incidence_opened"] is False;assert r["singleton"] is True;assert r["research_boundary"]["candidate_count"]==1;assert r["research_boundary"]["grid"] is False;assert r["research_boundary"]["repair_of_prior_candidate"] is False
def test_ccbrr_freezes_causal_residual_and_gates():
 r=prereg.build();p=r["policy"];assert p["prior_observations"]==252;assert p["prior_min_observations"]==126;assert p["absolute_standardized_residual_min"]==1.;assert p["hold_hours"]==12;assert r["economic_gates"]["cagr_to_strict_mdd_min"]==3.;assert r["economic_gates"]["stress_cagr_to_strict_mdd_min"]==2.5
def test_ccbrr_hash_binds_core():
 r=prereg.build();assert r["manifest_hash"]==prereg.canonical_hash({k:v for k,v in r.items() if k!="manifest_hash"})
