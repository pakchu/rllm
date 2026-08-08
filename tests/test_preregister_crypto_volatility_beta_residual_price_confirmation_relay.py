from training import preregister_crypto_volatility_beta_residual_price_confirmation_relay as prereg
def test_cvbrpcr_is_outcome_blind_singleton():
 r=prereg.build();assert r["policy_id"]=="CVBRPCR-12";assert r["outcomes_opened"] is False;assert r["source_incidence_opened"] is False;assert r["singleton"] is True;assert r["research_boundary"]["candidate_count"]==1;assert r["research_boundary"]["grid"] is False;assert r["research_boundary"]["repair_of_prior_candidate"] is False
def test_cvbrpcr_freezes_residual_confirmation_and_gates():
 r=prereg.build();p=r["policy"];assert p["prior_days"]==90;assert p["prior_min_days"]==60;assert p["absolute_standardized_residual_min"]==1.;assert p["hold_hours"]==12;assert r["economic_gates"]["cagr_to_strict_mdd_min"]==3.;assert r["economic_gates"]["stress_cagr_to_strict_mdd_min"]==2.5
def test_cvbrpcr_hash_binds_core():
 r=prereg.build();assert r["manifest_hash"]==prereg.canonical_hash({k:v for k,v in r.items() if k!="manifest_hash"})
