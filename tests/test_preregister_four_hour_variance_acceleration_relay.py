from training import preregister_four_hour_variance_acceleration_relay as prereg
def test_fhvar_is_outcome_blind_independent_singleton():
 r=prereg.build();assert r["policy_id"]=="FHVAR-2";assert r["outcomes_opened"] is False;assert r["source_incidence_opened"] is False;assert r["singleton"] is True;assert r["research_boundary"]["candidate_count"]==1;assert r["research_boundary"]["grid"] is False;assert r["research_boundary"]["repair_of_prior_candidate"] is False
def test_fhvar_freezes_variance_acceleration_and_gates():
 r=prereg.build();p=r["policy"];assert p["history_observations"]==180;assert p["minimum_history_observations"]==120;assert p["variation_rank_min"]==.65;assert p["second_to_first_variation_min"]==1.5;assert p["hold_hours"]==2;assert r["source_plan"]["btc_1m"]["interval"]=="1m";assert r["economic_gates"]["cagr_to_strict_mdd_min"]==3.;assert r["economic_gates"]["stress_cagr_to_strict_mdd_min"]==2.5
def test_fhvar_hash_binds_core():
 r=prereg.build();assert r["manifest_hash"]==prereg.canonical_hash({k:v for k,v in r.items() if k!="manifest_hash"})
