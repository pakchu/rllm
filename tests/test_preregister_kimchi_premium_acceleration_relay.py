from training import preregister_kimchi_premium_acceleration_relay as prereg

def test_kpar_is_outcome_blind_singleton():
 r=prereg.build();assert r["policy_id"]=="KPAR-12" and r["outcomes_opened"] is False and r["source_incidence_opened"] is False and r["singleton"] is True;assert r["research_boundary"]["candidate_count"]==1 and r["research_boundary"]["grid"] is False and r["research_boundary"]["repair_of_prior_candidate"] is False
def test_kpar_freezes_acceleration_and_gates():
 r=prereg.build();p=r["policy"];assert p["premium_horizon_hours"]==6 and p["prior_sessions"]==90 and p["prior_min_sessions"]==60 and p["absolute_premium_change_z_min"]==1. and p["realized_variation_rank_min"]==.65 and p["hold_hours"]==12;assert r["economic_gates"]["cagr_to_strict_mdd_min"]==3. and r["economic_gates"]["stress_cagr_to_strict_mdd_min"]==2.5
def test_kpar_hash_binds_core():
 r=prereg.build();assert r["manifest_hash"]==prereg.canonical_hash({k:v for k,v in r.items() if k!="manifest_hash"})
