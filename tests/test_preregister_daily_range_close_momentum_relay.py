from training import preregister_daily_range_close_momentum_relay as prereg

def test_drcmr_is_outcome_blind_independent_singleton():
 r=prereg.build();assert r["policy_id"]=="DRCMR-12";assert r["outcomes_opened"] is False;assert r["source_incidence_opened"] is False;assert r["singleton"] is True;assert r["research_boundary"]["candidate_count"]==1;assert r["research_boundary"]["grid"] is False;assert r["research_boundary"]["repair_of_prior_candidate"] is False

def test_drcmr_freezes_daily_auction_and_strict_gates():
 r=prereg.build();p=r["policy"];assert p["prior_days"]==90;assert p["prior_min_days"]==60;assert p["daily_range_rank_min"]==.65;assert p["upper_close_location_min"]==.8;assert p["lower_close_location_max"]==.2;assert p["hold_hours"]==12;assert r["economic_gates"]["cagr_to_strict_mdd_min"]==3.;assert r["economic_gates"]["stress_cagr_to_strict_mdd_min"]==2.5

def test_drcmr_hash_binds_core():
 r=prereg.build();assert r["manifest_hash"]==prereg.canonical_hash({k:v for k,v in r.items() if k!="manifest_hash"})
