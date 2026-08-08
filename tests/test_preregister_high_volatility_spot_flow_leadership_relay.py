from training import preregister_high_volatility_spot_flow_leadership_relay as prereg
def test_hvsflr_is_outcome_blind_independent_singleton():
 r=prereg.build();assert r["policy_id"]=="HVSFLR-6";assert r["outcomes_opened"] is False;assert r["source_incidence_opened"] is False;assert r["singleton"] is True;assert r["research_boundary"]["candidate_count"]==1;assert r["research_boundary"]["grid"] is False;assert r["research_boundary"]["repair_of_prior_candidate"] is False
def test_hvsflr_freezes_spot_leadership_and_gates():
 r=prereg.build();p=r["policy"];assert p["history_observations"]==270;assert p["minimum_history_observations"]==180;assert p["variation_rank_min"]==.65;assert p["absolute_spot_taker_imbalance_min"]==.1;assert p["spot_to_perp_imbalance_ratio_min"]==1.5;assert p["hold_hours"]==6;assert r["source_plan"]["spot_1m"]["table"]=="bars_binance_spot";assert r["economic_gates"]["cagr_to_strict_mdd_min"]==3.
def test_hvsflr_hash_binds_core():
 r=prereg.build();assert r["manifest_hash"]==prereg.canonical_hash({k:v for k,v in r.items() if k!="manifest_hash"})
