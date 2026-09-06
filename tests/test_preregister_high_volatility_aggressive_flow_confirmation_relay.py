from training import preregister_high_volatility_aggressive_flow_confirmation_relay as prereg
def test_hvafc_is_outcome_blind_independent_singleton():
 r=prereg.build();assert r["policy_id"]=="HVAFC-6";assert r["outcomes_opened"] is False;assert r["source_incidence_opened"] is False;assert r["singleton"] is True;assert r["research_boundary"]["candidate_count"]==1;assert r["research_boundary"]["grid"] is False;assert r["research_boundary"]["repair_of_prior_candidate"] is False
def test_hvafc_freezes_flow_confirmation_and_gates():
 r=prereg.build();p=r["policy"];assert p["history_observations"]==270;assert p["minimum_history_observations"]==180;assert p["variation_rank_min"]==.65;assert p["absolute_late_taker_imbalance_min"]==.1;assert p["hold_hours"]==6;assert r["source_plan"]["btc_1m"]["columns"][-1]=="taker_buy_quote";assert r["economic_gates"]["cagr_to_strict_mdd_min"]==3.
def test_hvafc_hash_binds_core():
 r=prereg.build();assert r["manifest_hash"]==prereg.canonical_hash({k:v for k,v in r.items() if k!="manifest_hash"})
