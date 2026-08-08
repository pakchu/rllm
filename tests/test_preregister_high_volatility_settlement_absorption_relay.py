from training import preregister_high_volatility_settlement_absorption_relay as prereg
def test_hvsar_is_outcome_blind_independent_singleton():
 r=prereg.build();assert r["policy_id"]=="HVSAR-6";assert r["outcomes_opened"] is False;assert r["source_incidence_opened"] is False;assert r["singleton"] is True;assert r["research_boundary"]["candidate_count"]==1;assert r["research_boundary"]["grid"] is False;assert r["research_boundary"]["repair_of_prior_candidate"] is False
def test_hvsar_freezes_absorption_path_and_gates():
 r=prereg.build();p=r["policy"];assert p["history_observations"]==270;assert p["minimum_history_observations"]==180;assert p["variation_rank_min"]==.65;assert p["late_to_early_absolute_return_max"]==.5;assert p["late_quote_volume_share_min"]==.3;assert p["hold_hours"]==6;assert r["economic_gates"]["cagr_to_strict_mdd_min"]==3.
def test_hvsar_hash_binds_core():
 r=prereg.build();assert r["manifest_hash"]==prereg.canonical_hash({k:v for k,v in r.items() if k!="manifest_hash"})
