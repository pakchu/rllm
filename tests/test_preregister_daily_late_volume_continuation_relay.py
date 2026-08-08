from training import preregister_daily_late_volume_continuation_relay as prereg
def test_dlvcr_is_outcome_blind_independent_singleton():
 r=prereg.build();assert r["policy_id"]=="DLVCR-12";assert r["outcomes_opened"] is False;assert r["source_incidence_opened"] is False;assert r["singleton"] is True;assert r["research_boundary"]["candidate_count"]==1;assert r["research_boundary"]["grid"] is False;assert r["research_boundary"]["repair_of_prior_candidate"] is False
def test_dlvcr_freezes_late_volume_direction_and_gates():
 r=prereg.build();p=r["policy"];assert p["prior_days"]==90;assert p["prior_min_days"]==60;assert p["variation_rank_min"]==.65;assert p["late_quote_volume_share_min"]==.35;assert p["late_absolute_move_share_min"]==.5;assert p["hold_hours"]==12;assert r["source_plan"]["btc_1m"]["interval"]=="1m";assert r["economic_gates"]["cagr_to_strict_mdd_min"]==3.
def test_dlvcr_hash_binds_core():
 r=prereg.build();assert r["manifest_hash"]==prereg.canonical_hash({k:v for k,v in r.items() if k!="manifest_hash"})
