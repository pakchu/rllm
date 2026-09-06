from training import preregister_asian_carry_unwind_concordance_relay as prereg
def test_acucr_is_outcome_blind_independent_singleton():
 r=prereg.build();assert r["policy_id"]=="ACUCR-12";assert r["outcomes_opened"] is False;assert r["source_incidence_opened"] is False;assert r["gross9_rows_opened"] is False;assert r["singleton"] is True;assert r["research_boundary"]["candidate_count"]==1;assert r["research_boundary"]["repair_of_prior_candidate"] is False
def test_acucr_freezes_carry_concordance_and_volatile_regime():
 r=prereg.build();assert r["source_plan"]["fx"]["symbols"]==["USDAUD","USDJPY"];assert r["policy"]["variation_rank_min"]==.65;assert r["policy"]["hold_hours"]==12;assert r["clock"]["entry"]=="exact BTCUSDT D 08:05 UTC open"
def test_acucr_is_not_shrr_or_usdjpy_control():
 r=prereg.build();assert "SHRR" in r["mechanism"]["why_distinct"];assert r["research_boundary"]["prior_event_sets_reused"] is False;assert r["research_boundary"]["promoted_prior_control"] is False
def test_acucr_hash_binds_core():
 r=prereg.build();assert r["manifest_hash"]==prereg.canonical_hash({k:v for k,v in r.items() if k!="manifest_hash"})
