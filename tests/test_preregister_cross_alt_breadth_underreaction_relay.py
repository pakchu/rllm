from training import preregister_cross_alt_breadth_underreaction_relay as prereg

def test_cabur_is_outcome_blind_independent_singleton():
 r=prereg.build();assert r["policy_id"]=="CABUR-8";assert r["outcomes_opened"] is False;assert r["source_incidence_opened"] is False;assert r["gross9_rows_opened"] is False;assert r["singleton"] is True;assert r["research_boundary"]["candidate_count"]==1;assert r["research_boundary"]["repair_of_prior_candidate"] is False

def test_cabur_freezes_breadth_underreaction_and_volatile_regime():
 r=prereg.build();assert len(r["source_plan"]["symbols"])==7;assert r["policy"]["breadth_min"]==4;assert r["policy"]["underreaction_ratio_max"]==1.;assert r["policy"]["alt_impulse_rank_min"]==.70;assert r["policy"]["variation_rank_min"]==.65;assert r["policy"]["hold_hours"]==8

def test_cabur_is_not_cablr_control_promotion():
 r=prereg.build();assert "CABLR" in r["mechanism"]["why_distinct"];assert r["research_boundary"]["prior_event_sets_reused"] is False;assert r["research_boundary"]["promoted_prior_control"] is False

def test_cabur_hash_binds_core():
 r=prereg.build();assert r["manifest_hash"]==prereg.canonical_hash({k:v for k,v in r.items() if k!="manifest_hash"})
