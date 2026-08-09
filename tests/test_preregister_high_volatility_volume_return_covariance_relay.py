from training import preregister_high_volatility_volume_return_covariance_relay as prereg
def test_hvvrcr_is_outcome_blind_independent_singleton():
 r=prereg.build();assert r["policy_id"]=="HVVRCR-12";assert r["outcomes_opened"] is False;assert r["source_incidence_opened"] is False;assert r["gross9_rows_opened"] is False;assert r["singleton"] is True;assert r["research_boundary"]["candidate_count"]==1;assert r["research_boundary"]["repair_of_prior_candidate"] is False
def test_hvvrcr_freezes_covariance_volatile_regime_and_clock():
 r=prereg.build();assert r["policy"]["bars"]==288;assert r["policy"]["absolute_correlation_rank_min"]==.70;assert r["policy"]["variation_rank_min"]==.65;assert r["policy"]["hold_hours"]==12;assert r["clock"]["entry"]=="exact BTCUSDT 02:05 UTC open"
def test_hvvrcr_is_not_vspcr_control_promotion():
 r=prereg.build();assert "VSPCR" in r["mechanism"]["why_distinct"];assert r["research_boundary"]["prior_event_sets_reused"] is False;assert r["research_boundary"]["promoted_prior_control"] is False
def test_hvvrcr_hash_binds_core():
 r=prereg.build();assert r["manifest_hash"]==prereg.canonical_hash({k:v for k,v in r.items() if k!="manifest_hash"})
