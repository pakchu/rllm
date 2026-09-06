from training import preregister_cross_market_multi_horizon_trend_concordance_relay as prereg
def test_cmmtcr_is_outcome_blind_independent_singleton():
 r=prereg.build();assert r["policy_id"]=="CMMTCR-24";assert r["outcomes_opened"] is False;assert r["source_incidence_opened"] is False;assert r["gross9_rows_opened"] is False;assert r["singleton"] is True;assert r["research_boundary"]["candidate_count"]==1;assert r["research_boundary"]["repair_of_prior_candidate"] is False
def test_cmmtcr_freezes_cross_market_trend_and_volatile_regime():
 r=prereg.build();assert r["source_plan"]["symbols"]==["BTCUSDT","ETHUSDT"];assert r["policy"]["variation_rank_min"]==.65;assert r["policy"]["hold_hours"]==24;assert r["clock"]["entry"]=="exact BTCUSDT 02:05 UTC open"
def test_cmmtcr_is_not_stcr_or_eth_leadership_repair():
 r=prereg.build();assert "STCR" in r["mechanism"]["why_distinct"];assert r["research_boundary"]["prior_event_sets_reused"] is False;assert r["research_boundary"]["promoted_prior_control"] is False
def test_cmmtcr_hash_binds_core():
 r=prereg.build();assert r["manifest_hash"]==prereg.canonical_hash({k:v for k,v in r.items() if k!="manifest_hash"})
