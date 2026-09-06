from training import preregister_asian_peg_adjusted_won_stress_relay as prereg

def test_apwsr_is_outcome_blind_independent_singleton():
 r=prereg.build(); assert r["policy_id"]=="APWSR-12"; assert r["outcomes_opened"] is False; assert r["source_incidence_opened"] is False; assert r["gross9_rows_opened"] is False; assert r["singleton"] is True; assert r["research_boundary"]["candidate_count"]==1; assert r["research_boundary"]["repair_of_prior_candidate"] is False; assert r["research_boundary"]["promoted_prior_control"] is False

def test_apwsr_freezes_peg_adjusted_spread_and_volatile_regime():
 r=prereg.build(); assert r["source_plan"]["fx"]["symbols"]==["USDKRW","USDHKD"]; assert r["policy"]["stress_absolute_rank_min"]==.70; assert r["policy"]["realized_variation_rank_min"]==.65; assert r["policy"]["hold_hours"]==12; assert r["clock"]["entry"]=="exact BTCUSDT D 08:05 UTC open"

def test_apwsr_is_not_hpprr_or_emdf_repair():
 r=prereg.build(); assert "HPPRR" in r["mechanism"]["why_distinct"]; assert r["research_boundary"]["prior_event_sets_reused"] is False; assert r["research_boundary"]["prior_candidate_outcomes_used_to_set_apwsr_source_direction_rank_hold_or_clock"] is False

def test_apwsr_hash_binds_core():
 r=prereg.build(); assert r["manifest_hash"]==prereg.canonical_hash({k:v for k,v in r.items() if k!="manifest_hash"})
