from training import preregister_high_volatility_quote_turnover_concentration_continuation_relay as prereg

def test_hvtccr_boundary_and_policy_are_frozen():
 r=prereg.build();assert r["policy_id"]=="HVTCCR-8";assert r["outcomes_opened"] is False;assert r["source_incidence_opened"] is False;assert r["gross9_rows_opened"] is False;assert r["singleton"] is True;assert r["research_boundary"]["candidate_count"]==1;assert r["research_boundary"]["grid"] is False;assert r["research_boundary"]["repair_of_prior_candidate"] is False
 p=r["policy"];assert p["concentration_rank_min"]==.8;assert p["variation_rank_min"]==.65;assert p["hold_hours"]==8

def test_hvtccr_hash_binds_core():
 r=prereg.build();assert r["manifest_hash"]==prereg.canonical_hash({k:v for k,v in r.items() if k!="manifest_hash"})
