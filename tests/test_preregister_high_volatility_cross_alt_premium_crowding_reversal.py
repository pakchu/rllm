from training import preregister_high_volatility_cross_alt_premium_crowding_reversal as prereg
def test_hvcapcr_preregistration_is_outcome_blind_and_hash_bound():
 r=prereg.build();assert r["policy_id"]=="HVCAPCR-6";assert r["outcomes_opened"] is False;assert r["source_incidence_opened"] is False;assert r["gross9_rows_opened"] is False;assert r["policy"]["residual_rank_min"]==.80;core={k:v for k,v in r.items() if k!="manifest_hash"};assert r["manifest_hash"]==prereg.canonical_hash(core)
