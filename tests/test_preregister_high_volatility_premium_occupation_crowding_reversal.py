from training import preregister_high_volatility_premium_occupation_crowding_reversal as prereg
def test_hvpocr_preregistration_is_outcome_blind_and_hash_bound():
 r=prereg.build();assert r["policy_id"]=="HVPOCR-8";assert r["outcomes_opened"] is False;assert r["source_incidence_opened"] is False;assert r["gross9_rows_opened"] is False;assert r["policy"]["upper_occupation_min"]==.75;core={k:v for k,v in r.items() if k!="manifest_hash"};assert r["manifest_hash"]==prereg.canonical_hash(core)
