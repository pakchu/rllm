from training import preregister_high_volatility_cash_participation_migration_relay as prereg
def test_hvcpmr_preregistration_is_outcome_blind_and_hash_bound():
 r=prereg.build();assert r["policy_id"]=="HVCPMR-8";assert r["outcomes_opened"] is False;assert r["source_incidence_opened"] is False;assert r["gross9_rows_opened"] is False;assert r["policy"]["migration_rank_min"]==.70;core={k:v for k,v in r.items() if k!="manifest_hash"};assert r["manifest_hash"]==prereg.canonical_hash(core)
