from training import preregister_confirmation_ladder_multistate_ridge_relay as prereg
def test_clmsrr_preregistration_is_outcome_blind_and_hash_bound():
 r=prereg.build();assert r["policy_id"]=="CLMSRR-6";assert r["outcomes_opened"] is False;assert r["oos_source_incidence_opened"] is False;assert r["gross9_rows_opened"] is False;assert len(r["feature_contract"]["ordered_features"])==24;core={k:v for k,v in r.items() if k!="manifest_hash"};assert r["manifest_hash"]==prereg.canonical_hash(core)
