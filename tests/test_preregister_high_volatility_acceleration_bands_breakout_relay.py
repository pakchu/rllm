from training import preregister_high_volatility_acceleration_bands_breakout_relay as p
def test_boundary():
 v=p.build();assert v["policy_id"]=="HVABANDS-B10-W4-24" and v["outcomes_opened"] is False and v["source_incidence_opened"] is False and v["gross9_rows_opened"] is False
 assert v["policy"]["period"]==10 and v["policy"]["width"]==4 and v["policy"]["hold_hours"]==24 and v["research_boundary"]["candidate_count"]==1 and v["research_boundary"]["grid"] is False
def test_hash():
 v=p.build();assert v["manifest_hash"]==p.canonical_hash({k:x for k,x in v.items() if k!="manifest_hash"})
