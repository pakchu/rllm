from training import preregister_high_volatility_hammer_hanging_man_context_reversal_relay as p
def test_boundary():
 v=p.build();assert v["policy_id"]=="HVHHM-C10-N5-8" and v["outcomes_opened"] is False and v["source_incidence_opened"] is False and v["gross9_rows_opened"] is False
 assert v["policy"]["indicator_period_hours"]==12 and v["policy"]["hold_hours"]==8 and v["research_boundary"]["candidate_count"]==1 and v["research_boundary"]["grid"] is False
def test_hash():
 v=p.build();assert v["manifest_hash"]==p.canonical_hash({k:x for k,x in v.items() if k!="manifest_hash"})
