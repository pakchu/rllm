from training import preregister_high_volatility_wide_range_body_continuation_relay as p
def test_boundary():
 x=p.build();assert x["policy_id"]=="HVWRBC-24" and x["as_of_date"]=="2026-08-11" and x["outcomes_opened"] is False and x["source_incidence_opened"] is False and x["gross9_rows_opened"] is False and x["singleton"] is True
 assert x["research_boundary"]["candidate_count"]==1 and x["research_boundary"]["grid"] is False and x["research_boundary"]["repair_of_prior_candidate"] is False
 assert x["policy"]["body_efficiency_min"]==.5 and x["policy"]["hold_hours"]==24
def test_hash():
 x=p.build();core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==p.canonical_hash(core)
