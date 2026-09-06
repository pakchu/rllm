from training import preregister_high_volatility_drawup_drawdown_asymmetry_relay as p
def test_boundary():
 x=p.build();assert x["policy_id"]=="HVDUDAR-8" and x["outcomes_opened"] is False and x["source_incidence_opened"] is False and x["gross9_rows_opened"] is False and x["singleton"] is True
 assert x["research_boundary"]["candidate_count"]==1 and x["research_boundary"]["grid"] is False and x["research_boundary"]["repair_of_prior_candidate"] is False
 assert x["policy"]["asymmetry_rank_min"]==.70 and x["policy"]["variation_rank_min"]==.65 and x["policy"]["hold_hours"]==8
def test_hash():
 x=p.build();core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==p.canonical_hash(core)
