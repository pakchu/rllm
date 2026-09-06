from training import preregister_high_volatility_variance_ratio_trend_onset_relay as p

def test_boundary():
 x=p.build();assert x["policy_id"]=="HVVRT-12" and x["outcomes_opened"] is False and x["source_incidence_opened"] is False and x["gross9_rows_opened"] is False and x["singleton"] is True
 assert x["policy"]["sample_returns"]==288 and x["policy"]["aggregation_steps"]==12 and x["policy"]["unity"]==1
 assert x["research_boundary"]["grid"] is False and x["research_boundary"]["repair_of_prior_candidate"] is False and x["research_boundary"]["promoted_prior_control"] is False

def test_hash():
 x=p.build();core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==p.canonical_hash(core)
