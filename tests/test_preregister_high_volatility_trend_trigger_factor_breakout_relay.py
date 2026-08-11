from training import preregister_high_volatility_trend_trigger_factor_breakout_relay as p

def test_boundary():
 x=p.build();assert x["policy_id"]=="HVTTF-24" and x["outcomes_opened"] is False and x["source_incidence_opened"] is False and x["gross9_rows_opened"] is False and x["singleton"] is True
 assert x["policy"]["trend_trigger_periods"]==15 and x["policy"]["upper_trigger"]==100 and x["policy"]["lower_trigger"]==-100
 assert x["research_boundary"]["grid"] is False and x["research_boundary"]["repair_of_prior_candidate"] is False and x["research_boundary"]["promoted_prior_control"] is False

def test_hash():
 x=p.build();core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==p.canonical_hash(core)
