from training import preregister_high_volatility_credit_risk_appetite_relay as p
def test_boundary():
 x=p.build();assert x["policy_id"]=="HVCREDIT-24" and x["outcomes_opened"] is False and x["source_incidence_opened"] is False and x["gross9_rows_opened"] is False and x["singleton"] is True
 assert x["features"]["no_adjusted_close"] is True and x["policy"]["variation_rank_min"]==.65
 assert x["research_boundary"]["credit_values_opened"] is False and x["research_boundary"]["grid"] is False and x["research_boundary"]["repair_of_prior_candidate"] is False and x["research_boundary"]["promoted_prior_control"] is False
def test_hash():
 x=p.build();core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==p.canonical_hash(core)
