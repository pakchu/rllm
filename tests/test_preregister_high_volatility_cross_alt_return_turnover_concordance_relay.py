from training import preregister_high_volatility_cross_alt_return_turnover_concordance_relay as p
def test_boundary():
 x=p.build();assert x["policy_id"]=="HVCARTC-8" and x["outcomes_opened"] is False and x["source_incidence_opened"] is False and x["gross9_rows_opened"] is False and x["singleton"] is True
 assert len(x["features"]["symbols"])==6 and x["policy"]["minimum_turnover_history_decisions"]==180 and x["policy"]["hold_hours"]==8 and x["research_boundary"]["grid"] is False
def test_hash():
 x=p.build();core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==p.canonical_hash(core)
