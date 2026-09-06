from training import preregister_high_volatility_know_sure_thing_crossover_relay as p
def test_boundary():
 x=p.build();assert x["policy_id"]=="HVKST-24" and x["outcomes_opened"] is False and x["source_incidence_opened"] is False and x["gross9_rows_opened"] is False and x["singleton"] is True
 assert x["policy"]["roc_periods"]==[10,15,20,30] and x["policy"]["sma_periods"]==[10,10,10,15] and x["policy"]["weights"]==[1,2,3,4] and x["policy"]["signal_periods"]==9 and x["research_boundary"]["grid"] is False
def test_hash():
 x=p.build();core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==p.canonical_hash(core)
