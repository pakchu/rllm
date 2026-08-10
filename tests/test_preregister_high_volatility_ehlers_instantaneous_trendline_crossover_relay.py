from training import preregister_high_volatility_ehlers_instantaneous_trendline_crossover_relay as p
def test_boundary():
 x=p.build();assert x["policy_id"]=="HVEIT-24" and x["outcomes_opened"] is False and x["source_incidence_opened"] is False and x["gross9_rows_opened"] is False and x["singleton"] is True
 assert x["policy"]["alpha"]==.07 and x["policy"]["initialization_observations"]==6 and x["research_boundary"]["grid"] is False
def test_hash():
 x=p.build();core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==p.canonical_hash(core)
