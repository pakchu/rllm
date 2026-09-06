from training import preregister_high_volatility_smi_ergodic_crossover_relay as p
def test_boundary():
 x=p.build();assert x["policy_id"]=="HVSMIE-24" and x["outcomes_opened"] is False and x["source_incidence_opened"] is False and x["gross9_rows_opened"] is False and x["singleton"] is True
 assert (x["policy"]["fast_periods"],x["policy"]["slow_periods"],x["policy"]["signal_periods"])==(5,20,5) and x["research_boundary"]["grid"] is False
def test_hash():
 x=p.build();core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==p.canonical_hash(core)
