from training import preregister_high_volatility_macd_crossover_relay as p
def test_boundary():
 x=p.build();assert x["policy_id"]=="HVMACD-24" and x["outcomes_opened"] is False and x["source_incidence_opened"] is False and x["gross9_rows_opened"] is False and x["singleton"] is True
 assert x["policy"]["fast_ema_days"]==12 and x["policy"]["slow_ema_days"]==26 and x["policy"]["signal_ema_days"]==9 and x["research_boundary"]["grid"] is False
def test_hash():
 x=p.build();core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==p.canonical_hash(core)
