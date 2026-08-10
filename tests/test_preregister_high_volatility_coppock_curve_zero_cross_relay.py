from training import preregister_high_volatility_coppock_curve_zero_cross_relay as p
def test_boundary():
 x=p.build();assert x["policy_id"]=="HVCC-24" and x["outcomes_opened"] is False and x["source_incidence_opened"] is False and x["gross9_rows_opened"] is False and x["singleton"] is True
 assert (x["policy"]["short_roc_periods"],x["policy"]["long_roc_periods"],x["policy"]["wma_periods"])==(11,14,10) and x["research_boundary"]["grid"] is False
def test_hash():
 x=p.build();core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==p.canonical_hash(core)
