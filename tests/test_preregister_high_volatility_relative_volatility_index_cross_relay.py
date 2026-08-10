from training import preregister_high_volatility_relative_volatility_index_cross_relay as p
def test_boundary():
 x=p.build();assert x["policy_id"]=="HVRVOL-24" and x["outcomes_opened"] is False and x["source_incidence_opened"] is False and x["gross9_rows_opened"] is False and x["singleton"] is True
 assert (x["policy"]["standard_deviation_periods"],x["policy"]["smoothing_periods"],x["policy"]["balance_line"])==(10,14,50.0) and x["research_boundary"]["grid"] is False
def test_hash():
 x=p.build();core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==p.canonical_hash(core)
