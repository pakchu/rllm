from training import preregister_high_volatility_chaikin_money_flow_zero_cross_relay as p
def test_boundary():
 x=p.build();assert x["policy_id"]=="HVCMF-24" and x["outcomes_opened"] is False and x["source_incidence_opened"] is False and x["gross9_rows_opened"] is False and x["singleton"] is True
 assert x["policy"]["cmf_periods"]==20 and x["policy"]["zero_level"]==0 and "quote_asset_volume" in x["source_plan"]["bars"]["columns"] and x["research_boundary"]["grid"] is False
def test_hash():
 x=p.build();core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==p.canonical_hash(core)
