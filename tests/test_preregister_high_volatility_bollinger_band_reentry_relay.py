from training import preregister_high_volatility_bollinger_band_reentry_relay as p
def test_boundary():
 x=p.build();assert x["policy_id"]=="HVBB-24" and x["outcomes_opened"] is False and x["source_incidence_opened"] is False and x["gross9_rows_opened"] is False and x["singleton"] is True
 assert x["policy"]["window_periods"]==20 and x["policy"]["standard_deviation_ddof"]==0 and x["policy"]["band_multiplier"]==2 and "quote_asset_volume" not in x["source_plan"]["bars"]["columns"] and x["research_boundary"]["grid"] is False
def test_hash():
 x=p.build();core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==p.canonical_hash(core)
