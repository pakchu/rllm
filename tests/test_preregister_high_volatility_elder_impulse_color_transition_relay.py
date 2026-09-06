from training import preregister_high_volatility_elder_impulse_color_transition_relay as p
def test_boundary():
 x=p.build();assert x["policy_id"]=="HVEIS-24" and x["outcomes_opened"] is False and x["source_incidence_opened"] is False and x["gross9_rows_opened"] is False and x["singleton"] is True
 assert x["policy"]["trend_ema_periods"]==13 and x["policy"]["macd_fast_periods"]==12 and x["policy"]["macd_slow_periods"]==26 and x["policy"]["macd_signal_periods"]==9 and "volume" not in x["source_plan"]["bars"]["columns"] and x["research_boundary"]["grid"] is False
def test_hash():
 x=p.build();core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==p.canonical_hash(core)
