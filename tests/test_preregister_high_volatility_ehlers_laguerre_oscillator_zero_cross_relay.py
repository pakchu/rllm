from training import preregister_high_volatility_ehlers_laguerre_oscillator_zero_cross_relay as p
def test_frozen():
 x=p.build();p.validate(x);assert x["policy_id"]=="HVELO-24" and x["policy"]["gamma"]==.5 and x["policy"]["smoother_periods"]==30 and x["policy"]["rms_periods"]==100
 assert x["research_boundary"]["repository_laguerre_rsi_found_and_distinguished"] and not x["research_boundary"]["candidate_incidence_opened"]
