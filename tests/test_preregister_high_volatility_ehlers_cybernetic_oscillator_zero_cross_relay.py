from training import preregister_high_volatility_ehlers_cybernetic_oscillator_zero_cross_relay as p
def test_frozen():
 x=p.build();p.validate(x);assert x["policy_id"]=="HVECO-24" and x["policy"]["high_pass_periods"]==30 and x["policy"]["low_pass_periods"]==20 and x["policy"]["rms_periods"]==100
 assert x["diagnostic_controls"]["cannot_be_promoted"] and not x["research_boundary"]["candidate_incidence_opened"]
