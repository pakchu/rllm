from training import preregister_high_volatility_ehlers_synthetic_oscillator_turn_relay as p
def test_frozen():
 x=p.build();p.validate(x);assert x["policy_id"]=="HVESO-24" and x["policy"]["lower_bound"]==15 and x["policy"]["upper_bound"]==25 and x["policy"]["trade_hann_periods"]==4
 assert x["diagnostic_controls"]["cannot_be_promoted"] and not x["research_boundary"]["candidate_incidence_opened"]
