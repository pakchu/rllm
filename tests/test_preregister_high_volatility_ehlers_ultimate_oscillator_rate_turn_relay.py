from training import preregister_high_volatility_ehlers_ultimate_oscillator_rate_turn_relay as p
def test_frozen():
 x=p.build();p.validate(x);assert x["policy_id"]=="HVEUO-24" and x["policy"]["band_edge"]==20 and x["policy"]["bandwidth"]==2 and x["policy"]["rms_periods"]==100
 assert x["diagnostic_controls"]["cannot_be_promoted"] and not x["research_boundary"]["candidate_incidence_opened"] and x["research_boundary"]["classic_larry_williams_ultimate_oscillator_found_and_distinguished"]
