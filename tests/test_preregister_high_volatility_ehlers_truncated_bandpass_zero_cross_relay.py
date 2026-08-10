from training import preregister_high_volatility_ehlers_truncated_bandpass_zero_cross_relay as p
def test_frozen():
 x=p.build();p.validate(x)
 assert x["policy_id"]=="HVTBP-24" and x["policy"]["period"]==20 and x["policy"]["bandwidth"]==.1 and x["policy"]["truncation_length"]==10
 assert x["diagnostic_controls"]["cannot_be_promoted"] and not x["research_boundary"]["candidate_incidence_opened"]
