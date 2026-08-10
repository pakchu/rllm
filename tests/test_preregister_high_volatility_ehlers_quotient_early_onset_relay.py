from training import preregister_high_volatility_ehlers_quotient_early_onset_relay as p
def test_frozen():
 x=p.build();p.validate(x)
 assert x["policy_id"]=="HVEQT-24" and x["policy"]["high_pass_period"]==100 and x["policy"]["low_pass_period"]==30
 assert x["policy"]["peak_decay"]==.991 and x["policy"]["linearity_control"]==.85
 assert x["diagnostic_controls"]["cannot_be_promoted"] and not x["research_boundary"]["candidate_incidence_opened"]
