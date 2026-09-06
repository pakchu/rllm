from training import preregister_high_volatility_ehlers_reflex_zero_cross_relay as p
def test_frozen():
 x=p.build();p.validate(x)
 assert x["policy_id"]=="HVERF-24" and x["policy"]["length"]==20
 assert x["policy"]["mean_square_current_weight"]==.04 and x["policy"]["mean_square_prior_weight"]==.96
 assert x["diagnostic_controls"]["cannot_be_promoted"] and not x["research_boundary"]["candidate_incidence_opened"]
