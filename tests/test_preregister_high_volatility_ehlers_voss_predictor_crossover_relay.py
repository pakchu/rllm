from training import preregister_high_volatility_ehlers_voss_predictor_crossover_relay as p
def test_frozen():
 x=p.build();p.validate(x)
 assert x["policy_id"]=="HVEVP-24" and x["policy"]["period"]==20 and x["policy"]["predict"]==3 and x["policy"]["bandwidth"]==.25 and x["policy"]["order"]==9
 assert x["diagnostic_controls"]["cannot_be_promoted"] and not x["research_boundary"]["candidate_incidence_opened"]
