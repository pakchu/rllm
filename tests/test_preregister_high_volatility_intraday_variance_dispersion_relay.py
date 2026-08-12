from training import preregister_high_volatility_intraday_variance_dispersion_relay as p
def test_frozen_singleton_contract():
 x=p.build();p.validate(x);assert x["policy_id"]=="HVIVDR-8";assert x["source_incidence_opened"] is False;assert x["research_boundary"]["grid"] is False;assert x["policy"]["subwindows"]==16
def test_strict_gates():
 x=p.build();assert x["source_support_gates"]["minimum_events"]=={"train":8,"test":12,"eval":12,"final":8};assert x["economic_gates"]["cagr_to_strict_mdd_min"]==3.;assert x["novelty_gates"]["candidate_near_6h_share_max"]==.35
