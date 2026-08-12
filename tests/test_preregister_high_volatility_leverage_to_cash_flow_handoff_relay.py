from training import preregister_high_volatility_leverage_to_cash_flow_handoff_relay as p
def test_boundary():
 x=p.build();p.validate(x);assert x["policy_id"]=="HVLCFH-8";assert x["source_incidence_opened"] is False;assert x["outcomes_opened"] is False;assert x["gross9_rows_opened"] is False
def test_operator_and_gates():
 x=p.build();assert x["policy"]["handoff_rank_min"]==.70;assert x["policy"]["variation_rank_min"]==.65;assert x["features"]["decision_grid"].startswith("exact 03:00");assert x["source_support_gates"]["minimum_events"]=={"train":8,"test":12,"eval":12,"final":8}
def test_no_repair():
 x=p.build();assert x["research_boundary"]["repair_of_prior_candidate"] is False;assert x["diagnostic_controls"]["cannot_be_promoted"] is True;assert x["economic_gates"]["cagr_to_strict_mdd_min"]==3.
