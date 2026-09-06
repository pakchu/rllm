from training import preregister_high_volatility_online_expert_rotation_relay as p
def test_boundary():
 x=p.build();p.validate(x);assert x["policy_id"]=="HVOER-8";assert x["candidate_incidence_opened"] is False;assert x["current_candidate_outcomes_opened"] is False;assert x["gross9_rows_opened"] is False
def test_rotation_contract():
 x=p.build();assert x["policy"]["expert_memory_labels"]==60;assert x["policy"]["minimum_mature_labels"]==30;assert x["features"]["decision_grid"].startswith("exact 02:00");assert x["policy"]["hold_hours"]==8
def test_gates():
 x=p.build();assert x["source_support_gates"]["minimum_events"]=={"train":8,"test":12,"eval":12,"final":8};assert x["novelty_gates"]["candidate_near_6h_share_max"]==.35;assert x["research_boundary"]["repair_of_prior_candidate"] is False
