from training import preregister_high_volatility_directional_ticket_size_asymmetry_relay as p
def test_blind_singleton():
 x=p.build();p.validate(x);assert x["policy_id"]=="HVDTSA-8" and x["outcomes_opened"] is False and x["source_incidence_opened"] is False and x["gross9_rows_opened"] is False and x["research_boundary"]["candidate_count"]==1
def test_frozen():
 x=p.build();assert x["policy"]["minimum_direction_minutes"]==120 and x["policy"]["asymmetry_magnitude_rank_min"]==.75 and x["policy"]["variation_rank_min"]==.65 and x["clock"]["side"]=="strict sign(ticket_asymmetry)" and x["clock"]["hold"]=="8 elapsed hours"
