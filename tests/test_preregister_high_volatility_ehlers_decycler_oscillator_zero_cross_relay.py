from training import preregister_high_volatility_ehlers_decycler_oscillator_zero_cross_relay as p
def test_frozen():
 x=p.build();p.validate(x);assert x["policy_id"]=="HVDEC-24" and x["policy"]=={"hp_periods":125,"k":1.0,"fast_control_hp_periods":100,"fast_control_k":1.2,"variation_hours":24,"variation_history_decisions":180,"minimum_variation_history_decisions":120,"variation_rank_min":.65,"entry_delay_minutes":5,"hold_hours":24,"leverage":.5,"base_cost_per_notional_side":.0006,"stress_cost_per_notional_side":.001}
 assert x["diagnostic_controls"]["cannot_be_promoted"] and not x["research_boundary"]["candidate_incidence_opened"]
