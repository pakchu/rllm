from training import preregister_high_volatility_basis_regime_transition_relay as p
def test_prereg_is_deterministic_and_blind():
 x=p.build();assert x==p.build();p.validate(x);assert not x['outcomes_opened'] and not x['source_incidence_opened'] and not x['gross9_rows_opened']
def test_policy_and_gates_are_frozen():
 x=p.build();assert x['policy_id']=='HVBSRT-8';assert x['policy']['minimum_history_hours']==1440;assert x['policy']['hold_hours']==8;assert x['clock']['side']=='new basis-residual sign';assert x['novelty_gates']['must_pass_before_economics'] is True;assert x['economic_gates']['stop_on_first_failure'] is True;assert x['diagnostic_controls']['cannot_be_promoted'] is True
