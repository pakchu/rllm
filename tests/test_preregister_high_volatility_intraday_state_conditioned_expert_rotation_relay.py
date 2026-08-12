from training import preregister_high_volatility_intraday_state_conditioned_expert_rotation_relay as p

def test_hviscer_contract_is_singleton_causal_and_strict():
 value=p.build();p.validate(value)
 assert value['policy_id']=='HVISCER-8'
 assert value['current_candidate_outcomes_opened'] is False and value['candidate_incidence_opened'] is False
 assert value['features']['experts_in_tie_order']==['momentum_4h','reversal_4h','momentum_12h','reversal_12h']
 assert value['features']['label_origin_gate'].endswith('variation rank>=0.65')
 assert value['clock']['hold']=='8 elapsed hours'
 assert value['research_boundary']['repair_of_prior_candidate'] is False
