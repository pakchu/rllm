from training import preregister_high_volatility_state_conditioned_expert_rotation_relay as p

def test_hvscer_contract_is_singleton_causal_and_strict():
 value=p.build();p.validate(value)
 assert value['policy_id']=='HVSCER-24'
 assert value['current_candidate_outcomes_opened'] is False and value['candidate_incidence_opened'] is False
 assert value['features']['label_origin_gate'].endswith('variation rank>=0.65')
 assert value['features']['rotation'].startswith('current conditional winner differs')
 assert value['features']['positive_score'].endswith('score>0')
 assert value['clock']['hold']=='24 elapsed hours'
 assert value['research_boundary']['repair_of_prior_candidate'] is False
