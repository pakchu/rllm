from training import preregister_high_volatility_top_two_causal_expert_concordance_relay as p


def test_hvtcec_contract_is_singleton_causal_and_strict():
    value=p.build();p.validate(value)
    assert value['policy_id']=='HVTCEC-24'
    assert value['current_candidate_outcomes_opened'] is False and value['candidate_incidence_opened'] is False
    assert value['features']['experts_in_tie_order']==['momentum_4h','reversal_4h','momentum_24h','reversal_24h']
    assert value['features']['top_two_concordance'].endswith('strict nonzero side')
    assert value['clock']['hold']=='24 elapsed hours'
    assert value['research_boundary']['repair_of_prior_candidate'] is False
