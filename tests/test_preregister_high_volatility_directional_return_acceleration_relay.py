from training import preregister_high_volatility_directional_return_acceleration_relay as prereg

def test_single_outcome_blind_directional_acceleration_policy_is_frozen():
    policy=prereg.build();prereg.validate(policy)
    assert policy['policy_id']=='HVDRA-8' and policy['singleton']
    assert not policy['outcomes_opened'] and not policy['source_incidence_opened'] and not policy['gross9_rows_opened']
    assert policy['policy']['acceleration_ratio_strict_min']==1 and policy['policy']['variation_rank_min']==.65
    assert policy['features']['halves']=='first 48 and final 48 completed 5m bars'
    assert policy['clock']['hold']=='8 elapsed hours'
