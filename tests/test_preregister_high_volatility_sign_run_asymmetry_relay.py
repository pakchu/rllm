from training import preregister_high_volatility_sign_run_asymmetry_relay as prereg

def test_preregistration_freezes_single_outcome_blind_run_asymmetry_policy():
    policy=prereg.build();prereg.validate(policy)
    assert policy['policy_id']=='HVSRAR-8' and policy['singleton']
    assert not policy['outcomes_opened'] and not policy['source_incidence_opened'] and not policy['gross9_rows_opened']
    assert policy['policy']['run_share_rank_min']==.8 and policy['policy']['variation_rank_min']==.65
    assert policy['clock']['hold']=='8 elapsed hours' and policy['research_boundary']['candidate_count']==1
