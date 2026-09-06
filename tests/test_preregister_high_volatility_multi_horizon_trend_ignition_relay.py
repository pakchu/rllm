from training import preregister_high_volatility_multi_horizon_trend_ignition_relay as p


def test_hvmti_contract_is_singleton_outcome_blind_and_strict():
    value=p.build();p.validate(value)
    assert value['policy_id']=='HVMTI-48'
    assert value['outcomes_opened'] is False and value['source_incidence_opened'] is False
    assert value['features']['trend_consensus'].endswith('identical')
    assert value['features']['ignition'].startswith('current variation rank>=0.65')
    assert value['clock']['hold']=='48 elapsed hours'
    assert value['research_boundary']['repair_of_prior_candidate'] is False
