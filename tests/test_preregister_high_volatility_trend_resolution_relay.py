from training import preregister_high_volatility_trend_resolution_relay as p


def test_hvtrr_contract_is_singleton_outcome_blind_and_strict():
    value=p.build();p.validate(value)
    assert value['policy_id']=='HVTRR-72'
    assert value['outcomes_opened'] is False and value['source_incidence_opened'] is False
    assert value['features']['resolution'].startswith('current 5d/20d signs agree')
    assert value['features']['variation_gate']=='current variation rank>=0.65'
    assert value['clock']['hold']=='72 elapsed hours'
    assert value['research_boundary']['repair_of_prior_candidate'] is False
