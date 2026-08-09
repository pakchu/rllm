from training import preregister_high_volatility_geopolitical_act_threat_transition_relay as prereg


def test_preregistration_is_outcome_blind_before_source_download():
    policy = prereg.build()
    prereg.validate(policy)
    assert policy["policy_id"] == "HVGATA-24"
    assert policy["singleton"]
    assert not policy["outcomes_opened"]
    assert not policy["source_incidence_opened"]
    assert not policy["gross9_rows_opened"]
    assert policy["policy"]["publication_delay_days"] == 2
    assert policy["source_plan"]["gpr"]["download_after_preregistration_commit"]
