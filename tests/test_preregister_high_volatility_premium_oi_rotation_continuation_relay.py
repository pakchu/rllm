from training import preregister_high_volatility_premium_oi_rotation_continuation_relay as p


def test_hvporc_contract_is_singleton_outcome_blind_and_strict():
    value = p.build()
    p.validate(value)
    assert value["policy_id"] == "HVPORC-8"
    assert value["outcomes_opened"] is False
    assert value["source_incidence_opened"] is False
    assert value["features"]["premium_rotation"].endswith("opposite")
    assert value["features"]["current_oi_block"].endswith("current log change>0")
    assert value["clock"]["side"] == "follow current completed premium displacement"
    assert value["clock"]["hold"] == "8 elapsed hours"
    assert value["research_boundary"]["repair_of_prior_candidate"] is False
    assert value["economic_gates"]["stop_on_first_failure"] is True
