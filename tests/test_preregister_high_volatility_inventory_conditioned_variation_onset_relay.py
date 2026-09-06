from training import preregister_high_volatility_inventory_conditioned_variation_onset_relay as p


def test_hvicvo_contract_is_singleton_outcome_blind_and_strict():
    value = p.build()
    p.validate(value)
    assert value["policy_id"] == "HVICVO-12"
    assert value["outcomes_opened"] is False and value["source_incidence_opened"] is False
    assert value["features"]["onset"].startswith("current variation rank>=0.65")
    assert value["mechanism"]["side"].startswith("product of strict completed BTC")
    assert value["clock"]["hold"] == "12 elapsed hours"
    assert value["research_boundary"]["repair_of_prior_candidate"] is False
    assert value["economic_gates"]["stop_on_first_failure"] is True
