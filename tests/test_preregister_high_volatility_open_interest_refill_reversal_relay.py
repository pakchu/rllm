from training import preregister_high_volatility_open_interest_refill_reversal_relay as p


def test_hvoirr_contract_is_singleton_outcome_blind_and_strict():
    value = p.build()
    p.validate(value)
    assert value["policy_id"] == "HVOIRR-8"
    assert value["outcomes_opened"] is False
    assert value["source_incidence_opened"] is False
    assert value["features"]["refill_fraction"].endswith(">=0.25")
    assert value["clock"]["hold"] == "8 elapsed hours"
    assert value["source_support_gates"]["minimum_events"] == {"train": 8, "test": 12, "eval": 12, "final": 8}
    assert value["research_boundary"]["repair_of_prior_candidate"] is False
    assert value["economic_gates"]["stop_on_first_failure"] is True
