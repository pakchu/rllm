from training import preregister_high_volatility_oi_conditioned_late_variation_ignition_relay as p


def test_hvoilvi_contract_is_singleton_outcome_blind_and_strict():
    value = p.build(); p.validate(value)
    assert value["policy_id"] == "HVOILVI-12"
    assert value["outcomes_opened"] is False and value["source_incidence_opened"] is False
    assert value["features"]["late_variance_share"].startswith("sum of final 24")
    assert value["features"]["eligibility"] == "variation rank>=0.65 and late-variance-share rank>=0.75"
    assert value["clock"]["hold"] == "12 elapsed hours"
    assert value["research_boundary"]["repair_of_prior_candidate"] is False
