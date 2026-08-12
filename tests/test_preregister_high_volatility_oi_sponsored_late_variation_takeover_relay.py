from training import preregister_high_volatility_oi_sponsored_late_variation_takeover_relay as p


def test_hvoilvt_contract_is_singleton_outcome_blind_and_strict():
    value = p.build(); p.validate(value)
    assert value["policy_id"] == "HVOILVT-12"
    assert value["outcomes_opened"] is False and value["source_incidence_opened"] is False
    assert value["features"]["directional_takeover"].endswith("opposite")
    assert value["features"]["oi_change"].endswith("strictly positive")
    assert value["clock"]["side"] == "follow completed final-two-hour BTC return"
    assert value["research_boundary"]["repair_of_prior_candidate"] is False
