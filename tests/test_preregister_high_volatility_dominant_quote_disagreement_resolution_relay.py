from training import preregister_high_volatility_dominant_quote_disagreement_resolution_relay as prereg


def test_preregistration_is_outcome_blind_and_singleton():
    payload = prereg.build()
    prereg.validate(payload)
    assert payload["policy_id"] == "HVDQDR-8"
    assert payload["singleton"]
    assert not payload["outcomes_opened"]
    assert not payload["source_incidence_opened"]
    assert not payload["gross9_rows_opened"]
    assert payload["policy"]["variation_rank_min"] == 0.65
    assert payload["diagnostic_controls"]["cannot_be_promoted"]
