from training import preregister_high_volatility_premium_index_wick_rejection_relay as prereg


def test_preregistration_is_singleton_and_keeps_outcomes_closed():
    payload = prereg.build()
    prereg.validate(payload)
    assert payload["policy_id"] == "HVPIWR-12"
    assert payload["singleton"]
    assert not payload["outcomes_opened"]
    assert not payload["source_incidence_opened"]
    assert not payload["gross9_rows_opened"]
    assert payload["policy"]["magnitude_rank_min"] == 0.70
    assert payload["diagnostic_controls"]["cannot_be_promoted"]
