from training import preregister_high_volatility_cross_quote_predictive_leadership_relay as prereg


def test_preregistration_is_deterministic_and_sealed():
    first, second = prereg.build(), prereg.build()
    assert first == second
    prereg.validate(first)
    assert first["policy_id"] == "HVQPLR-6"
    assert first["features"]["books"] == ["BTCUSDT", "BTCUSDC", "BTCFDUSD"]
    assert first["policy"]["lead_share_min"] == 0.60
    assert first["clock"]["hold"] == "6 elapsed hours"
    assert not first["outcomes_opened"] and not first["source_incidence_opened"] and not first["gross9_rows_opened"]
