from training import preregister_high_volatility_month_phase_seasonality_relay as prereg


def test_outcome_sequenced_singleton():
    report = prereg.build(); prereg.validate(report)
    assert report["policy_id"] == "HVMPS-12"
    assert report["singleton"] is True
    assert report["oos_outcomes_opened"] is False
    assert report["oos_source_incidence_opened"] is False
    assert report["training_contract"]["refit_after_2022_12_31"] is False
