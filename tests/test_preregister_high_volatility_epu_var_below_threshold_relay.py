from training import preregister_high_volatility_epu_var_below_threshold_relay as prereg


def test_preregistration_freezes_published_threshold_rule_before_incidence() -> None:
    result = prereg.build()
    prereg.validate(result)
    assert result["policy_id"] == "HVEPUVBT-24"
    assert result["singleton"]
    assert not result["outcomes_opened"]
    assert not result["source_incidence_opened"]
    assert not result["gross9_rows_opened"]
    assert result["policy"]["var_lags"] == 1
    assert result["policy"]["forecast_dispersion_minimum"] == 365
    assert "otherwise +1" in result["mechanism"]["side"]
    assert result["research_boundary"]["prior_epu_source_rows_opened_for_terminal_hvepuh"]
    assert not result["research_boundary"]["repair_of_prior_candidate"]
