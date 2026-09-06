from training import preregister_high_volatility_macro_news_sentiment_dispersion_relay as prereg


def test_singleton_is_outcome_and_incidence_blind():
    value = prereg.build()
    prereg.validate(value)
    assert value["policy_id"] == "HVMNSD-24"
    assert value["singleton"] is True
    assert value["outcomes_opened"] is False
    assert value["source_incidence_opened"] is False
    assert value["gross9_rows_opened"] is False


def test_dispersion_operator_and_direction_are_frozen():
    value = prereg.build()
    assert value["policy"]["dispersion_window_days"] == 7
    assert value["policy"]["dispersion_estimator"] == "population_standard_deviation_ddof_0"
    assert value["clock"]["side"] == "negative dispersion-change sign; rising dispersion short"
    assert value["policy"]["btc_variation_rank_min"] == 0.65
    assert "sentiment_shock_rank" not in value["policy"]


def test_gates_and_no_repair_boundary_are_frozen():
    value = prereg.build()
    assert value["source_support_gates"]["minimum_events"] == {
        "train": 8,
        "test": 12,
        "eval": 12,
        "final": 8,
    }
    assert value["economic_gates"]["cagr_to_strict_mdd_min"] == 3.0
    assert value["research_boundary"]["repair_of_prior_candidate"] is False
    assert value["diagnostic_controls"]["cannot_be_promoted"] is True
