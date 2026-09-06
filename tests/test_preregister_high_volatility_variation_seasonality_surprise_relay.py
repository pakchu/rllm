from training import preregister_high_volatility_variation_seasonality_surprise_relay as p


def test_singleton_boundary():
    value = p.build()
    p.validate(value)
    assert value["policy_id"] == "HVVSSR-8"
    assert value["source_incidence_opened"] is False
    assert value["research_boundary"]["grid"] is False


def test_frozen_gates_and_seasonal_reference():
    value = p.build()
    assert value["policy"]["seasonal_weeks"] == 8
    assert value["policy"]["surprise_rank_min"] == 0.80
    assert value["source_support_gates"]["minimum_events"] == {
        "train": 8,
        "test": 12,
        "eval": 12,
        "final": 8,
    }
    assert value["economic_gates"]["cagr_to_strict_mdd_min"] == 3.0
    assert "D-56d" in value["features"]["seasonal_reference"]
