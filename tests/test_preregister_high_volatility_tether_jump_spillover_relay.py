from training import preregister_high_volatility_tether_jump_spillover_relay as prereg


def test_preregistration_freezes_unopened_usdt_jump_candidate() -> None:
    result = prereg.build()
    prereg.validate(result)
    assert result["policy_id"] == "HVUSDTJ-24"
    assert result["singleton"]
    assert not result["outcomes_opened"]
    assert not result["source_incidence_opened"]
    assert not result["gross9_rows_opened"]
    assert result["source_plan"]["coin_metrics"]["asset"] == "usdt"
    assert result["source_plan"]["coin_metrics"]["metrics"] == [
        "PriceUSD",
        "AssetEODCompletionTime",
    ]
    assert result["mechanism"]["side"].startswith("positive USDT log return maps BTC short")
    assert not result["research_boundary"]["usdt_historical_source_values_opened"]
    assert not result["research_boundary"]["repair_of_prior_candidate"]
