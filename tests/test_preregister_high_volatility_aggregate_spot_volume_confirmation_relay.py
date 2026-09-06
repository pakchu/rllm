from training import preregister_high_volatility_aggregate_spot_volume_confirmation_relay as prereg


def test_preregistration_freezes_unopened_aggregate_spot_volume_candidate() -> None:
    result = prereg.build(); prereg.validate(result)
    assert result["policy_id"] == "HVASVC-24"
    assert result["singleton"]
    assert not result["outcomes_opened"] and not result["source_incidence_opened"] and not result["gross9_rows_opened"]
    assert result["source_plan"]["coin_metrics"]["metrics"] == ["volume_reported_spot_usd_1d", "AssetEODCompletionTime"]
    assert result["mechanism"]["side"].startswith("strict sign of BTCUSDT return")
    assert not result["research_boundary"]["historical_aggregate_spot_volume_values_opened"]
    assert not result["research_boundary"]["repair_of_prior_candidate"]
