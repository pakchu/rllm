from training import preregister_high_volatility_eth_leadership_relay as prereg


def test_hvelr_is_outcome_blind_independent_singleton():
    result = prereg.build()
    assert result["policy_id"] == "HVELR-6"
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["singleton"] is True
    assert result["research_boundary"]["candidate_count"] == 1
    assert result["research_boundary"]["grid"] is False
    assert result["research_boundary"]["repair_of_prior_candidate"] is False
    assert result["research_boundary"]["promoted_prior_control"] is False


def test_hvelr_freezes_cross_asset_leadership_and_rank_history():
    result = prereg.build()
    policy = result["policy"]
    assert policy["history_observations"] == 270
    assert policy["minimum_history_observations"] == 180
    assert policy["btc_variation_rank_min"] == 0.65
    assert policy["eth_absolute_return_rank_min"] == 0.80
    assert policy["minimum_eth_to_btc_absolute_return_ratio"] == 1.5
    assert policy["hold_hours"] == 6
    assert result["source_plan"]["binance_1m"]["symbols"] == ["BTCUSDT", "ETHUSDT"]


def test_hvelr_hash_binds_core():
    result = prereg.build()
    assert result["manifest_hash"] == prereg.canonical_hash({key: value for key, value in result.items() if key != "manifest_hash"})
