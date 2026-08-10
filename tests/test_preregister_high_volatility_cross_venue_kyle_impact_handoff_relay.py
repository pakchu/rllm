from training import preregister_high_volatility_cross_venue_kyle_impact_handoff_relay as prereg


def test_hvckihr_is_outcome_blind_singleton():
    result = prereg.build()
    assert result["policy_id"] == "HVCKIHR-8"
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert result["singleton"] is True
    assert result["research_boundary"]["candidate_count"] == 1
    assert result["research_boundary"]["grid"] is False
    assert result["research_boundary"]["repair_of_prior_candidate"] is False


def test_hvckihr_freezes_impact_handoff_policy():
    result = prereg.build()
    policy = result["policy"]
    assert policy["handoff_rank_min"] == 0.80
    assert policy["variation_rank_min"] == 0.65
    assert policy["hold_hours"] == 8
    assert result["source_plan"]["spot"]["table"] == "bars_binance_spot"
    assert result["economic_gates"]["cagr_to_strict_mdd_min"] == 3.0


def test_hvckihr_hash_binds_core():
    result = prereg.build()
    assert result["manifest_hash"] == prereg.canonical_hash(
        {key: value for key, value in result.items() if key != "manifest_hash"}
    )
