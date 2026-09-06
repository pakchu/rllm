from training import preregister_high_volatility_range_boundary_fade_relay as prereg


def test_hvrbfr_is_honest_outcome_sealed_singleton():
    result = prereg.build()
    assert result["policy_id"] == "HVRBFR-2"
    assert result["exact_candidate_outcomes_opened"] is False
    assert result["exact_candidate_source_incidence_opened"] is False
    assert result["singleton"] is True
    assert result["research_boundary"]["related_price_structure_family_outcomes_known"] is True
    assert result["research_boundary"]["candidate_count"] == 1
    assert result["research_boundary"]["grid"] is False


def test_hvrbfr_freezes_symmetric_high_volatility_boundary_fade():
    result = prereg.build()
    policy = result["policy"]
    assert policy["history_observations"] == 180
    assert policy["minimum_history_observations"] == 120
    assert policy["range_rank_min"] == 0.8
    assert policy["lower_close_location_max"] == 0.1
    assert policy["upper_close_location_min"] == 0.9
    assert policy["hold_hours"] == 2
    assert result["economic_gates"]["cagr_to_strict_mdd_min"] == 3.0
    assert result["economic_gates"]["stress_cagr_to_strict_mdd_min"] == 2.5


def test_hvrbfr_hash_binds_core():
    result = prereg.build()
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == prereg.canonical_hash(core)
