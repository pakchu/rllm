from training import preregister_daily_cross_venue_participation_rotation_relay as prereg


def test_dcvpr_preregistration_is_outcome_blind_singleton():
    result = prereg.build()
    assert result["policy_id"] == "DCVPR-12"
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["singleton"] is True
    assert result["research_boundary"]["candidate_count"] == 1
    assert result["research_boundary"]["grid"] is False
    assert result["research_boundary"]["repair_of_prior_candidate"] is False


def test_dcvpr_freezes_daily_rotation_and_strict_gates():
    result = prereg.build()
    policy = result["policy"]
    assert policy["participation_z_window_bars_5m"] == 288
    assert policy["rotation_lag_bars_5m"] == 288
    assert policy["absolute_rotation_rank_min"] == 0.65
    assert policy["realized_variation_rank_min"] == 0.65
    assert policy["hold_hours"] == 12
    assert result["economic_gates"]["cagr_to_strict_mdd_min"] == 3.0
    assert result["economic_gates"]["stress_cagr_to_strict_mdd_min"] == 2.5


def test_dcvpr_manifest_hash_binds_core():
    result = prereg.build()
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == prereg.canonical_hash(core)
