from training import preregister_daily_oi_price_dislocation_reconciliation_relay as prereg


def test_dopdr_preregistration_is_outcome_blind_and_singleton():
    result = prereg.build()
    assert result["policy_id"] == "DOPDR-12"
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["singleton"] is True
    assert result["research_boundary"]["candidate_count"] == 1
    assert result["research_boundary"]["grid"] is False
    assert result["research_boundary"]["repair_of_prior_candidate"] is False


def test_dopdr_freezes_daily_clock_and_strict_gates():
    result = prereg.build()
    assert result["policy"]["prior_days"] == 180
    assert result["policy"]["prior_min_days"] == 126
    assert result["policy"]["realized_variation_rank_min"] == 0.65
    assert result["policy"]["displacement_rank_min"] == 0.65
    assert result["policy"]["entry_delay_minutes"] == 5
    assert result["policy"]["hold_hours"] == 12
    assert result["novelty_gates"]["candidate_near_6h_share_max"] == 0.35
    assert result["economic_gates"]["cagr_to_strict_mdd_min"] == 3.0
    assert result["economic_gates"]["stress_cagr_to_strict_mdd_min"] == 2.5


def test_dopdr_manifest_hash_binds_core():
    result = prereg.build()
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == prereg.canonical_hash(core)
