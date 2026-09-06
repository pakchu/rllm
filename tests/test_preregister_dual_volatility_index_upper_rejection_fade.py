from training import preregister_dual_volatility_index_upper_rejection_fade as prereg


def test_dvurf_is_outcome_blind_independent_singleton():
    result = prereg.build()
    assert result["policy_id"] == "DVURF-6"
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert result["singleton"] is True
    assert result["research_boundary"]["candidate_count"] == 1
    assert result["research_boundary"]["grid"] is False
    assert result["research_boundary"]["repair_of_prior_candidate"] is False


def test_dvurf_freezes_dual_rejection_ranks_and_clock():
    result = prereg.build()
    assert result["policy"]["joint_rejection_rank_min"] == 0.75
    assert result["policy"]["joint_range_rank_min"] == 0.50
    assert result["policy"]["btc_absolute_return_rank_min"] == 0.75
    assert result["policy"]["hold_hours"] == 6
    assert result["clock"]["entry"] == "exact BTCUSDT T+5m open"
    assert result["clock"]["side"] == "opposite completed-hour BTC return sign"


def test_dvurf_keeps_rv20_as_post_stage_audit_only():
    result = prereg.build()
    assert result["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False
    assert result["research_boundary"]["prior_event_sets_reused"] is False
    assert result["research_boundary"]["promoted_prior_control"] is False


def test_dvurf_hash_binds_core():
    result = prereg.build()
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == prereg.canonical_hash(core)
