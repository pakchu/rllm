from training import preregister_high_volatility_temporal_variance_ignition_relay as prereg


def test_hvtvir_is_outcome_blind_independent_singleton():
    result = prereg.build()
    assert result["policy_id"] == "HVTVIR-12"
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert result["singleton"] is True
    assert result["research_boundary"]["candidate_count"] == 1
    assert result["research_boundary"]["grid"] is False
    assert result["research_boundary"]["repair_of_prior_candidate"] is False


def test_hvtvir_freezes_temporal_variance_ignition_and_clock():
    result = prereg.build()
    assert result["policy"]["return_bars"] == 288
    assert result["policy"]["late_return_bars"] == 36
    assert result["policy"]["variation_rank_min"] == 0.65
    assert result["policy"]["ignition_rank_min"] == 0.80
    assert result["policy"]["hold_hours"] == 12
    assert result["clock"]["entry"] == "exact BTCUSDT 02:05 UTC open"
    assert result["clock"]["side"] == "common strict early/late return sign"


def test_hvtvir_does_not_promote_prior_variance_candidates():
    result = prereg.build()
    assert "SPVTA" in result["mechanism"]["why_distinct"]
    assert result["research_boundary"]["prior_event_sets_reused"] is False
    assert result["research_boundary"]["promoted_prior_control"] is False


def test_hvtvir_hash_binds_core():
    result = prereg.build()
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == prereg.canonical_hash(core)
