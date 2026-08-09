from training import preregister_high_volatility_weekly_momentum_relay as prereg


def test_hvwmr_is_outcome_blind_independent_singleton():
    result = prereg.build()
    assert result["policy_id"] == "HVWMR-72"
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert result["singleton"] is True
    assert result["research_boundary"]["candidate_count"] == 1
    assert result["research_boundary"]["grid"] is False
    assert result["research_boundary"]["repair_of_prior_candidate"] is False


def test_hvwmr_freezes_weekly_high_variation_clock():
    result = prereg.build()
    assert result["policy"]["return_bars"] == 2016
    assert result["policy"]["prior_weeks"] == 52
    assert result["policy"]["prior_min_weeks"] == 26
    assert result["policy"]["variation_rank_min"] == 0.60
    assert result["policy"]["hold_hours"] == 72
    assert result["clock"]["entry"] == "exact BTCUSDT Wednesday 00:05 UTC open"


def test_hvwmr_does_not_reuse_stcr_and_keeps_rv20_late():
    result = prereg.build()
    assert "STCR" in result["mechanism"]["why_distinct"]
    assert result["research_boundary"]["prior_event_sets_reused"] is False
    assert result["research_boundary"]["prior_candidate_controls_promoted"] is False
    assert result["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False


def test_hvwmr_hash_binds_core():
    result = prereg.build(); core = {k: v for k, v in result.items() if k != "manifest_hash"}
    assert result["manifest_hash"] == prereg.canonical_hash(core)
