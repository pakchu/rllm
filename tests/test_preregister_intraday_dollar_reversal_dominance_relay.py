from training import preregister_intraday_dollar_reversal_dominance_relay as prereg


def test_idrdr_is_outcome_blind_independent_singleton():
    result = prereg.build()
    assert result["policy_id"] == "IDRDR-12"
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert result["singleton"] is True
    assert result["research_boundary"]["candidate_count"] == 1
    assert result["research_boundary"]["grid"] is False
    assert result["research_boundary"]["repair_of_prior_candidate"] is False


def test_idrdr_freezes_reversal_dominance_and_clock():
    result = prereg.build()
    assert result["policy"]["fx_prior_sessions"] == 90
    assert result["policy"]["fx_prior_min_sessions"] == 60
    assert result["policy"]["late_to_early_absolute_factor_min"] == 1.0
    assert result["policy"]["realized_variation_rank_min"] == 0.65
    assert result["policy"]["hold_hours"] == 12
    assert result["clock"]["entry"] == "exact BTCUSDT D 21:05 UTC open"


def test_idrdr_keeps_rv20_late_and_does_not_reuse_full_session_factor():
    result = prereg.build()
    assert result["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False
    assert "DFSR" in result["mechanism"]["why_distinct"]
    assert result["research_boundary"]["prior_fx_event_sets_reused"] is False
    assert result["research_boundary"]["promoted_prior_control"] is False


def test_idrdr_hash_binds_core():
    result = prereg.build(); core = {k: v for k, v in result.items() if k != "manifest_hash"}
    assert result["manifest_hash"] == prereg.canonical_hash(core)
