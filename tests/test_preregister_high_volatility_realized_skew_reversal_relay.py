from training import preregister_high_volatility_realized_skew_reversal_relay as prereg


def test_hvrsr_is_outcome_blind_independent_singleton():
    result = prereg.build()
    assert result["policy_id"] == "HVRSR-12"
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert result["singleton"] is True
    assert result["research_boundary"]["candidate_count"] == 1
    assert result["research_boundary"]["grid"] is False
    assert result["research_boundary"]["repair_of_prior_candidate"] is False


def test_hvrsr_freezes_third_moment_volatile_regime_and_clock():
    result = prereg.build()
    assert result["policy"]["return_bars"] == 288
    assert result["policy"]["variation_rank_min"] == 0.65
    assert result["policy"]["absolute_skew_rank_min"] == 0.70
    assert result["policy"]["hold_hours"] == 12
    assert result["clock"]["entry"] == "exact BTCUSDT 02:05 UTC open"
    assert result["clock"]["side"] == "opposite strict realized-skew sign"


def test_hvrsr_does_not_promote_semivariance_or_entropy_controls():
    result = prereg.build()
    assert "HVSIR" in result["mechanism"]["why_distinct"]
    assert result["research_boundary"]["prior_event_sets_reused"] is False
    assert result["research_boundary"]["promoted_prior_control"] is False


def test_hvrsr_hash_binds_core():
    result = prereg.build()
    assert result["manifest_hash"] == prereg.canonical_hash({key: value for key, value in result.items() if key != "manifest_hash"})
