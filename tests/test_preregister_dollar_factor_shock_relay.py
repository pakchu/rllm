from training import preregister_dollar_factor_shock_relay as prereg


def test_dfsr_is_outcome_blind_independent_singleton():
    result = prereg.build()
    assert result["policy_id"] == "DFSR-12"
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["singleton"] is True
    assert result["research_boundary"]["candidate_count"] == 1
    assert result["research_boundary"]["grid"] is False
    assert result["research_boundary"]["repair_of_prior_candidate"] is False
    assert result["research_boundary"]["promoted_prior_control"] is False


def test_dfsr_freezes_continuous_dollar_factor_and_volatile_regime():
    result = prereg.build()
    policy = result["policy"]
    assert policy["fx_prior_sessions"] == 90
    assert policy["fx_prior_min_sessions"] == 60
    assert policy["factor_absolute_rank_min"] == 0.70
    assert policy["realized_variation_rank_min"] == 0.65
    assert policy["hold_hours"] == 12
    assert len(result["source_plan"]["fx"]["symbols"]) == 6
    assert result["clock"]["side"] == "opposite strict dollar-factor sign"


def test_dfsr_does_not_promote_prior_fx_control():
    result = prereg.build()
    assert "HVDBR-12" in result["mechanism"]["why_distinct"]
    assert result["research_boundary"]["prior_fx_event_sets_reused"] is False
    assert result["research_boundary"]["prior_fx_candidate_outcomes_used_to_set_dfsr_factor_direction_rank_hold_or_clock"] is False


def test_dfsr_hash_binds_core():
    result = prereg.build()
    assert result["manifest_hash"] == prereg.canonical_hash({key: value for key, value in result.items() if key != "manifest_hash"})
