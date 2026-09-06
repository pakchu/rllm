from training import preregister_high_volatility_dollar_breadth_relay as prereg


def test_hvdbr_is_outcome_blind_independent_singleton():
    result = prereg.build()
    assert result["policy_id"] == "HVDBR-12"
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["singleton"] is True
    assert result["research_boundary"]["candidate_count"] == 1
    assert result["research_boundary"]["grid"] is False
    assert result["research_boundary"]["repair_of_prior_candidate"] is False
    assert result["research_boundary"]["promoted_prior_control"] is False


def test_hvdbr_freezes_broad_dollar_shock_and_volatile_regime():
    result = prereg.build()
    policy = result["policy"]
    assert policy["fx_prior_sessions"] == 90
    assert policy["fx_prior_min_sessions"] == 60
    assert policy["minimum_agreeing_pairs"] == 5
    assert policy["median_absolute_pair_z_min"] == 1.0
    assert policy["realized_variation_rank_min"] == 0.65
    assert policy["hold_hours"] == 12
    assert len(result["source_plan"]["fx"]["symbols"]) == 6
    assert result["clock"]["side"] == "opposite common canonical dollar direction"


def test_hvdbr_does_not_promote_single_pair_or_dense_fx_control():
    result = prereg.build()
    assert "UJCVR used one USDJPY return" in result["mechanism"]["why_distinct"]
    assert result["research_boundary"]["prior_fx_event_sets_reused"] is False
    assert result["research_boundary"]["prior_fx_candidate_outcomes_used_to_set_hvdbr_direction_threshold_hold_or_clock"] is False


def test_hvdbr_hash_binds_core():
    result = prereg.build()
    assert result["manifest_hash"] == prereg.canonical_hash({key: value for key, value in result.items() if key != "manifest_hash"})
