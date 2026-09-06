from training import preregister_fx_volatility_sponsored_momentum_relay as prereg


def test_fvsmr_is_outcome_blind_independent_singleton():
    result = prereg.build()
    assert result["policy_id"] == "FVSMR-12"
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["singleton"] is True
    assert result["research_boundary"]["candidate_count"] == 1
    assert result["research_boundary"]["grid"] is False
    assert result["research_boundary"]["repair_of_prior_candidate"] is False
    assert result["research_boundary"]["promoted_prior_control"] is False


def test_fvsmr_freezes_fx_shock_sponsorship_and_btc_momentum_side():
    result = prereg.build()
    policy = result["policy"]
    assert policy["fx_prior_sessions"] == 90
    assert policy["fx_prior_min_sessions"] == 60
    assert policy["fx_shock_rank_min"] == 0.70
    assert policy["realized_variation_rank_min"] == 0.65
    assert policy["hold_hours"] == 12
    assert len(result["source_plan"]["fx"]["symbols"]) == 6
    assert result["clock"]["side"] == "strict sign of completed BTC 13:00-21:00 UTC session return"
    assert "median of the absolute values" in result["features"]["fx_shock"]
    assert "all eight hours" in result["features"]["btc_session_return"]


def test_fvsmr_does_not_promote_prior_fx_control():
    result = prereg.build()
    assert "DFSR-12" in result["mechanism"]["why_distinct"]
    assert result["research_boundary"]["prior_fx_event_sets_reused"] is False
    assert result["research_boundary"]["prior_fx_candidate_outcomes_used_to_set_fvsmr_shock_rank_btc_side_hold_or_clock"] is False


def test_fvsmr_ignores_fx_direction_and_uses_frozen_controls():
    result = prereg.build()
    assert "signs are never canonicalized" in result["features"]["fx_pair_returns"]
    assert result["diagnostic_controls"]["names"] == [
        "no_volatility_gate",
        "no_fx_shock_tail",
        "raw_fx_absolute_return_shock",
        "one_session_stale_fx_shock",
        "direction_flip",
    ]


def test_fvsmr_hash_binds_core():
    result = prereg.build()
    assert result["manifest_hash"] == prereg.canonical_hash({key: value for key, value in result.items() if key != "manifest_hash"})
