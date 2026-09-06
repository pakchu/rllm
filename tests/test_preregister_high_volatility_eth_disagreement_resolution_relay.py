from training import preregister_high_volatility_eth_disagreement_resolution_relay as prereg


def test_hvedr_is_outcome_blind_independent_singleton():
    result = prereg.build()
    assert result["policy_id"] == "HVEDR-6"
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["singleton"] is True
    assert result["research_boundary"]["candidate_count"] == 1
    assert result["research_boundary"]["grid"] is False
    assert result["research_boundary"]["repair_of_prior_candidate"] is False
    assert result["research_boundary"]["promoted_prior_control"] is False


def test_hvedr_freezes_disagreement_resolution_and_rank_history():
    result = prereg.build()
    policy = result["policy"]
    assert policy["history_observations"] == 270
    assert policy["minimum_history_observations"] == 180
    assert policy["btc_variation_rank_min"] == 0.65
    assert policy["absolute_relative_return_spread_rank_min"] == 0.80
    assert policy["hold_hours"] == 6
    assert "opposite strict nonzero signs" in result["features"]["direction_gate"]
    assert result["mechanism"]["side"].startswith("strict nonzero sign of final-two-hour ETHUSDT")


def test_hvedr_is_disjoint_from_hvelr_same_direction_control_family():
    result = prereg.build()
    assert "disjoint same-direction gate" in result["mechanism"]["why_distinct"]
    assert result["diagnostic_controls"]["diagnostic_controls_cannot_be_promoted"] is True
    assert result["research_boundary"]["prior_candidate_outcomes_used_to_set_hvedr_direction_threshold_hold_or_clock"] is False


def test_hvedr_hash_binds_core():
    result = prereg.build()
    assert result["manifest_hash"] == prereg.canonical_hash({key: value for key, value in result.items() if key != "manifest_hash"})
