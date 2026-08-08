from training import preregister_high_volatility_passive_absorption_relay as prereg


def test_hvpar_is_outcome_blind_independent_singleton():
    result = prereg.build()
    assert result["policy_id"] == "HVPAR-6"
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["singleton"] is True
    assert result["research_boundary"]["candidate_count"] == 1
    assert result["research_boundary"]["grid"] is False
    assert result["research_boundary"]["repair_of_prior_candidate"] is False
    assert result["research_boundary"]["promoted_prior_control"] is False


def test_hvpar_freezes_price_flow_contradiction_and_gates():
    result = prereg.build()
    policy = result["policy"]
    assert policy["history_observations"] == 270
    assert policy["minimum_history_observations"] == 180
    assert policy["variation_rank_min"] == 0.65
    assert policy["absolute_late_taker_imbalance_min"] == 0.1
    assert policy["hold_hours"] == 6
    assert "opposes late_return" in result["features"]["absorption_gate"]
    assert result["source_plan"]["btc_1m"]["columns"][-1] == "taker_buy_quote"
    assert result["economic_gates"]["cagr_to_strict_mdd_min"] == 3.0


def test_hvpar_does_not_promote_hvafc_direction_fade():
    result = prereg.build()
    assert "merely reversed the trade side" in result["mechanism"]["why_distinct"]
    assert result["diagnostic_controls"]["diagnostic_controls_cannot_be_promoted"] is True
    assert result["research_boundary"]["prior_candidate_outcomes_used_to_set_hvpar_direction_threshold_hold_or_clock"] is False


def test_hvpar_hash_binds_core():
    result = prereg.build()
    assert result["manifest_hash"] == prereg.canonical_hash({key: value for key, value in result.items() if key != "manifest_hash"})
