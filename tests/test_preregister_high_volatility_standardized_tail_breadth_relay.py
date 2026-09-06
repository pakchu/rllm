from training import preregister_high_volatility_standardized_tail_breadth_relay as prereg


def test_preregistration_is_singleton_outcome_blind_and_hash_bound() -> None:
    payload = prereg.build()
    prereg.validate(payload)
    assert payload["policy_id"] == "HVSTBR-8"
    assert payload["singleton"] is True
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert payload["research_boundary"]["candidate_count"] == 1
    assert payload["research_boundary"]["grid"] is False
    assert payload["research_boundary"]["repair_of_prior_candidate"] is False


def test_preregistration_freezes_tail_breadth_and_strict_gates() -> None:
    payload = prereg.build()
    policy = payload["policy"]
    assert policy["close_observations"] == 480
    assert policy["return_observations"] == 479
    assert policy["tail_scale_multiple"] == 1.0
    assert policy["tail_breadth_share_rank_min"] == 0.75
    assert policy["variation_rank_min"] == 0.65
    assert policy["entry_delay_minutes"] == 5
    assert policy["hold_hours"] == 8
    assert payload["source_support_gates"]["minimum_events"] == {
        "train": 8,
        "test": 12,
        "eval": 12,
        "final": 8,
    }
    assert payload["economic_gates"]["cagr_to_strict_mdd_min"] == 3.0
    assert payload["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False
