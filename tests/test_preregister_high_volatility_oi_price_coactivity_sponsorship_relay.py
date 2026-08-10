from training import (
    preregister_high_volatility_oi_price_coactivity_sponsorship_relay as prereg,
)


def test_preregistration_is_singleton_outcome_blind_and_hash_bound() -> None:
    payload = prereg.build()
    prereg.validate(payload)
    assert payload["policy_id"] == "HVOIPCSR-8"
    assert payload["singleton"] is True
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert payload["research_boundary"]["candidate_count"] == 1
    assert payload["research_boundary"]["grid"] is False
    assert payload["research_boundary"]["repair_of_prior_candidate"] is False


def test_preregistration_freezes_coactivity_and_strict_gates() -> None:
    payload = prereg.build()
    policy = payload["policy"]
    assert policy["block_hours"] == 3
    assert policy["price_bars"] == 36
    assert policy["oi_points"] == 37
    assert policy["coactivity_rank_min"] == 0.75
    assert policy["gross_oi_activity_rank_min"] == 0.60
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
