import json

from training import preregister_high_volatility_btc_dominance_rotation_relay as prereg


def test_manifest_and_frozen_policy_are_valid():
    payload = prereg.build()
    prereg.validate(payload)
    assert payload["policy_id"] == "HVBDRR-8"
    assert payload["policy"] == {
        "prior_blocks": 270,
        "minimum_prior_blocks": 180,
        "absolute_residual_rank_min": 0.70,
        "alt_dispersion_rank_min": 0.65,
        "variation_rank_min": 0.65,
        "entry_delay_minutes": 5,
        "hold_hours": 8,
        "leverage": 0.5,
        "base_cost_per_notional_side": 0.0006,
        "stress_cost_per_notional_side": 0.001,
    }


def test_boundaries_and_controls_are_frozen_before_incidence():
    payload = prereg.build()
    assert payload["outcomes_opened"] is False
    assert payload["candidate_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert payload["diagnostic_controls"]["names"] == [
        "no_residual_tail",
        "no_dispersion_gate",
        "no_variation_gate",
        "alt_factor_direction",
        "one_block_stale_geometry",
        "direction_flip",
        "same_clock_forced_long",
    ]
    encoded = json.dumps(payload)
    assert "RV20 q90 only after all economics pass" in encoded
    assert "full-calendar CAGR" in encoded
