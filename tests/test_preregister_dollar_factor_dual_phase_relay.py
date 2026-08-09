import json

from training import preregister_dollar_factor_dual_phase_relay as prereg


def test_manifest_and_policy_are_frozen():
    payload = prereg.build()
    prereg.validate(payload)
    assert payload["policy_id"] == "DFDPR-12"
    assert payload["policy"]["persistence_rank_min"] == 0.60
    assert payload["policy"]["variation_rank_min"] == 0.65
    assert payload["clock"]["hold"] == "12 elapsed hours"


def test_outcome_boundary_and_controls_are_frozen():
    payload = prereg.build()
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert payload["diagnostic_controls"]["names"] == [
        "no_persistence_rank",
        "no_variation_gate",
        "early_factor_only",
        "late_factor_only",
        "one_session_stale_phases",
        "direction_flip",
        "same_clock_forced_long",
    ]
    encoded = json.dumps(payload)
    assert "full-calendar CAGR" in encoded
    assert "RV20 q90 only after all economics pass" in encoded
