import json

from training import preregister_fear_greed_persistence_diffusion_relay as prereg


def test_preregistration_is_singleton_and_outcome_blind():
    payload = prereg.build()
    assert payload["policy_id"] == "FGPDR-24"
    assert payload["singleton"] is True
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["research_boundary"]["grid"] is False
    assert payload["research_boundary"]["repair_of_prior_candidate"] is False


def test_policy_and_terminal_gates_are_frozen():
    payload = prereg.build()
    assert payload["policy"]["persistence_days"] == 3
    assert payload["policy"]["persistence_magnitude_rank_min"] == 0.60
    assert payload["policy"]["realized_variation_rank_min"] == 0.65
    assert payload["policy"]["hold_hours"] == 24
    assert payload["features"]["btc_direction_forbidden"] is True
    assert payload["economic_gates"]["cagr_to_strict_mdd_min"] == 3.0
    assert payload["economic_gates"]["stress_cagr_to_strict_mdd_min"] == 2.5
    assert payload["economic_gates"]["each_calendar_half_positive"] is True


def test_manifest_hash_replays():
    payload = prereg.build()
    manifest_hash = payload.pop("manifest_hash")
    assert prereg.canonical_hash(payload) == manifest_hash
    assert json.dumps(payload, allow_nan=False)
