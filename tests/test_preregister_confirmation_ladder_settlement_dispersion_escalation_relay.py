from training import preregister_confirmation_ladder_settlement_dispersion_escalation_relay as module


def test_manifest_and_frozen_boundary():
    payload = module.build()
    module.validate(payload)
    assert payload["policy_id"] == "CLSDER-6"
    assert payload["source_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert payload["research_boundary"]["grid"] is False
    assert payload["features"]["dispersion_escalation"].startswith("late_dispersion>")
    assert payload["source_plan"]["blocks"]["sha256"] == "e62fb5bc98a49e819ca43d8a8b0529a901f07a1ef1d07fe9ae3beb4d5f3585e8"


def test_thresholds_and_canonical_unicode_hashing():
    payload = module.build()
    assert payload["source_support_gates"]["minimum_events"] == {"train": 8, "test": 12, "eval": 12, "final": 8}
    assert payload["novelty_gates"]["candidate_near_6h_share_max"] == 0.35
    assert payload["economic_gates"]["cagr_to_strict_mdd_min"] == 3.0
    assert module.canonical_hash({"한글": "알파"}) == module.canonical_hash({"한글": "알파"})
