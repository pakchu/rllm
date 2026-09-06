from training import preregister_doj_digital_asset_enforcement_lifecycle_relay as prereg


def test_preregistration_is_hash_bound_and_blind():
    payload = prereg.build()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == prereg.canonical_hash(core)
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert payload["research_boundary"]["doj_press_release_values_or_incidence_opened"] is False


def test_taxonomy_and_gates_are_fixed():
    payload = prereg.build()
    assert "bitcoin" in payload["taxonomy"]["digital_asset_terms"]
    assert "charged" in payload["taxonomy"]["initiation_terms"]
    assert "sentenced" in payload["taxonomy"]["resolution_terms"]
    assert payload["clock"]["hold"] == "24 elapsed hours"
    assert payload["source_support_gates"]["minimum_events"] == {
        "train": 8, "test": 12, "eval": 12, "final": 8
    }
