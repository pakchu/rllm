from training import preregister_confirmation_ladder_settlement_impact_efficiency_relay as prereg


def test_clsier_preregistration_is_outcome_blind_and_hash_bound():
    result = prereg.build()
    assert result["policy_id"] == "CLSIER-6"
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert "settlement_impact" in result["features"]
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == prereg.canonical_hash(core)
