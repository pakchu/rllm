from training import preregister_high_volatility_cross_alt_synchronized_flip_relay as prereg


def test_hvcasfr_preregistration_is_outcome_blind_and_hash_bound():
    result = prereg.build()
    assert result["policy_id"] == "HVCASFR-8"
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert result["policy"]["minimum_symbol_flips"] == 4
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == prereg.canonical_hash(core)
