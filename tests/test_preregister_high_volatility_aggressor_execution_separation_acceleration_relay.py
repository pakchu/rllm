from training import preregister_high_volatility_aggressor_execution_separation_acceleration_relay as prereg


def test_hvaesar_preregistration_is_outcome_blind_and_hash_bound():
    report = prereg.build()
    assert report["policy_id"] == "HVAESAR-8"
    assert report["outcomes_opened"] is False
    assert report["source_incidence_opened"] is False
    assert report["gross9_rows_opened"] is False
    assert report["policy"]["minimum_acceleration_ratio"] == 1.0
    assert report["policy"]["variation_rank_min"] == 0.65
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == prereg.canonical_hash(core)
