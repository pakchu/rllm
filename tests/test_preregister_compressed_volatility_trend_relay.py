from training import preregister_compressed_volatility_trend_relay as prereg


def test_cvtr_is_singleton_outcome_blind_and_distinct():
    report = prereg.build()
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == prereg.canonical_hash(core)
    assert report["outcomes_opened"] is False
    assert report["research_boundary"]["candidate_count"] == 1
    assert report["research_boundary"]["grid"] is False
    assert report["research_boundary"]["repair_of_prior_candidate"] is False
    assert report["clock"]["volatility"].endswith("strictly negative")
    assert report["clock"]["side"] == "common sign of the two completed-hour returns"
