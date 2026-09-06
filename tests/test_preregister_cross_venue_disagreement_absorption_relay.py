from training import preregister_cross_venue_disagreement_absorption_relay as prereg


def test_cvdar_is_singleton_outcome_blind_and_distinct():
    report = prereg.build(); core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == prereg.canonical_hash(core)
    assert report["outcomes_opened"] is False
    assert report["research_boundary"]["candidate_count"] == 1
    assert report["research_boundary"]["grid"] is False
    assert report["research_boundary"]["repair_of_prior_candidate"] is False
    assert report["clock"]["side"] == "sign of second-half return"
    assert "opposite" in report["clock"]["volatility"]
