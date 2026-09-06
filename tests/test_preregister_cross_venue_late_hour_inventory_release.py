from training import preregister_cross_venue_late_hour_inventory_release as prereg


def test_cvlir_is_singleton_outcome_blind_and_not_a_prior_repair():
    report = prereg.build()
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == prereg.canonical_hash(core)
    assert report["outcomes_opened"] is False
    assert report["research_boundary"]["candidate_count"] == 1
    assert report["research_boundary"]["grid"] is False
    assert report["research_boundary"]["repair_of_prior_candidate"] is False


def test_cvlir_is_quiet_to_tail_inventory_release():
    report = prereg.build()
    assert "q40" in report["clock"]["quiet_first_half"]
    assert "q90" in report["clock"]["late_breakout"]
    assert "absolute" in report["clock"]["oi"]
    assert report["clock"]["side"] == "sign of second-half return"
