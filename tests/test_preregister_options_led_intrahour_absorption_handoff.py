from training import preregister_options_led_intrahour_absorption_handoff as prereg


def test_oliah_is_singleton_outcome_blind_and_not_a_prior_repair():
    report = prereg.build()
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == prereg.canonical_hash(core)
    assert report["outcomes_opened"] is False
    assert report["research_boundary"]["candidate_count"] == 1
    assert report["research_boundary"]["grid"] is False
    assert report["research_boundary"]["repair_of_prior_candidate"] is False


def test_oliah_follows_completed_intrahour_absorption_without_funding_side():
    report = prereg.build()
    assert report["clock"]["side"] == "sign of second-half return"
    assert "crosses the hour open" in report["clock"]["absorption"]
    assert report["clock"]["funding"].startswith("not a signal input")
    assert report["policy"]["minimum_absorption_ratio"] == 0.5
    assert report["policy"]["hold_hours"] == 6
