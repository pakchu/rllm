from training import preregister_options_implied_vol_flush_absorption_reversal as prereg


def test_oifar_is_singleton_outcome_blind_and_not_a_prior_candidate_repair():
    report = prereg.build()
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == prereg.canonical_hash(core)
    assert report["outcomes_opened"] is False
    assert report["research_boundary"]["candidate_count"] == 1
    assert report["research_boundary"]["grid"] is False
    assert report["research_boundary"]["repair_of_prior_candidate"] is False


def test_oifar_fades_two_sided_oi_flushes_without_funding_direction():
    report = prereg.build()
    assert report["clock"]["side"] == "opposite completed-hour return"
    assert "q25" in report["clock"]["oi"]
    assert report["clock"]["funding"].startswith("not a signal input")
    assert report["policy"]["hold_hours"] == 6
    assert report["economic_gates"]["stop_on_first_failure"] is True
