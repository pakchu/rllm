from training import preregister_options_loaded_quiet_price_breakout as prereg


def test_olqpb_is_singleton_outcome_blind_and_not_a_prior_repair():
    report = prereg.build()
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == prereg.canonical_hash(core)
    assert report["outcomes_opened"] is False
    assert report["research_boundary"]["candidate_count"] == 1
    assert report["research_boundary"]["grid"] is False
    assert report["research_boundary"]["repair_of_prior_candidate"] is False


def test_olqpb_is_high_iv_quiet_price_continuation():
    report = prereg.build()
    assert report["clock"]["side"] == "sign of completed-hour return"
    assert "q75" in report["clock"]["volatility"]
    assert "q40" in report["clock"]["price"]
    assert "q50" in report["clock"]["funding"]
    assert report["policy"]["hold_hours"] == 6
