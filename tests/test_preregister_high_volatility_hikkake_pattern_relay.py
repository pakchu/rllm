from training import preregister_high_volatility_hikkake_pattern_relay as prereg


def test_preregistration_boundary() -> None:
    registration = prereg.build()
    assert registration["policy_id"] == "HVHIKKAKE-C3-8"
    assert registration["as_of_date"] == "2026-08-11"
    assert registration["outcomes_opened"] is False
    assert registration["source_incidence_opened"] is False
    assert registration["gross9_rows_opened"] is False
    assert registration["singleton"] is True
    assert registration["policy"]["confirmation_bars"] == 3
    assert registration["policy"]["confirmation_absolute_output"] == 2
    assert registration["policy"]["variation_rank_min"] == 0.65
    assert registration["policy"]["hold_hours"] == 8
    assert registration["research_boundary"]["candidate_count"] == 1
    assert registration["research_boundary"]["grid"] is False
    assert registration["research_boundary"]["repair_of_prior_candidate"] is False


def test_manifest_hash() -> None:
    registration = prereg.build()
    core = {key: value for key, value in registration.items() if key != "manifest_hash"}
    assert registration["manifest_hash"] == prereg.canonical_hash(core)
