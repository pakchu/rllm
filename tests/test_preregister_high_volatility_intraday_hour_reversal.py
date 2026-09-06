from training import preregister_high_volatility_intraday_hour_reversal as prereg


def test_boundary_is_singleton_and_outcome_blind():
    candidate = prereg.build()
    assert candidate["policy_id"] == "HVIHR-1"
    assert candidate["as_of_date"] == "2026-08-11"
    assert candidate["outcomes_opened"] is False
    assert candidate["source_incidence_opened"] is False
    assert candidate["gross9_rows_opened"] is False
    assert candidate["singleton"] is True
    assert candidate["research_boundary"]["candidate_count"] == 1
    assert candidate["research_boundary"]["grid"] is False
    assert candidate["research_boundary"]["repair_of_prior_candidate"] is False
    assert candidate["policy"]["predictor_magnitude_rank_min"] == 0.70
    assert candidate["policy"]["hold_minutes"] == 55


def test_manifest_hash():
    candidate = prereg.build()
    core = {key: value for key, value in candidate.items() if key != "manifest_hash"}
    assert candidate["manifest_hash"] == prereg.canonical_hash(core)
