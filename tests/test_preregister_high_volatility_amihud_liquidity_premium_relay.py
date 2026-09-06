from training import preregister_high_volatility_amihud_liquidity_premium_relay as prereg


def test_boundary():
    candidate = prereg.build()
    assert candidate["policy_id"] == "HVALPR-24"
    assert candidate["as_of_date"] == "2026-08-11"
    assert candidate["outcomes_opened"] is False
    assert candidate["source_incidence_opened"] is False
    assert candidate["gross9_rows_opened"] is False
    assert candidate["singleton"] is True
    assert candidate["research_boundary"]["candidate_count"] == 1
    assert candidate["research_boundary"]["grid"] is False
    assert candidate["research_boundary"]["repair_of_prior_candidate"] is False
    assert candidate["policy"]["reference_days"] == 30
    assert candidate["policy"]["hold_hours"] == 24


def test_hash():
    candidate = prereg.build()
    core = {key: value for key, value in candidate.items() if key != "manifest_hash"}
    assert candidate["manifest_hash"] == prereg.canonical_hash(core)
