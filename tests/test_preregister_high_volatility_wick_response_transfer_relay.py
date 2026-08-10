from training import preregister_high_volatility_wick_response_transfer_relay as p


def test_hvwrtr_boundary_and_policy():
    result = p.build()
    assert result["policy_id"] == "HVWRTR-8"
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert result["singleton"] is True
    assert result["research_boundary"]["candidate_count"] == 1
    assert result["research_boundary"]["grid"] is False
    assert result["research_boundary"]["repair_of_prior_candidate"] is False
    assert result["policy"]["response_rank_min"] == 0.8
    assert result["policy"]["variation_rank_min"] == 0.65
    assert result["policy"]["hold_hours"] == 8


def test_hvwrtr_hash():
    result = p.build()
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == p.canonical_hash(core)
