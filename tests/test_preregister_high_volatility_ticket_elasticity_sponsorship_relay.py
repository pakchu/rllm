from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as p


def test_hvter_boundary_and_policy():
    result = p.build()
    assert result["policy_id"] == "HVTER-8"
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert result["singleton"] is True
    assert result["research_boundary"]["candidate_count"] == 1
    assert result["research_boundary"]["grid"] is False
    assert result["research_boundary"]["repair_of_prior_candidate"] is False
    assert result["policy"]["elasticity_floor"] == 1.0
    assert result["policy"]["elasticity_rank_min"] == 0.75
    assert result["policy"]["variation_rank_min"] == 0.65
    assert result["policy"]["hold_hours"] == 8


def test_hvter_hash():
    result = p.build()
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == p.canonical_hash(core)
