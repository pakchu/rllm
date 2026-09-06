import json

from training import preregister_high_volatility_return_curvature_scarcity_relay as p


def test_hvcsr_is_outcome_blind_singleton():
    result = p.build()
    assert result["policy_id"] == "HVCSR-8"
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert result["research_boundary"]["candidate_count"] == 1
    assert result["policy"]["close_observations"] == 480
    assert result["policy"]["variation_rank_min"] == 0.65
    assert result["policy"]["curvature_rank_max"] == 0.25
    assert result["research_boundary"]["repair_of_prior_candidate"] is False
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == p.canonical_hash(core)
    json.dumps(result, allow_nan=False)
