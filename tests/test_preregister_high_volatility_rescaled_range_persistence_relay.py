import json

from training import preregister_high_volatility_rescaled_range_persistence_relay as p


def test_hvrsh_is_outcome_blind_singleton():
    result = p.build()
    assert result["policy_id"] == "HVRSH-8"
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert result["research_boundary"]["candidate_count"] == 1
    assert result["policy"]["segment_lengths_returns"] == [30, 60, 120]
    assert result["policy"]["variation_rank_min"] == 0.65
    assert result["policy"]["hurst_rank_min"] == 0.75
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == p.canonical_hash(core)
    json.dumps(result, allow_nan=False)
