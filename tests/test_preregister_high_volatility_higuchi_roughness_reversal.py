import json

from training import preregister_high_volatility_higuchi_roughness_reversal as p


def test_hvhrr_is_outcome_blind_singleton():
    result = p.build()
    assert result["policy_id"] == "HVHRR-8"
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert result["research_boundary"]["candidate_count"] == 1
    assert result["policy"]["higuchi_lags"] == list(range(1, 9))
    assert result["policy"]["variation_rank_min"] == 0.65
    assert result["policy"]["roughness_rank_min"] == 0.75
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == p.canonical_hash(core)
    json.dumps(result, allow_nan=False)
