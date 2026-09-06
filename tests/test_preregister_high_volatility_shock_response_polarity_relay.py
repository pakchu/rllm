import json

from training import preregister_high_volatility_shock_response_polarity_relay as p


def test_hvsrp_is_outcome_blind_singleton():
    result = p.build()
    assert result["policy_id"] == "HVSRP-8"
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert result["research_boundary"]["candidate_count"] == 1
    assert result["policy"]["history_blocks"] == 270
    assert result["policy"]["variation_rank_min"] == 0.65
    assert result["policy"]["response_strength_rank_min"] == 0.75
    assert result["features"]["response_correlation"] == (
        "Pearson correlation of abs(return[t]) and return[t+1] across 479 ordered pairs; "
        "finite positive variance required"
    )
    assert result["research_boundary"]["prior_hvdfr_outcome_known"] is True
    assert result["research_boundary"]["repair_of_prior_candidate"] is False
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == p.canonical_hash(core)
    json.dumps(result, allow_nan=False)
