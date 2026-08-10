import json

from training import preregister_spot_perpetual_intraday_desynchronization_relay as p


def test_spidr_is_outcome_blind_singleton():
    result = p.build()
    assert result["policy_id"] == "SPIDR-8"
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert result["research_boundary"]["candidate_count"] == 1
    assert result["policy"]["history_blocks"] == 270
    assert result["policy"]["variation_rank_min"] == 0.65
    assert result["policy"]["desynchronization_rank_min"] == 0.75
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == p.canonical_hash(core)
    json.dumps(result, allow_nan=False)
