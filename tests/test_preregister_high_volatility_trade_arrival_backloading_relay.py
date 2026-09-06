import json

from training import preregister_high_volatility_trade_arrival_backloading_relay as prereg


def test_preregistration_is_outcome_blind_and_singleton():
    result = prereg.build()
    assert result["policy_id"] == "HVTAB-6"
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert result["research_boundary"]["candidate_count"] == 1
    assert result["research_boundary"]["grid"] is False
    assert result["policy"]["late_share_rank_min"] == 0.75
    assert result["policy"]["variation_rank_min"] == 0.65
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == prereg.canonical_hash(core)
    json.dumps(result, allow_nan=False)
