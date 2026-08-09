import json

from training import preregister_high_volatility_intrinsic_topology_ridge_relay as p


def test_hvitr_is_oos_blind_singleton():
    result = p.build()
    assert result["policy_id"] == "HVITR-8"
    assert result["oos_outcomes_opened"] is False
    assert result["oos_source_incidence_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert result["research_boundary"]["candidate_count"] == 1
    assert result["policy"]["ridge_alpha"] == 100.0
    assert result["policy"]["prediction_strength_quantile"] == 0.75
    assert tuple(result["feature_contract"]["ordered_features"]) == p.FEATURES
    assert result["research_boundary"]["repair_of_prior_candidate"] is False
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == p.canonical_hash(core)
    json.dumps(result, allow_nan=False)
