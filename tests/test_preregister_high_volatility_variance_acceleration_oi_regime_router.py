import json
from training import preregister_high_volatility_variance_acceleration_oi_regime_router as p

def test_router_is_frozen_singleton_before_combined_incidence():
    value = p.build(); p.validate(value)
    assert value["policy_id"] == "HVCVAROIR-8"
    assert value["candidate_family"] == ["HVCVAROIR-8"]
    assert value["construction"]["additional_or_tuned_thresholds"] == "none"
    assert value["research_boundary"]["combined_incidence_opened"] is False
    assert value["research_boundary"]["contraction_postentry_outcomes_known"] is False
    json.dumps(value, allow_nan=False)

def test_every_component_is_hash_pinned():
    for artifacts in p.COMPONENTS.values():
        for artifact in artifacts.values():
            assert p.sha256(artifact["path"]) == artifact["sha256"]
