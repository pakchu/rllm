import json

from training import preregister_high_volatility_directional_change_ticket_participation_close_location_router as p


def test_close_location_router_is_frozen_outcome_blind():
    result = p.build()
    assert result["policy_id"] == "HVDCSATPCLR-8"
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert result["policy"]["long_acceptance_close_location_min"] == 0.75
    assert result["policy"]["short_acceptance_close_location_max"] == 0.25
    assert result["research_boundary"]["exact_close_location_router_incidence_or_outcomes_known"] is False
    assert result["research_boundary"]["repair_of_prior_candidate"] is False
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == p.canonical_hash(core)
    json.dumps(result, allow_nan=False)
