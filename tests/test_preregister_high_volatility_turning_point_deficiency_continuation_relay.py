import json

from training import preregister_high_volatility_turning_point_deficiency_continuation_relay as p


def test_hvtpdcr_is_outcome_blind_singleton():
    result = p.build()
    assert result["policy_id"] == "HVTPDCR-8"
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert result["research_boundary"]["candidate_count"] == 1
    assert result["research_boundary"]["repair_of_prior_candidate"] is False
    assert result["policy"]["five_minute_observations"] == 96
    assert result["policy"]["interior_observations"] == 94
    assert result["policy"]["turning_point_rank_max"] == 0.20
    assert result["policy"]["variation_rank_min"] == 0.65
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == p.canonical_hash(core)
    p.validate(result)
    json.dumps(result, allow_nan=False)


def test_hvtpdcr_fixed_gates_and_calendar():
    result = p.build()
    assert result["source_support_gates"]["minimum_events"] == {
        "train": 8, "test": 12, "eval": 12, "final": 8
    }
    assert result["novelty_gates"]["exact_entry_jaccard_max"] == 0.10
    assert result["economic_gates"]["cagr_to_strict_mdd_min"] == 3.0
    assert result["clock"]["gross_exposure"] == 0.5
    assert result["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False
