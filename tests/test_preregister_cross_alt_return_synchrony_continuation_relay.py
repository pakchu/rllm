import json

from training import preregister_cross_alt_return_synchrony_continuation_relay as p


def test_carsc_is_outcome_blind_singleton():
    result = p.build()
    p.validate(result)
    assert result["policy_id"] == "CARSC-8"
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert result["policy"]["synchrony_rank_min"] == 0.70
    assert result["policy"]["variation_rank_min"] == 0.65
    assert result["research_boundary"]["candidate_count"] == 1
    json.dumps(result, allow_nan=False)
