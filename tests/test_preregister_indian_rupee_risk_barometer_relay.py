import json

from training import preregister_indian_rupee_risk_barometer_relay as prereg


def test_irbr_is_outcome_blind_singleton_with_frozen_clock():
    result = prereg.build()
    prereg.validate(result)
    assert result["policy_id"] == "IRBR-12"
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert result["features"]["fx_session"] == (
        "bars_polygon USDINR 1m rows [03:45,10:00) UTC on each Monday-Friday"
    )
    assert "rank>=0.70" in result["features"]["absolute_fx_return_rank"]
    assert result["clock"]["entry"] == "exact BTCUSDT 10:05 UTC 5m open"
    assert result["clock"]["side"] == "-sign(fx_return)"
    assert result["clock"]["hold"] == "12 elapsed hours"
    assert result["research_boundary"]["candidate_count"] == 1
    assert result["research_boundary"]["grid"] is False
    assert result["research_boundary"]["repair_of_prior_candidate"] is False
    json.dumps(result, allow_nan=False)
