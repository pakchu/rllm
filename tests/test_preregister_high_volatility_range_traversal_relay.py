from training import preregister_high_volatility_range_traversal_relay as prereg

def test_preregistration_is_hash_valid_and_frozen() -> None:
    result = prereg.build(); prereg.validate(result)
    assert result["policy_id"] == "HVRTR-12"
    assert result["source_incidence_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert result["policy"]["range_rank_min"] == 0.65
    assert result["policy"]["upper_close_location_min"] == 0.75
    assert result["clock"]["entry"] == "exact BTCUSDT D 00:05 UTC open"
    assert result["clock"]["hold"] == "12 elapsed hours"

def test_singleton_controls_cannot_be_promoted() -> None:
    result = prereg.build()
    assert result["singleton"] is True
    assert result["research_boundary"]["candidate_count"] == 1
    assert result["research_boundary"]["grid"] is False
    assert result["diagnostic_controls"]["cannot_be_promoted"] is True
