from training import preregister_high_volatility_oi_lead_sponsorship_relay as prereg


def test_preregistration_is_hash_valid_and_outcome_blind() -> None:
    result = prereg.build(); prereg.validate(result)
    assert result["policy_id"] == "HVOILSR-12"
    assert result["source_incidence_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert result["features"]["causal_ranks"].startswith("strict-prior")
    assert result["policy"]["lead_pairs"] == 287
    assert result["clock"]["entry"] == "exact BTCUSDT perpetual D 00:05 UTC open"
    assert result["clock"]["hold"] == "12 elapsed hours"
    assert result["economic_gates"]["cagr_to_strict_mdd_min"] == 3.0


def test_singleton_and_no_control_promotion() -> None:
    result = prereg.build()
    assert result["singleton"] is True
    assert result["research_boundary"]["candidate_count"] == 1
    assert result["research_boundary"]["grid"] is False
    assert result["diagnostic_controls"]["cannot_be_promoted"] is True
