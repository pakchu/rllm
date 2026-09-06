from training import preregister_debt_public_supply_liquidity_relay as prereg


def test_preregistration_is_outcome_blind_and_singleton():
    payload = prereg.payload()
    assert payload["policy_id"] == "DPSLR-24"
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["singleton"] is True
    assert payload["research_boundary"]["debt_to_the_penny_rows_opened"] is False


def test_supply_rule_and_conservative_clock_are_frozen():
    payload = prereg.payload()
    assert payload["features"]["public_supply_change"].startswith("log(current debt_held_public_amt")
    assert "rank>=0.70" in payload["features"]["magnitude_rank"]
    assert "rank>=0.65" in payload["features"]["volatility_rank"]
    assert payload["clock"]["decision"] == "exact record_date + 7 calendar days at 00:00 UTC"
    assert payload["clock"]["hold"] == "24 elapsed hours"


def test_manifest_and_terminal_gates_are_bound():
    payload = prereg.payload()
    manifest_hash = payload.pop("manifest_hash")
    assert prereg.canonical_hash(payload) == manifest_hash
    assert payload["source_support_gates"]["minimum_events"] == {
        "train": 8, "test": 12, "eval": 12, "final": 8,
    }
    assert payload["economic_gates"]["cagr_to_strict_mdd_min"] == 3.0
    assert payload["economic_gates"]["stress_cagr_to_strict_mdd_min"] == 2.5
