from training import seal_high_volatility_ethereum_staking_deposit_pressure_relay_source_rejection as s


def test_frozen_terminal_source_rejection():
    value = s.build()
    core = dict(value)
    manifest_hash = core.pop("manifest_hash")
    assert manifest_hash == s.canonical_hash(core)
    assert value["decision"] == "terminal_source_contract_reject"
    assert value["failed_contract"]["http_status"] == 408
    assert value["failed_contract"]["failure_recurred_after_all_retries"] is True
    assert value["research_boundary"]["historical_deposit_logs_opened"] is False
    assert value["research_boundary"]["btc_variation_rows_opened"] is False
    assert value["advance_to_gross9_novelty"] is False
    assert value["repair_authorized"] is False
