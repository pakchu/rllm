from training import seal_high_volatility_uniswap_wbtc_liquidity_shock_relay_source_rejection as s


def test_frozen_terminal_source_rejection():
    value = s.build()
    core = dict(value)
    manifest_hash = core.pop("manifest_hash")
    assert manifest_hash == s.canonical_hash(core)
    assert value["decision"] == "terminal_source_contract_reject"
    assert value["failed_contract"]["transport_role"] == "verification"
    assert value["failed_contract"]["method"] == "eth_chainId"
    assert value["failed_contract"]["request"]["id"] == 1
    assert value["failed_contract"]["response"]["id"] is None
    assert value["failed_contract"]["response"]["error"]["code"] == -32001
    assert value["failed_contract"]["first_failure_short_circuit"] is True
    assert value["research_boundary"]["historical_uniswap_swap_logs_opened"] is False
    assert value["research_boundary"]["candidate_source_incidence_opened"] is False
    assert value["research_boundary"]["btc_variation_rows_opened"] is False
    assert value["advance_to_gross9_novelty"] is False
    assert value["repair_authorized"] is False


def test_transport_corrections_did_not_change_source_semantics():
    value = s.build()
    assert value["bounded_transport_history"]["semantic_or_source_identity_change"] is False
    assert value["bounded_transport_history"]["historical_incidence_used_for_correction"] is False
    assert value["replacement_provider_authorized"] is False
