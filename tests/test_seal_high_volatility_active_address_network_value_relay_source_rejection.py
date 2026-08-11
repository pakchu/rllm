from training import seal_high_volatility_active_address_network_value_relay_source_rejection as s


def test_frozen_terminal_source_rejection():
    value = s.build()
    core = dict(value)
    manifest = core.pop("manifest_hash")
    assert manifest == s.canonical_hash(core)
    assert value["decision"] == "terminal_source_contract_reject"
    assert value["failed_contract"]["first_offending_response_index_zero_based"] == 214
    assert value["failed_contract"]["lateness_seconds"] == 299
    assert value["failed_contract"]["first_failure_short_circuit"] is True
    assert value["research_boundary"]["candidate_clock_or_support_incidence_computed"] is False
    assert value["research_boundary"]["btc_variation_rows_opened"] is False
    assert value["advance_to_gross9_novelty"] is False
    assert value["completion_window_repair_authorized"] is False
    assert value["repair_authorized"] is False
