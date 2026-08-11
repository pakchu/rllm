from training import seal_high_volatility_who_outbreak_disclosure_pressure_relay_source_rejection as s


def test_frozen_who_source_rejection():
    value = s.build()
    core = dict(value)
    manifest_hash = core.pop("manifest_hash")
    assert manifest_hash == s.canonical_hash(core)
    assert value["decision"] == "terminal_source_contract_reject"
    assert value["failed_contract"]["json_path"] == "value[0].SystemSourceKey"
    assert value["failed_contract"]["observed_value"] is None
    assert value["research_boundary"]["full_collection_incidence_opened"] is False
    assert value["research_boundary"]["btc_variation_rows_opened"] is False
    assert value["advance_to_gross9_novelty"] is False
    assert value["membership_repair_authorized"] is False
