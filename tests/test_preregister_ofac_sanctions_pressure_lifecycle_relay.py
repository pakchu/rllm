from training import preregister_ofac_sanctions_pressure_lifecycle_relay as prereg

def test_hash_and_blind_boundary():
    payload=prereg.build();core={k:v for k,v in payload.items() if k!="manifest_hash"}
    assert payload["manifest_hash"]==prereg.canonical_hash(core)
    assert payload["outcomes_opened"] is False
    assert payload["research_boundary"]["ofac_action_values_or_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False

def test_fixed_contract():
    payload=prereg.build()
    assert "designation" in payload["taxonomy"]["pressure_terms"]
    assert "general license" in payload["taxonomy"]["relief_terms"]
    assert payload["clock"]["hold"]=="24 elapsed hours"
    assert payload["source_support_gates"]["minimum_events"]=={"train":8,"test":12,"eval":12,"final":8}
