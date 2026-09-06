from training import preregister_high_volatility_alt_leader_concentration_relay as p


def test_manifest_is_canonical_outcome_blind_and_singleton():
    payload = p.build()
    p.validate(payload)
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == p.canonical_hash(core)
    assert payload["policy_id"] == "HVALCR-8"
    assert payload["singleton"] is True
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert payload["policy"]["concentration_rank_min"] == 0.75
    assert payload["diagnostic_controls"]["cannot_be_promoted"] is True


def test_stopping_rule_forbids_hhi_and_direction_repairs():
    rule = p.build()["stopping_rule"]
    assert "no universe" in rule
    assert "HHI" in rule
    assert "side" in rule
