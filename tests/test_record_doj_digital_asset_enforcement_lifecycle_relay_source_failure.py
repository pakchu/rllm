import json

from training import record_doj_digital_asset_enforcement_lifecycle_relay_source_failure as record


def test_terminal_failure_is_hash_bound_and_pre_market():
    payload = json.loads(record.OUTPUT.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == record.canonical_hash(core)
    assert payload["first_failed_gate"] == "official_source_date_identity"
    assert payload["failure"]["observed"]["date"] == ""
    assert payload["candidate_incidence_computed"] is False
    assert payload["btc_source_rows_opened"] == 0
    assert payload["gross9_rows_opened"] == 0
    assert payload["advance_to_economic_outcomes"] is False
