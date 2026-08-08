import hashlib, json
from training import build_daily_taker_flow_acceleration_relay_support as support


def test_dtfar_source_support_is_frozen_pass_without_outcomes():
    assert hashlib.sha256(support.RESULT.read_bytes()).hexdigest() == "dd5bab5b2061d36d0535a0463fb7a593308b3a14c7a79df2fc2eed955e83b795"
    assert hashlib.sha256(support.CLOCK.read_bytes()).hexdigest() == "0d323a439d2acf9974b4c2d3a152b5fc252a2f367db1872b017eecb1d4511411"
    payload = json.loads(support.RESULT.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == support.canonical_hash(core)
    assert payload["support_passed"] is True
    assert payload["advance_to_gross9_novelty"] is True
    assert payload["advance_to_economic_outcomes"] is False
    assert payload["decision"] == "pass_to_novelty"
    assert payload["postentry_return_pnl_execution_price_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert [payload["support"][split]["events"] for split in support.SPLITS] == [14, 37, 34, 22]
    assert all(not control["promotion_authorized"] for control in payload["controls"].values())
