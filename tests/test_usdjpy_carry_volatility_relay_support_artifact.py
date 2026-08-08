import hashlib, json
from training import build_usdjpy_carry_volatility_relay_support as support


def test_ujcvr_support_artifact_is_terminal_before_outcomes():
    assert hashlib.sha256(support.RESULT.read_bytes()).hexdigest() == "a9ea4e91552c99c4c40c439d19488f55179be240645b1381f5f48bfec09e5d2f"
    payload = json.loads(support.RESULT.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == support.chash(core)
    assert payload["decision"] == "terminal_source_support_reject"
    assert payload["support_passed"] is False
    assert payload["advance_to_gross9_novelty"] is False
    assert payload["advance_to_economic_outcomes"] is False
    assert payload["postentry_return_pnl_execution_price_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert [payload["support"][split]["events"] for split in support.SPLITS] == [9, 18, 30, 12]
    assert payload["support"]["train"]["max_month_share"] > 0.45
    assert all(not item["promotion_authorized"] for item in payload["controls"].values())
