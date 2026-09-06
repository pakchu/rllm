import hashlib
import json

from training import build_funding_settlement_volatility_ignition_breakout_relay_support as support


def test_fsvibr_source_support_is_frozen_terminal_without_outcome_access():
    assert hashlib.sha256(support.RESULT.read_bytes()).hexdigest() == (
        "06c3a5cccfa4e1f8eab8bbd518a91130e3a22543d07ec075b3f8d1037896ac26"
    )
    assert hashlib.sha256(support.CLOCK.read_bytes()).hexdigest() == (
        "865053515b50e2e3d969b00d4b7b47a1945e1ef7416f6695b635e7311dc9053b"
    )
    payload = json.loads(support.RESULT.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == support.canonical_hash(core)
    assert payload["support_passed"] is False
    assert payload["advance_to_gross9_novelty"] is False
    assert payload["advance_to_economic_outcomes"] is False
    assert payload["decision"] == "terminal_source_support_reject"
    assert payload["postentry_return_pnl_execution_price_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert [payload["support"][name]["events"] for name in ("train", "test", "eval", "final")] == [
        33,
        28,
        0,
        2,
    ]
    assert payload["support_checks"]["eval_minimum_events"] is False
    assert payload["support_checks"]["final_minimum_events"] is False
    assert all(not item["promotion_authorized"] for item in payload["controls"].values())
