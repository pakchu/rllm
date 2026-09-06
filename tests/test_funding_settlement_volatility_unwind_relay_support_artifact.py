import hashlib
import json

from training import build_funding_settlement_volatility_unwind_relay_support as support


def test_fsvur_source_support_is_frozen_terminal():
    assert hashlib.sha256(support.RESULT.read_bytes()).hexdigest() == (
        "c09885dd2b5fe0a4cb10dbedfa8494177616f3302f75703b04b122ae035fd71e"
    )
    assert hashlib.sha256(support.CLOCK.read_bytes()).hexdigest() == (
        "787640f39f90a25c05cf312d0a11437a99a0b82c38f7045f4e82d47d87caebe1"
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

    for split in ("train", "test", "eval"):
        assert payload["support"][split]["longs"] == 0
        assert payload["support"][split]["minority_side_share"] == 0
    assert payload["support"]["train"]["max_month_share"] > 0.45
