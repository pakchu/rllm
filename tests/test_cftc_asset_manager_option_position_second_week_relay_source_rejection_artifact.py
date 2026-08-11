import hashlib
import json
from pathlib import Path


RESULT = Path("results/cftc_asset_manager_option_position_second_week_relay_support_2026-08-11.json")


def test_frozen_source_rejection_artifact():
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == "305aec824987c79b1ad0bd6595fb18e63c28535f755c02c0d4fec8c6296e0990"
    payload = json.loads(RESULT.read_text())
    assert payload["decision"] == "terminal_source_support_reject"
    assert payload["support_passed"] is False
    assert payload["advance_to_gross9_novelty"] is False
    assert payload["advance_to_economic_outcomes"] is False
    assert payload["postentry_return_pnl_execution_price_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert [payload["support"][name]["events"] for name in ("train", "test", "eval", "final")] == [8, 30, 21, 7]
    assert payload["support_checks"]["final_minimum_events"] is False
    assert payload["support_checks"]["final_month_concentration"] is False
