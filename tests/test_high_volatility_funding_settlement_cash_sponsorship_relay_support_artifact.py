import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/high_volatility_funding_settlement_cash_sponsorship_relay_"
    "support_2026-08-13.json"
)


def test_source_artifact_is_terminal_before_novelty_or_outcomes():
    payload = json.loads(RESULT.read_text())
    assert payload["support_passed"] is False
    assert payload["advance_to_gross9_novelty"] is False
    assert payload["advance_to_economic_outcomes"] is False
    assert all(value["events"] == 0 for value in payload["support"].values())
    assert payload["postentry_return_pnl_execution_price_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == (
        "108991e4c4bd2492dc2438b9f0f0ca740621fcbeccb68bd520fb263564b2da07"
    )
