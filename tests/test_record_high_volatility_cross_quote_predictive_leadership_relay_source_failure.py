import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_cross_quote_predictive_leadership_relay_source_failure_2026-08-11.json")


def test_terminal_source_contract_failure_is_frozen():
    payload = json.loads(RESULT.read_text())
    assert payload["decision"] == "terminal_source_contract_reject"
    assert payload["missing_symbols"] == ["BTCFDUSD", "BTCUSDC"]
    assert payload["candidate_clock_rows_built"] == 0
    assert payload["advance_to_gross9_novelty"] is False
    assert payload["advance_to_economic_outcomes"] is False
    assert payload["postentry_return_pnl_execution_price_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == "ef0a5244506e90fc585e5ecf21567e076aaff1382b0648168b3da7b9d1d92b75"
