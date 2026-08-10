import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_crypto_market_mode_ignition_relay_gross9_novelty_2026-08-10.json")


def test_novelty_artifact_passes_every_sleeve_without_outcomes():
    payload = json.loads(RESULT.read_text())
    assert payload["every_gross9_sleeve_passed"] is True
    assert payload["advance_to_economic_outcomes"] is True
    assert all(item["passed"] for item in payload["gross9_sleeves"].values())
    boundary = payload["evidence_boundary"]
    assert boundary["outcomes_opened"] is False
    assert boundary["btc_execution_rows_opened"] == 0
    assert boundary["funding_rows_opened"] == 0
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == "d00a81c1a4021bad945d9498e449819e04e43505ef44b83688f57d6a8ddbebfc"
