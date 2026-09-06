import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_treasury_belly_curvature_relay_gross9_novelty_2026-08-10.json")


def test_novelty_artifact_passes_every_sleeve_without_outcomes():
    payload = json.loads(RESULT.read_text())
    assert payload["every_gross9_sleeve_passed"] is True
    assert payload["advance_to_economic_outcomes"] is True
    assert all(item["passed"] for item in payload["gross9_sleeves"].values())
    boundary = payload["evidence_boundary"]
    assert boundary["outcomes_opened"] is False
    assert boundary["btc_execution_rows_opened"] == 0
    assert boundary["funding_rows_opened"] == 0
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == "6b7e9a8ad82e67506f758907dcab618d1203e9480c8b1062e34e0aa9935fb023"
