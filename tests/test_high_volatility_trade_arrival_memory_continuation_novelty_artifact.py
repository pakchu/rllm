import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_trade_arrival_memory_continuation_gross9_novelty_2026-08-10.json")


def test_novelty_pass_is_frozen_blind_and_complete():
    artifact = json.loads(RESULT.read_text())
    assert artifact["every_gross9_sleeve_passed"] is True
    assert artifact["advance_to_economic_outcomes"] is True
    assert artifact["evidence_boundary"]["outcomes_opened"] is False
    assert artifact["evidence_boundary"]["funding_rows_opened"] == 0
    assert set(artifact["gross9_sleeves"]) == set(artifact["gross9_structural_clocks"]["complete_roster"])
    assert all(value["passed"] for value in artifact["gross9_sleeves"].values())
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == "01c222d91f2f303b5f4b750a7ff3a63d115a11dc50c7af76c0c1aead84d1bbb5"
