import hashlib
import json

from training import evaluate_sterling_euro_risk_beta_relay_gross9_novelty as novelty


def test_serbr_novelty_pass_is_outcome_sealed():
    assert hashlib.sha256(novelty.OUTPUT.read_bytes()).hexdigest() == (
        "9afdb488e01eb4501db7691a7a86791cb2f2223000e0b7fda0af827f0541ac3e"
    )
    result = json.loads(novelty.OUTPUT.read_text())
    assert result["policy_id"] == "SERBR-12"
    assert result["every_gross9_sleeve_passed"] is True
    assert result["gross9_novelty_status"] == "passed"
    assert result["advance_to_economic_outcomes"] is True
    boundary = result["evidence_boundary"]
    assert boundary["btc_execution_rows_opened"] == 0
    assert boundary["funding_rows_opened"] == 0
    assert boundary["economic_outcome_rows_opened"] == 0
    assert boundary["outcomes_opened"] is False


def test_serbr_passes_every_frozen_metric():
    result = json.loads(novelty.OUTPUT.read_text())
    for comparison in result["gross9_sleeves"].values():
        assert comparison["passed"] is True
        assert all(comparison["checks"].values())
        for metric, limit in novelty.LIMITS.items():
            assert comparison["metrics"][metric] <= limit
