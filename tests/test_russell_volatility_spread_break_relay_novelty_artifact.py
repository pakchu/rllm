import json

from training import evaluate_russell_volatility_spread_break_relay_gross9_novelty as novelty


def test_rvsbr_novelty_pass_is_outcome_sealed():
    result = json.loads(novelty.OUTPUT.read_text())
    assert result["policy_id"] == "RVSBR-12"
    assert result["every_gross9_sleeve_passed"] is True
    assert result["gross9_novelty_status"] == "passed"
    assert result["advance_to_economic_outcomes"] is True
    boundary = result["evidence_boundary"]
    assert boundary["btc_execution_rows_opened"] == 0
    assert boundary["funding_rows_opened"] == 0
    assert boundary["economic_outcome_rows_opened"] == 0
    assert boundary["outcomes_opened"] is False


def test_rvsbr_passes_every_frozen_metric_for_every_sleeve():
    result = json.loads(novelty.OUTPUT.read_text())
    for sleeve in result["gross9_sleeves"].values():
        assert sleeve["passed"] is True
        assert all(sleeve["checks"].values())
        assert sleeve["metrics"]["exact_entry_jaccard"] == 0.0
        assert sleeve["metrics"]["one_to_one_6h_max_matched_share"] <= 0.35
