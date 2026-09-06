import hashlib
import json

from training import evaluate_cross_alt_flow_acceleration_exhaustion_reversal_gross9_novelty as novelty


def test_cafaer_novelty_failure_is_outcome_sealed():
    assert hashlib.sha256(novelty.OUTPUT.read_bytes()).hexdigest() == "76550ae53bdcd8bfc62b0bbd7b431750ea933f87b75b615ed00acbd5cbefd326"
    result = json.loads(novelty.OUTPUT.read_text())
    assert result["policy_id"] == "CAFAER-12"
    assert result["every_gross9_sleeve_passed"] is False
    assert result["gross9_novelty_status"] == "failed"
    assert result["advance_to_economic_outcomes"] is False
    boundary = result["evidence_boundary"]
    assert boundary["btc_execution_rows_opened"] == 0
    assert boundary["funding_rows_opened"] == 0
    assert boundary["economic_outcome_rows_opened"] == 0
    assert boundary["outcomes_opened"] is False


def test_only_fresh_kimchi_near_six_hour_share_fails():
    result = json.loads(novelty.OUTPUT.read_text())
    failures = [name for name, sleeve in result["gross9_sleeves"].items() if not sleeve["passed"]]
    assert failures == ["fresh_kimchi_fx"]
    sleeve = result["gross9_sleeves"]["fresh_kimchi_fx"]
    assert sleeve["metrics"]["one_to_one_6h_max_matched_share"] == 16 / 37
    assert sleeve["checks"]["one_to_one_6h_max_matched_share"] is False
    assert all(value for key, value in sleeve["checks"].items() if key != "one_to_one_6h_max_matched_share")
