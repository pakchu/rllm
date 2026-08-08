import json

from training import evaluate_russell_volatility_spread_break_relay_economics as economics


def test_rvsbr_train_failure_is_terminal_and_later_stages_are_sealed():
    result = json.loads(economics.OUTPUTS["train"].read_text())
    assert result["policy_id"] == "RVSBR-12"
    assert result["stage"] == "train"
    assert result["passed"] is False
    assert result["advance_to_next_stage"] is False
    assert result["decision"] == "terminal_reject_no_repair"
    assert result["later_stage_outcomes_opened"] is False
    assert result["primary"]["base"]["trades"] == 18
    assert result["primary"]["base"]["absolute_return_pct"] < 0
    assert result["primary"]["stress"]["absolute_return_pct"] < 0


def test_rvsbr_train_report_is_hash_bound():
    result = json.loads(economics.OUTPUTS["train"].read_text())
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == economics.canonical_hash(core)
    assert result["novelty_authorization"]["sha256"] == economics.NOVELTY_SHA
