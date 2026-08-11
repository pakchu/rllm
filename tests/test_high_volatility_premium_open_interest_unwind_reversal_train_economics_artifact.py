import json
from pathlib import Path


def test_train_failure_is_terminal_and_unrepaired():
    value = json.loads(Path("results/high_volatility_premium_open_interest_unwind_reversal_train_economics_2026-08-11.json").read_text())
    assert value["policy_id"] == "HVPOIUR-8"
    assert value["stage"] == "train"
    assert value["passed"] is False
    assert value["decision"] == "terminal_reject_no_repair"
    assert value["advance_to_next_stage"] is False
    assert value["later_stage_outcomes_opened"] is False
    assert value["primary"]["base"]["absolute_return_pct"] == -0.7641044514464679
    assert value["primary"]["base"]["mean_gross_underlying_bp"] == 9.756455713275601
    assert value["primary"]["cluster_signflip"]["pvalue"] == 0.5610143898561014
