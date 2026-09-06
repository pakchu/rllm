import json

from training import evaluate_high_volatility_quarter_hour_lagged_flow_relay_economics as economics


EXPECTED_SHA = "ad77dc428b1d43dfcbfad6b4e7f1eac85b60f3876433f872a6c7567aa43521f6"


def test_reproduced_train_failure_is_terminal_and_sealed() -> None:
    path = economics.OUTPUTS["train"]
    value = json.loads(path.read_text())
    assert economics.sha256(path) == EXPECTED_SHA
    assert value["stage"] == "train"
    assert value["passed"] is False
    assert value["advance_to_next_stage"] is False
    assert value["advance_to_post_stage_volatility_audit"] is False
    assert value["later_stage_outcomes_opened"] is False
    assert value["decision"] == "terminal_reject_no_repair"
    assert value["primary"]["base"]["absolute_return_pct"] < 0
    assert value["primary"]["base"]["mean_gross_underlying_bp"] < 20
    assert value["primary"]["cluster_signflip"]["pvalue"] > 0.1
    assert not any(
        half["absolute_return_pct"] > 0
        for half in value["primary"]["calendar_halves"].values()
    )
