import hashlib
import json

from training import evaluate_sterling_euro_risk_beta_relay_economics as economics


def test_serbr_train_failure_is_terminal_and_later_stages_are_sealed():
    path = economics.OUTPUTS["train"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "48dc19a3d4b6380bcfb7de886fde6088a8ef900b00209c5328245af5634e91da"
    )
    result = json.loads(path.read_text())
    assert result["policy_id"] == "SERBR-12"
    assert result["stage"] == "train"
    assert result["passed"] is False
    assert result["advance_to_next_stage"] is False
    assert result["decision"] == "terminal_reject_no_repair"
    assert result["later_stage_outcomes_opened"] is False
    assert result["primary"]["base"]["trades"] == 20
    assert result["primary"]["base"]["absolute_return_pct"] < 0
    assert result["primary"]["base"]["mean_gross_underlying_bp"] < 0
    assert result["primary"]["stress"]["absolute_return_pct"] < 0
    assert result["primary"]["calendar_halves"]["first"]["trades"] == 0
    assert not economics.OUTPUTS["test"].exists()
    assert not economics.OUTPUTS["eval"].exists()
    assert not economics.OUTPUTS["final"].exists()


def test_serbr_train_report_is_hash_bound():
    result = json.loads(economics.OUTPUTS["train"].read_text())
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == economics.canonical_hash(core)
    assert result["novelty_authorization"]["sha256"] == economics.NOVELTY_SHA
