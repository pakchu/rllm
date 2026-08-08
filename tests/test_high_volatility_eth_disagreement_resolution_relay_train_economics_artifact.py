import hashlib
import json

from training import evaluate_high_volatility_eth_disagreement_resolution_relay_economics as economics


def test_hvedr_train_economics_is_terminal_and_later_stages_closed():
    assert hashlib.sha256(economics.OUTPUTS["train"].read_bytes()).hexdigest() == "32c71314188a43f66ac02f8ed0fafe0e21b1092d1fe74ba8c5dbb88b754ebbb8"
    payload = json.loads(economics.OUTPUTS["train"].read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == economics.canonical_hash(core)
    assert payload["stage"] == "train"
    assert payload["passed"] is False
    assert payload["decision"] == "terminal_reject_no_repair"
    assert payload["advance_to_next_stage"] is False
    assert payload["later_stage_outcomes_opened"] is False
    assert payload["primary"]["base"]["absolute_return_pct"] > 0
    assert payload["primary"]["base"]["mean_gross_underlying_bp"] < 20
    assert payload["primary"]["stress"]["absolute_return_pct"] < 0
    assert payload["primary"]["calendar_halves"]["first"]["absolute_return_pct"] < 0
    assert not economics.OUTPUTS["test"].exists()
    assert not economics.OUTPUTS["eval"].exists()
    assert not economics.OUTPUTS["final"].exists()
