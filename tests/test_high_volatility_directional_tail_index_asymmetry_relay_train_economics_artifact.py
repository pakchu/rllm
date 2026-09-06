import hashlib
import json

from training import evaluate_high_volatility_directional_tail_index_asymmetry_relay_economics as economics


def test_hvdtiar_train_rejection_is_terminal_and_later_stages_are_sealed():
    path = economics.OUTPUTS["train"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "de257d8300c7342678e5564cb055612b53fa180ae18a01544e76f22d7b19013c"
    result = json.loads(path.read_text())
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == economics.canonical_hash(core)
    assert result["passed"] is False
    assert result["decision"] == "terminal_reject_no_repair"
    assert result["later_stage_outcomes_opened"] is False
    assert result["primary"]["base"]["absolute_return_pct"] < -3
    assert result["primary"]["stress"]["absolute_return_pct"] < -5
    assert not economics.OUTPUTS["test"].exists()
    assert not economics.OUTPUTS["eval"].exists()
    assert not economics.OUTPUTS["final"].exists()
