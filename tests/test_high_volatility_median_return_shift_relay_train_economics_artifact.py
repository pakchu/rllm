import hashlib
import json

from training import evaluate_high_volatility_median_return_shift_relay_economics as economics


def test_hvmrsr_train_rejection_is_terminal_and_later_stages_are_sealed():
    path = economics.OUTPUTS["train"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "b82ac2b6254aadd36e2dd8eb5d21ce61e82371416d3e9f29574327478186b7f7"
    )
    result = json.loads(path.read_text())
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == economics.canonical_hash(core)
    assert result["passed"] is False
    assert result["decision"] == "terminal_reject_no_repair"
    assert result["advance_to_next_stage"] is False
    assert result["later_stage_outcomes_opened"] is False
    assert result["primary"]["base"]["absolute_return_pct"] > 9
    assert result["primary"]["stress"]["absolute_return_pct"] > 7
    assert result["checks"]["each_calendar_half_positive"] is False
    assert result["primary"]["calendar_halves"]["first"]["absolute_return_pct"] < 0
    assert not economics.OUTPUTS["test"].exists()
    assert not economics.OUTPUTS["eval"].exists()
    assert not economics.OUTPUTS["final"].exists()
