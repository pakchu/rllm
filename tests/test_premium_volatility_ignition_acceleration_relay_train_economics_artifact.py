import hashlib
import json

from training import evaluate_premium_volatility_ignition_acceleration_relay_economics as economics


def test_pviar_train_economics_is_terminal_and_later_stages_remain_sealed():
    path = economics.OUTPUTS["train"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "d9702e65ae4af8d61f24a1c024674b1f5a96b19dbc1a214396f60ec105a1deeb"
    report = json.loads(path.read_text())
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == economics.canonical_hash(core)
    assert report["stage"] == "train"
    assert report["passed"] is False
    assert report["decision"] == "terminal_reject_no_repair"
    assert report["advance_to_next_stage"] is False
    assert report["later_stage_outcomes_opened"] is False
    assert report["primary"]["base"]["absolute_return_pct"] < 0
    assert report["primary"]["stress"]["absolute_return_pct"] < 0
    assert all(
        half["absolute_return_pct"] < 0
        for half in report["primary"]["calendar_halves"].values()
    )
    assert not economics.OUTPUTS["test"].exists()
    assert not economics.OUTPUTS["eval"].exists()
    assert not economics.OUTPUTS["final"].exists()
