import hashlib, json
from training import evaluate_daily_taker_flow_acceleration_relay_economics as economics


def test_dtfar_train_economics_is_terminal_and_later_stages_closed():
    assert hashlib.sha256(economics.OUTPUTS["train"].read_bytes()).hexdigest() == "232c0fce609393a9d0a233099d851f13d2ce18022d780928422f7d5a2c357871"
    payload = json.loads(economics.OUTPUTS["train"].read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == economics.canonical_hash(core)
    assert payload["stage"] == "train"
    assert payload["passed"] is False
    assert payload["decision"] == "terminal_reject_no_repair"
    assert payload["advance_to_next_stage"] is False
    assert payload["later_stage_outcomes_opened"] is False
    assert payload["primary"]["base"]["absolute_return_pct"] < 0
    assert payload["primary"]["stress"]["absolute_return_pct"] < 0
    assert not economics.OUTPUTS["test"].exists()
    assert not economics.OUTPUTS["eval"].exists()
    assert not economics.OUTPUTS["final"].exists()
