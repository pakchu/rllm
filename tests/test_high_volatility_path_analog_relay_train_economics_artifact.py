import hashlib
import json

from training import evaluate_high_volatility_path_analog_relay_economics as economics


def test_hvpar_train_economics_is_terminal_and_later_stages_remain_sealed() -> None:
    output = economics.OUTPUTS["train"]
    assert hashlib.sha256(output.read_bytes()).hexdigest() == "55441f573d322670e674423378f32dc1f3206845a53fd0a8fb5bbf513ec7efeb"
    payload = json.loads(output.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == economics.canonical_hash(core)
    assert payload["passed"] is False
    assert payload["decision"] == "terminal_reject_no_repair"
    assert payload["later_stage_outcomes_opened"] is False
    assert payload["primary"]["base"]["absolute_return_pct"] < 0
    assert payload["primary"]["base"]["mean_gross_underlying_bp"] < 20
    assert payload["primary"]["stress"]["absolute_return_pct"] < 0
    for stage in ("test", "eval", "final"):
        assert not economics.OUTPUTS[stage].exists()

