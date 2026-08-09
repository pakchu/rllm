import hashlib
import json

from training import evaluate_high_volatility_intrinsic_topology_ridge_relay_economics as economics


def test_hvitr_train_economics_is_terminal_and_later_stages_remain_sealed() -> None:
    output = economics.OUTPUTS["train"]
    assert hashlib.sha256(output.read_bytes()).hexdigest() == "36a2d0e00aecd82bb38b8d28d43ff449b1eabc62862f6e0281aa7869f972a9ae"
    payload = json.loads(output.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == economics.canonical_hash(core)
    assert payload["passed"] is False
    assert payload["decision"] == "terminal_reject_no_repair"
    assert payload["later_stage_outcomes_opened"] is False
    assert payload["primary"]["base"]["absolute_return_pct"] > 0
    assert payload["primary"]["base"]["cagr_to_strict_mdd"] < 3
    assert payload["primary"]["stress"]["absolute_return_pct"] < 0
    assert payload["primary"]["calendar_halves"]["first"]["absolute_return_pct"] < 0
    for stage in ("test", "eval", "final"):
        assert not economics.OUTPUTS[stage].exists()

