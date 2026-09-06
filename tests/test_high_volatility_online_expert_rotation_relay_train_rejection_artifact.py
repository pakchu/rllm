import hashlib
import json

from training import evaluate_high_volatility_online_expert_rotation_relay_economics as e


def test_hvoer_train_rejection_is_terminal_and_frozen():
    path = e.OUTPUTS["train"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "b8614a90ecd17bcac3fa5b920277085030939ad1422064066a2bf28223657634"
    result = json.loads(path.read_text())
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == e.canonical_hash(core) == "5f6e3664e62b4bf3acafdd826cfa44ab75790ae57142491b8470cfe136dc98e0"
    assert result["stage"] == "train"
    assert result["passed"] is False
    assert result["decision"] == "terminal_reject_no_repair"
    assert result["later_stage_outcomes_opened"] is False
