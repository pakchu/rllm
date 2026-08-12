import hashlib
import json

from training import evaluate_high_volatility_oi_conditioned_late_variation_ignition_relay_economics as e


def test_hvoilvi_train_rejection_is_terminal_and_frozen():
    path = e.OUTPUTS["train"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "3b1b86b4a5c183a87540df5f90f501d1c7c996b08439229538f317bab5f5618b"
    result = json.loads(path.read_text())
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == e.canonical_hash(core) == "17dc7c1c53e5401f0535d123875deefebb56e26c4f484964478fb5496cb0367f"
    assert result["stage"] == "train" and result["passed"] is False
    assert result["decision"] == "terminal_reject_no_repair"
    assert result["later_stage_outcomes_opened"] is False
