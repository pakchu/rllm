import hashlib
import json

from training import evaluate_high_volatility_value_area_rejection_relay_economics as economics


def test_hvvar_train_rejection_is_frozen_and_terminal() -> None:
    path = economics.OUTPUTS["train"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "302f8d01298027013012b1802d325064e43e63a5a736c4a273631711e8c41bde"
    )
    result = json.loads(path.read_text())
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == economics.canonical_hash(core) == (
        "f298baaae504118d08d56da4660f23601561b4402c731596d88700624ef3f5a3"
    )
    assert result["stage"] == "train"
    assert result["passed"] is False
    assert result["decision"] == "terminal_reject_no_repair"
    assert result["later_stage_outcomes_opened"] is False
    assert result["checks"]["strict_mdd_max_15"] is True
    assert not all(result["checks"].values())
