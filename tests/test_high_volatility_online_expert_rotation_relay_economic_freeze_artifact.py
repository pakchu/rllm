import hashlib
import json
from pathlib import Path

from training import evaluate_high_volatility_online_expert_rotation_relay_economics as e


def test_evaluator_freeze_is_outcome_blind():
    value = json.loads(e.FREEZE.read_text())
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    assert value["manifest_hash"] == e.canonical_hash(core)
    assert value["outcomes_opened"] is False
    assert value["empty_clock_policy"].startswith("outcome-blind load_clock_allow_empty")
    assert value["evaluator"]["sha256"] == hashlib.sha256(Path(value["evaluator"]["path"]).read_bytes()).hexdigest()
