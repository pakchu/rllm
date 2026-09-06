import hashlib
import json
from pathlib import Path

from training import evaluate_high_volatility_value_area_rejection_relay_economics as economics


def test_evaluator_freeze_is_outcome_blind() -> None:
    value = json.loads(economics.FREEZE.read_text())
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    assert value["manifest_hash"] == economics.canonical_hash(core)
    assert value["outcomes_opened"] is False
    assert value["empty_clock_policy"].startswith("outcome-blind load_clock_allow_empty")
    assert value["evaluator"]["sha256"] == hashlib.sha256(
        Path(value["evaluator"]["path"]).read_bytes()
    ).hexdigest()
