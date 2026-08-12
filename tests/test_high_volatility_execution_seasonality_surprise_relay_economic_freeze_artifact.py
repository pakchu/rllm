import json
from pathlib import Path

from training import evaluate_high_volatility_execution_seasonality_surprise_relay_economics as e


def test_freeze():
    value = json.loads(e.FREEZE.read_text())
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    assert value["manifest_hash"] == e.canonical_hash(core)
    assert value["outcomes_opened"] is False
    assert value["load_clock_allow_empty"] is True
    assert value["evaluator"]["sha256"] == e.sha256(Path(e.__file__))
    assert value["authorization"]["sha256"] == e.NOVELTY_SHA
