import json
from pathlib import Path

from training import evaluate_premium_volatility_ignition_acceleration_relay_economics as economics


def test_pviar_economic_evaluator_was_frozen_before_outcomes():
    report = json.loads(economics.FREEZE.read_text())
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == economics.canonical_hash(core)
    assert report["outcomes_opened"] is False
    assert report["stop_on_first_failure"] is True
    assert report["evaluator"]["sha256"] == economics.sha256(Path(economics.__file__))
    assert report["authorization"]["sha256"] == economics.NOVELTY_SHA
