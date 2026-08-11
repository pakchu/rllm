import json
from pathlib import Path

from training import evaluate_high_volatility_hikkake_pattern_relay_economics as economics


def test_economic_freeze_is_outcome_blind_and_code_bound() -> None:
    artifact = json.loads(economics.FREEZE.read_text())
    core = {key: value for key, value in artifact.items() if key != "manifest_hash"}
    assert artifact["manifest_hash"] == economics.canonical_hash(core)
    assert artifact["outcomes_opened"] is False
    assert artifact["empty_diagnostic_controls_handled_before_outcomes"] is True
    assert economics.sha256(Path(artifact["evaluator"]["path"])) == artifact["evaluator"]["sha256"]
    assert economics.sha256(Path(artifact["authorization"]["path"])) == artifact["authorization"]["sha256"]
