import hashlib
import json
from pathlib import Path


FREEZE = Path(
    "results/high_volatility_block_range_breakout_retest_acceptance_relay_economic_evaluator_freeze_2026-08-10.json"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def test_hvrbrar_economic_freeze_is_outcome_blind_and_code_bound():
    artifact = json.loads(FREEZE.read_text())
    core = {key: value for key, value in artifact.items() if key != "manifest_hash"}
    assert artifact["manifest_hash"] == canonical_hash(core)
    assert artifact["outcomes_opened"] is False
    assert artifact["empty_diagnostic_controls_handled_before_outcomes"] is True
    assert sha256(Path(artifact["evaluator"]["path"])) == artifact["evaluator"]["sha256"]
    assert sha256(Path(artifact["authorization"]["path"])) == artifact["authorization"]["sha256"]
