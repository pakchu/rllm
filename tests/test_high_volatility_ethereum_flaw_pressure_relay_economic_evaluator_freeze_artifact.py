import hashlib
import json
from pathlib import Path

from training import evaluate_high_volatility_ethereum_flaw_pressure_relay_economics as evaluator


def test_hvefpr_economic_evaluator_freeze_is_outcome_blind_and_bound():
    assert hashlib.sha256(evaluator.FREEZE.read_bytes()).hexdigest() == (
        "67ffe94f14e0dd093cf16dab5c06de0ea7771ef8a22ad9ac1dec90cbb3ebb0ae"
    )
    payload = json.loads(evaluator.FREEZE.read_text())
    assert payload["policy_id"] == "HVEFPR-24"
    assert payload["outcomes_opened"] is False
    assert payload["stage_order"] == ["train", "test", "eval", "final"]
    assert payload["stop_on_first_failure"] is True
    assert payload["empty_diagnostic_controls_handled_before_outcomes"] is True
    assert payload["evaluator"]["sha256"] == evaluator.sha256(Path(evaluator.__file__))
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == evaluator.canonical_hash(core)


def test_train_verification_opens_no_outcomes():
    novelty, freeze = evaluator.verify("train")
    assert novelty["advance_to_economic_outcomes"] is True
    assert freeze["outcomes_opened"] is False
