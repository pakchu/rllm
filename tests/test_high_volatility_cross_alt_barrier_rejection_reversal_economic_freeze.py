import hashlib
import json
from pathlib import Path
from training import evaluate_high_volatility_cross_alt_barrier_rejection_reversal_economics as e


def test_freeze_binds_evaluator_before_outcomes():
    frozen = json.loads(e.FREEZE.read_text())
    manifest_hash = frozen.pop("manifest_hash")
    assert e.canonical_hash(frozen) == manifest_hash
    assert frozen["policy_id"] == "HVCABRR-8" and not frozen["outcomes_opened"]
    assert frozen["evaluator"]["sha256"] == hashlib.sha256(Path(e.__file__).read_bytes()).hexdigest()
    assert frozen["stage_order"] == ["train", "test", "eval", "final"]
    assert frozen["stop_on_first_failure"]
    assert "load_clock_allow_empty" in frozen["empty_clock_policy"]
