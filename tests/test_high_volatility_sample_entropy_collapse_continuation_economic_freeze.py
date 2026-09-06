import hashlib
import json
from pathlib import Path

from training import evaluate_high_volatility_sample_entropy_collapse_continuation_economics as economics


def test_freeze_binds_evaluator_before_outcomes() -> None:
    payload = json.loads(economics.FREEZE.read_text())
    manifest_hash = payload.pop("manifest_hash")
    assert economics.canonical_hash(payload) == manifest_hash
    assert payload["policy_id"] == "HVSENC-8"
    assert payload["outcomes_opened"] is False
    assert payload["evaluator"]["sha256"] == hashlib.sha256(Path(economics.__file__).read_bytes()).hexdigest()
    assert payload["stage_order"] == ["train", "test", "eval", "final"]
    assert payload["stop_on_first_failure"] is True
    assert "load_clock_allow_empty" in payload["empty_clock_policy"]
