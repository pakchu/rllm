import hashlib
import json
from pathlib import Path

from training import evaluate_high_volatility_return_volume_transfer_entropy_relay_economics as economics


def test_freeze_binds_evaluator_before_outcomes() -> None:
    value = json.loads(economics.FREEZE.read_text())
    manifest_hash = value.pop("manifest_hash")
    assert economics.canonical_hash(value) == manifest_hash
    assert value["policy_id"] == "HVRVTE-8"
    assert not value["outcomes_opened"]
    assert value["evaluator"]["sha256"] == hashlib.sha256(Path(economics.__file__).read_bytes()).hexdigest()
    assert value["stage_order"] == ["train", "test", "eval", "final"]
    assert value["stop_on_first_failure"]
    assert "load_clock_allow_empty" in value["empty_clock_policy"]
