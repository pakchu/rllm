import hashlib
import json
from pathlib import Path


FREEZE = Path(
    "results/high_volatility_daily_flow_impact_capacity_reversal_economic_evaluator_freeze_2026-08-13.json"
)
EXPECTED_SHA256 = "bb3780f58e67db7c54adb60ec81e9f3cea588354859c59f58d0d27b0de159e26"


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def test_evaluator_was_frozen_before_outcomes():
    assert hashlib.sha256(FREEZE.read_bytes()).hexdigest() == EXPECTED_SHA256
    payload = json.loads(FREEZE.read_text())
    manifest_hash = payload.pop("manifest_hash")
    assert canonical_hash(payload) == manifest_hash
    assert payload["outcomes_opened"] is False
    evaluator = Path(payload["evaluator"]["path"])
    assert hashlib.sha256(evaluator.read_bytes()).hexdigest() == payload["evaluator"]["sha256"]
    assert payload["stop_on_first_failure"] is True
    assert "load_clock_allow_empty" in payload["empty_clock_policy"]
