import hashlib
import json
from pathlib import Path


FREEZE = Path(
    "results/high_volatility_alt_breadth_diffusion_slope_relay_economic_evaluator_freeze_2026-08-13.json"
)
EXPECTED_SHA256 = "7a24378fbb1e8c013357dc5275061dce924919a7decc4b1ed13596d7b04d1311"


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
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
