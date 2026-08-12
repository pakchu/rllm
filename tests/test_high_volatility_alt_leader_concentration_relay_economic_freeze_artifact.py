import hashlib
import json
from pathlib import Path


FREEZE = Path(
    "results/high_volatility_alt_leader_concentration_relay_economic_evaluator_freeze_2026-08-13.json"
)
EXPECTED_SHA256 = "81fd7657d2666305998bc117a9ddeb95ffc47865c5891496f8856d58a27cf114"


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
