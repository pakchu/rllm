import hashlib
import json
from pathlib import Path


FREEZE = Path(
    "results/high_volatility_turning_point_deficiency_continuation_relay_economic_evaluator_freeze_2026-08-12.json"
)


def test_hvtpdcr_economic_evaluator_is_frozen_before_outcomes():
    report = json.loads(FREEZE.read_text())
    assert report["policy_id"] == "HVTPDCR-8"
    assert report["outcomes_opened"] is False
    assert report["stage_order"] == ["train", "test", "eval", "final"]
    assert report["stop_on_first_failure"] is True
    assert report["empty_diagnostic_controls_handled_before_outcomes"] is True
    path = Path(report["evaluator"]["path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == report["evaluator"]["sha256"]
