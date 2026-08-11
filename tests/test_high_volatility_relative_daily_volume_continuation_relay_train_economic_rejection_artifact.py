import hashlib
import json
from pathlib import Path

from training import (
    evaluate_high_volatility_relative_daily_volume_continuation_relay_economics as economics,
)


def test_hvrdv_train_economic_rejection_is_terminal_and_reproducible():
    path = economics.OUTPUTS["train"]
    result = json.loads(path.read_text())
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == economics.canonical_hash(core)
    assert result["policy_id"] == "HVRDV-8"
    assert result["stage"] == "train"
    assert result["passed"] is False
    assert result["decision"] == "terminal_reject_no_repair"
    assert result["advance_to_next_stage"] is False
    assert result["later_stage_outcomes_opened"] is False
    assert result["primary"]["base"]["absolute_return_pct"] < 0
    assert result["primary"]["base"]["mean_gross_underlying_bp"] < 0
    assert result["primary"]["stress"]["absolute_return_pct"] < 0
    assert not result["checks"]["each_calendar_half_positive"]
    assert not any(
        economics.OUTPUTS[stage].exists() for stage in ("test", "eval", "final")
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "28c4860bc9f52b7517d5b3d06697e9af171a7397ed6d413d12df4e36849fd73d"
    )
