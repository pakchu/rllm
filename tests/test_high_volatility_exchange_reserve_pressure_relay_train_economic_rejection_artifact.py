import hashlib
import json

from training import evaluate_high_volatility_exchange_reserve_pressure_relay_economics as e


def test_hvexrp_train_rejection_is_terminal_and_reproducible():
    path = e.OUTPUTS["train"]
    result = json.loads(path.read_text(encoding="utf-8"))
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == e.canonical_hash(core)
    assert result["policy_id"] == "HVEXRP-24"
    assert result["stage"] == "train"
    assert result["passed"] is False
    assert result["decision"] == "terminal_reject_no_repair"
    assert result["advance_to_next_stage"] is False
    assert result["later_stage_outcomes_opened"] is False
    assert result["primary"]["base"]["absolute_return_pct"] < 0
    assert result["primary"]["base"]["mean_gross_underlying_bp"] < 0
    assert result["primary"]["stress"]["absolute_return_pct"] < 0
    assert not result["checks"]["each_calendar_half_positive"]
    assert not any(e.OUTPUTS[stage].exists() for stage in ("test", "eval", "final"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "5afd8e93670dcb2cb5acfd3755ad52431f250d41870165e8dd7fa55ccc0cc164"
    )
