import hashlib
import json

from training import evaluate_high_volatility_exchange_deposit_pressure_relay_economics as e


def test_hvexdp_train_rejection_is_terminal_and_reproducible():
    path = e.OUTPUTS["train"]
    result = json.loads(path.read_text(encoding="utf-8"))
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == e.canonical_hash(core)
    assert result["policy_id"] == "HVEXDP-24"
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
        "75396e6f86f36608c61d3b234579bbf80bf36ab8b5c9ae623f0a53ee3825dff6"
    )
