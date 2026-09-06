import hashlib
import json

from training import evaluate_cboe_skew_acceleration_dual_confirmation_relay_economics as e


def test_cvskdmr_train_economics_frozen_terminal():
    path = e.OUTPUTS["train"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "8450b92f0c8e1ff2beb53f92f278a4d8a415a002e66c34c160ef5330fd781ee8"
    data = json.loads(path.read_text())
    core = {key: value for key, value in data.items() if key != "manifest_hash"}
    assert data["manifest_hash"] == e.canonical_hash(core)
    assert data["passed"] is False
    assert data["decision"] == "terminal_reject_no_repair"
    assert data["primary"]["base"]["absolute_return_pct"] > 3
    assert data["primary"]["stress"]["absolute_return_pct"] > 2
    assert data["checks"]["each_calendar_half_positive"] is True
    assert data["checks"]["cagr_to_strict_mdd_min_3"] is False
    assert data["checks"]["cluster_signflip_p_max_0_1"] is False
