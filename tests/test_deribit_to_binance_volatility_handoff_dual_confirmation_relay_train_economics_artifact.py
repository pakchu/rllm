import hashlib
import json

from training import evaluate_deribit_to_binance_volatility_handoff_dual_confirmation_relay_economics as economics


def test_dbvhdr_train_economics_frozen_terminal():
    path = economics.OUTPUTS["train"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "f4f0c379dc320b25d8ab2a1f0fa855ae1c5526abb76664cd15bca044c6ddb74e"
    data = json.loads(path.read_text())
    core = {key: value for key, value in data.items() if key != "manifest_hash"}
    assert data["manifest_hash"] == economics.canonical_hash(core)
    assert data["passed"] is False
    assert data["decision"] == "terminal_reject_no_repair"
    assert data["primary"]["base"]["absolute_return_pct"] < -6
    assert data["primary"]["base"]["mean_gross_underlying_bp"] < 0
    assert data["primary"]["stress"]["absolute_return_pct"] < -9
    assert data["checks"]["absolute_return_positive"] is False
    assert data["checks"]["each_calendar_half_positive"] is False
