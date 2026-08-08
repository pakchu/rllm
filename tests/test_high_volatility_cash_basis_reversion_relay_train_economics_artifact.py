import hashlib
import json

from training import evaluate_high_volatility_cash_basis_reversion_relay_economics as economics


def test_hvcbr_train_economics_is_terminal_and_later_stages_closed():
    assert hashlib.sha256(economics.OUTPUTS["train"].read_bytes()).hexdigest() == "99191a1671db1c6319861ce315952a3fdb5482bc81e1695cac415c375533c164"
    payload = json.loads(economics.OUTPUTS["train"].read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == economics.canonical_hash(core)
    assert payload["stage"] == "train"
    assert payload["passed"] is False
    assert payload["decision"] == "terminal_reject_no_repair"
    assert payload["advance_to_next_stage"] is False
    assert payload["later_stage_outcomes_opened"] is False
    assert payload["primary"]["base"]["absolute_return_pct"] < 0
    assert payload["primary"]["stress"]["absolute_return_pct"] < 0
    assert not economics.OUTPUTS["test"].exists()
    assert not economics.OUTPUTS["eval"].exists()
    assert not economics.OUTPUTS["final"].exists()
