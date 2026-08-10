import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_crypto_market_mode_ignition_relay_train_economics_2026-08-10.json")


def test_train_economics_is_terminal_and_later_stages_stay_sealed():
    payload = json.loads(RESULT.read_text())
    assert payload["stage"] == "train"
    assert payload["passed"] is False
    assert payload["advance_to_next_stage"] is False
    assert payload["advance_to_post_stage_volatility_audit"] is False
    assert payload["later_stage_outcomes_opened"] is False
    assert payload["decision"] == "terminal_reject_no_repair"
    assert payload["primary"]["base"]["absolute_return_pct"] == -4.2764955961521744
    assert payload["primary"]["base"]["mean_gross_underlying_bp"] == -2.903234223693916
    assert payload["primary"]["cluster_signflip"]["pvalue"] == 0.8901010989890101
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == "fcaf5c03c9ee65a6ed0a129710c0672e91ec7e3d66b4e4aaa15e1d03ab4146e0"
