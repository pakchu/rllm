import hashlib
import json

from training import build_high_volatility_inventory_conditioned_variation_onset_relay_support as s


def test_hvicvo_source_rejection_is_terminal_and_reproduced():
    assert hashlib.sha256(s.RESULT.read_bytes()).hexdigest() == "92fd671e2b94bfeccacfdb95618f696497d2bed56d657d9d7980e166f95a460f"
    result = json.loads(s.RESULT.read_text())
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == s.prereg.canonical_hash(core) == "7e1405c0b2c4fc1a4514ea13669a6166c063c34b57add141ba7bdb104d9f82ae"
    assert result["support_passed"] is False
    assert result["support_checks"]["test_month_concentration"] is False
    assert result["support"]["test"]["max_month_share"] == 19 / 42
    assert result["decision"] == "terminal_source_support_reject"
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["gross9_rows_opened"] is False
