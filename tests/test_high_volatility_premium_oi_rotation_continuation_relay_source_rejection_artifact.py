import hashlib
import json

from training import build_high_volatility_premium_oi_rotation_continuation_relay_support as s


def test_hvporc_source_rejection_is_terminal_and_reproduced():
    assert hashlib.sha256(s.RESULT.read_bytes()).hexdigest() == "15583c7c0212d9ef5ee3a049bbcb7e8809368fa82a6ddc980f6a1972dcefee13"
    result = json.loads(s.RESULT.read_text())
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == s.prereg.canonical_hash(core) == "34daf62ea0b23823c277d2fd98e78a93b92147e32e6c75a474064a6ccef7f55b"
    assert result["support_passed"] is False
    assert result["support_checks"]["final_month_concentration"] is False
    assert result["support"]["final"]["max_month_share"] == 14 / 30
    assert result["decision"] == "terminal_source_support_reject"
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["gross9_rows_opened"] is False
