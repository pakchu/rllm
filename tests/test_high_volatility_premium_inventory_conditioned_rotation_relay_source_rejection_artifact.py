import hashlib
import json

from training import build_high_volatility_premium_inventory_conditioned_rotation_relay_support as s


def test_hvpicr_source_rejection_is_terminal_and_reproduced():
    assert hashlib.sha256(s.RESULT.read_bytes()).hexdigest() == "21e7a781e20e62e2165dee9d14c80419e2495eb9d22d73ede1c822cf818cb392"
    result = json.loads(s.RESULT.read_text())
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == s.prereg.canonical_hash(core) == "c969c1767915b89bedd42281913b5d56d33dae87c675a072ebb84e3f11a0f1cc"
    assert result["support_passed"] is False
    assert result["support_checks"]["final_month_concentration"] is False
    assert result["support"]["final"]["max_month_share"] == 28 / 60
    assert result["decision"] == "terminal_source_support_reject"
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["gross9_rows_opened"] is False
