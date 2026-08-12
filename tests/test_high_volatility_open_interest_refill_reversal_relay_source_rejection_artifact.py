import hashlib
import json

from training import build_high_volatility_open_interest_refill_reversal_relay_support as s


def test_hvoirr_source_rejection_is_terminal_and_reproduced():
    assert hashlib.sha256(s.RESULT.read_bytes()).hexdigest() == "1dd29c1096ee5fbc7d693c551fd7fb9e47c5f52b6b3d108fd80e79300a26cf93"
    result = json.loads(s.RESULT.read_text())
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == s.prereg.canonical_hash(core) == "2e42934431c9d747168516634df106781fa54eb9f64a050a1bcfdff9402d3bfb"
    assert result["support_passed"] is False
    assert result["support_checks"]["final_month_concentration"] is False
    assert result["support"]["final"]["max_month_share"] == 11 / 24
    assert result["decision"] == "terminal_source_support_reject"
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["gross9_rows_opened"] is False
