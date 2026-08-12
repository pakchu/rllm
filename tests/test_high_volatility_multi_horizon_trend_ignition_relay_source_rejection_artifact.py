import hashlib
import json

from training import build_high_volatility_multi_horizon_trend_ignition_relay_support as s


def test_hvmti_source_rejection_is_terminal_and_reproduced():
    assert hashlib.sha256(s.RESULT.read_bytes()).hexdigest()=='9db3bc459c3458d3923d3b4fb505cba72b68bff2d7b3ad63b24630d6331d2bd5'
    result=json.loads(s.RESULT.read_text());core={k:v for k,v in result.items() if k!='manifest_hash'}
    assert result['manifest_hash']==s.prereg.canonical_hash(core)=='0147f5b89dd3b4551a2a5f80d66dc0191b9097391cfd8e7f13c45e547143ae20'
    assert result['support_passed'] is False and result['support_checks']['final_side_balance'] is False
    assert result['support']['final']['minority_side_share']==2/11
    assert result['decision']=='terminal_source_support_reject'
    assert result['postentry_return_pnl_execution_price_opened'] is False and result['gross9_rows_opened'] is False
