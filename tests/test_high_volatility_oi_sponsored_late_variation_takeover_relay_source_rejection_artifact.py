import hashlib
import json

from training import build_high_volatility_oi_sponsored_late_variation_takeover_relay_support as s


def test_hvoilvt_source_rejection_is_terminal_and_reproduced():
    assert hashlib.sha256(s.RESULT.read_bytes()).hexdigest() == "354b60af6b43f9ff8d044fd0230dde9a0e4c6d70b4dc3ef6743208688d42572e"
    result=json.loads(s.RESULT.read_text());core={k:v for k,v in result.items() if k!='manifest_hash'}
    assert result['manifest_hash']==s.prereg.canonical_hash(core)=='eaac2994a049d95f21c64d8ed87335c241f8eb65b2f222a8d92c09646f4af847'
    assert result['support_passed'] is False
    assert result['support_checks']['test_minimum_events'] is False
    assert result['support_checks']['test_side_balance'] is False
    assert result['support']['test']['events']==6
    assert result['decision']=='terminal_source_support_reject'
    assert result['postentry_return_pnl_execution_price_opened'] is False and result['gross9_rows_opened'] is False
