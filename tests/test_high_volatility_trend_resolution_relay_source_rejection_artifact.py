import hashlib,json
from training import build_high_volatility_trend_resolution_relay_support as s

def test_hvtrr_source_rejection_is_terminal_and_reproduced():
 assert hashlib.sha256(s.RESULT.read_bytes()).hexdigest()=='ed5e339421e9bc7e542dfe1c763bc75fd529b66711a451ec92ee629b8484abde'
 result=json.loads(s.RESULT.read_text());core={k:v for k,v in result.items() if k!='manifest_hash'}
 assert result['manifest_hash']==s.prereg.canonical_hash(core)=='6c48107b369b33ceb7c1443e410728f48cf4c575b4e2d581259bfc17efdc705f'
 assert result['support_passed'] is False
 assert result['support']['eval']['minority_side_share']==1/12 and result['support']['final']['events']==7
 assert result['decision']=='terminal_source_support_reject'
 assert result['postentry_return_pnl_execution_price_opened'] is False and result['gross9_rows_opened'] is False
