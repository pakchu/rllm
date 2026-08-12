import hashlib,json
from training import build_high_volatility_state_conditioned_expert_rotation_relay_support as s

def test_hvscer_source_rejection_is_terminal_and_reproduced():
 assert hashlib.sha256(s.RESULT.read_bytes()).hexdigest()=='6d62ce0f170de419d62801e667157f87c75fa3379035cac0cdd783e0d879cbd6'
 result=json.loads(s.RESULT.read_text());core={k:v for k,v in result.items() if k!='manifest_hash'}
 assert result['manifest_hash']==s.prereg.canonical_hash(core)=='b90107701860d436c264719065f81b496d1349f1bc98bf5452a3041b5b1fdb42'
 assert result['support_passed'] is False
 assert [result['support'][x]['events'] for x in ('train','test','eval','final')]==[3,6,6,4]
 assert result['decision']=='terminal_source_support_reject'
 assert result['funding_pnl_cagr_mdd_opened'] is False and result['gross9_rows_opened'] is False
