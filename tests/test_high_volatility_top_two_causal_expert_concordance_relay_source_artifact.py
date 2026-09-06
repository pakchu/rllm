import hashlib,json
from training import build_high_volatility_top_two_causal_expert_concordance_relay_support as s

def test_hvtcec_source_pass_is_frozen_and_reproduced():
 assert hashlib.sha256(s.RESULT.read_bytes()).hexdigest()=='043e4f749db4821054bd112a0c384a89b5e88ebfa868eea4412c5e8deaabb549'
 result=json.loads(s.RESULT.read_text());core={k:v for k,v in result.items() if k!='manifest_hash'}
 assert result['manifest_hash']==s.prereg.canonical_hash(core)=='2b911b54a9472c160995d9d87ad7b180b13e4bd385f8196b9ef849542da8ccff'
 assert result['support_passed'] is True and all(result['support_checks'].values())
 assert result['clock']['sha256']=='761371c5e82b232889d88320dc16b074018a8b64540abd30b0f077bf44991070'
 assert result['advance_to_gross9_novelty'] is True
 assert result['funding_pnl_cagr_mdd_opened'] is False and result['gross9_rows_opened'] is False
