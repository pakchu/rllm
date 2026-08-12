import hashlib,json
from training import build_high_volatility_intraday_state_conditioned_expert_rotation_relay_support as s

def test_hviscer_source_rejection_is_terminal_and_reproduced():
 assert hashlib.sha256(s.RESULT.read_bytes()).hexdigest()=='3d28e0e1b3db6a5e374036e787074b05b4acd69d9386dc5977226f116d00ebe6'
 result=json.loads(s.RESULT.read_text());core={k:v for k,v in result.items() if k!='manifest_hash'}
 assert result['manifest_hash']==s.prereg.canonical_hash(core)=='58336dc0588ceb22aedd52707439f15b123cabe76be0df099876288f16ce018c'
 assert result['support_passed'] is False and result['support_checks']['final_month_concentration'] is False
 assert result['support']['final']['max_month_share']==6/11
 assert result['decision']=='terminal_source_support_reject'
 assert result['funding_pnl_cagr_mdd_opened'] is False and result['gross9_rows_opened'] is False
