import hashlib,json
from training import evaluate_high_volatility_top_two_causal_expert_concordance_relay_economics as e

def test_hvtcec_train_rejection_is_terminal_and_frozen():
 path=e.OUTPUTS['train'];assert hashlib.sha256(path.read_bytes()).hexdigest()=='5cceb2d66fbfe5ec01d2787e4081ce535e9b2a1b671ed5e74759af117b028bc0'
 result=json.loads(path.read_text());core={k:v for k,v in result.items() if k!='manifest_hash'}
 assert result['manifest_hash']==e.canonical_hash(core)=='ad090f3cd5dacce47d94ad63cc938ace84da264d985dc316a9ca4acb4d942865'
 assert result['stage']=='train' and result['passed'] is False
 assert result['decision']=='terminal_reject_no_repair' and result['later_stage_outcomes_opened'] is False
