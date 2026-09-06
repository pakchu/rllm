import hashlib,json
from pathlib import Path
from training import evaluate_high_volatility_top_two_causal_expert_concordance_relay_economics as e

def test_evaluator_freeze_is_outcome_blind():
 value=json.loads(e.FREEZE.read_text());core={k:x for k,x in value.items() if k!='manifest_hash'}
 assert value['manifest_hash']==e.canonical_hash(core) and value['outcomes_opened'] is False
 assert value['empty_clock_policy'].startswith('outcome-blind load_clock_allow_empty')
 assert value['evaluator']['sha256']==hashlib.sha256(Path(value['evaluator']['path']).read_bytes()).hexdigest()
