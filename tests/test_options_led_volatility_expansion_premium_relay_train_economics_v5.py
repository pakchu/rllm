import hashlib,json
from pathlib import Path
from training import evaluate_options_led_volatility_expansion_premium_relay_economics_v5 as e

P=Path('results/options_led_volatility_expansion_premium_relay_train_economics_v5_2026-08-08.json')

def test_v5_train_result_is_hash_bound_terminal_rejection():
 assert hashlib.sha256(P.read_bytes()).hexdigest()=='efe8acd539d37f0fde44b10a82947ec9ecad056f4532661fbcd20ad99ed4a124'
 d=json.loads(P.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'}
 assert d['manifest_hash']==e.chash(core)=='c1d0b898cd989110a6304d0c91f483f41b5582077c7c7267928ad1b80430adfd'
 assert d['passed'] is False and d['decision']=='terminal_reject_no_repair'
 assert not any(d['checks'].values()) and d['later_stage_outcomes_opened'] is False
