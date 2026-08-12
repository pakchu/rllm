import json
from pathlib import Path
from training import evaluate_high_volatility_execution_count_elasticity_relay_economics as e
def test_freeze():
 x=json.loads(e.FREEZE.read_text());core={k:v for k,v in x.items() if k!='manifest_hash'};assert x['manifest_hash']==e.canonical_hash(core);assert x['outcomes_opened'] is False and x['load_clock_allow_empty'] is True;assert x['evaluator']['sha256']==e.sha256(Path(e.__file__));assert x['authorization']['sha256']==e.NOVELTY_SHA
