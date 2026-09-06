import json
from pathlib import Path
from training import evaluate_high_volatility_intraday_variance_dispersion_relay_economics as e

def test_outcome_blind_bound_freeze():
 x=json.loads(e.FREEZE.read_text());core={k:v for k,v in x.items() if k!='manifest_hash'}
 assert x['manifest_hash']==e.canonical_hash(core);assert x['outcomes_opened'] is False;assert x['load_clock_allow_empty'] is True
 assert x['evaluator']['sha256']==e.sha256(Path(e.__file__)) and x['authorization']['sha256']==e.NOVELTY_SHA
 assert x['stop_on_first_failure'] is True and x['stage_order']==['train','test','eval','final']
