import hashlib,json
from pathlib import Path
from training import evaluate_high_volatility_taker_imbalance_seasonal_innovation_economics as economics
def test_freeze_binds_evaluator_before_outcomes():
 x=json.loads(economics.FREEZE.read_text());h=x.pop('manifest_hash');assert economics.canonical_hash(x)==h and x['policy_id']=='HVTISI-8' and not x['outcomes_opened'];assert x['evaluator']['sha256']==hashlib.sha256(Path(economics.__file__).read_bytes()).hexdigest();assert x['stage_order']==['train','test','eval','final'] and x['stop_on_first_failure'] and 'load_clock_allow_empty' in x['empty_clock_policy']
