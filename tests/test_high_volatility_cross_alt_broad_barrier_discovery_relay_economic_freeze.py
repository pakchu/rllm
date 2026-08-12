import hashlib,json
from pathlib import Path
from training import evaluate_high_volatility_cross_alt_broad_barrier_discovery_relay_economics as e
P=e.FREEZE
def test_freeze_binds_evaluator_before_outcomes():
 x=json.loads(P.read_text());h=x.pop('manifest_hash');assert e.canonical_hash(x)==h;assert x['policy_id']=='HVCABBD-8' and not x['outcomes_opened'];assert x['evaluator']['sha256']==hashlib.sha256(Path(e.__file__).read_bytes()).hexdigest();assert x['stage_order']==['train','test','eval','final'] and x['stop_on_first_failure'];assert 'load_clock_allow_empty' in x['empty_clock_policy']
