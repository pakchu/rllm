import hashlib,json
from training import evaluate_high_volatility_cross_alt_broad_barrier_discovery_relay_economics as e
P=e.OUTPUTS['train'];EXPECTED='5e8540df57340ac3f36788b4167e24ba2fac79df4d0c88b146140c185e1110c4'
def test_train_rejection_is_immutable_and_later_stages_sealed():
 assert hashlib.sha256(P.read_bytes()).hexdigest()==EXPECTED;x=json.loads(P.read_text());h=x.pop('manifest_hash');assert e.canonical_hash(x)==h;assert x['stage']=='train' and not x['passed'] and x['decision']=='terminal_reject_no_repair' and not x['later_stage_outcomes_opened'];assert not x['advance_to_next_stage'] and not x['advance_to_post_stage_volatility_audit'];assert x['primary']['base']['absolute_return_pct']>0 and x['primary']['base']['cagr_to_strict_mdd']<3 and x['primary']['stress']['absolute_return_pct']<0;assert not e.OUTPUTS['test'].exists() and not e.OUTPUTS['eval'].exists() and not e.OUTPUTS['final'].exists()
