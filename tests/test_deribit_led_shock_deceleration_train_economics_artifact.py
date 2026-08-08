import hashlib,json
from training import evaluate_deribit_led_shock_deceleration_economics as e
P=e.OUTPUTS['train']
def test_dlsdr_train_economics_is_frozen_terminal_without_future_open():
 assert hashlib.sha256(P.read_bytes()).hexdigest()=='92af8c457332b2c2d235118eebe3e54bb17ed42710c7215e4b02dbe2925f92f7'
 d=json.loads(P.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'}
 assert d['manifest_hash']==e.canonical_hash(core)
 assert d['passed'] is False and d['decision']=='terminal_reject_no_repair'
 assert d['later_stage_outcomes_opened'] is False
 assert d['primary']['base']['absolute_return_pct']<0 and d['primary']['stress']['absolute_return_pct']<0
 assert d['primary']['base']['mean_gross_underlying_bp']<20
 assert not e.OUTPUTS['test'].exists() and not e.OUTPUTS['eval'].exists() and not e.OUTPUTS['final'].exists()
