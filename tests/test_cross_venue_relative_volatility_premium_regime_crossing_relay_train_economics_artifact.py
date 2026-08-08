import hashlib,json
from training import evaluate_cross_venue_relative_volatility_premium_regime_crossing_relay_economics as e
def test_cvrvpr_train_economics_frozen_terminal():
 p=e.OUTPUTS['train'];assert hashlib.sha256(p.read_bytes()).hexdigest()=='38b803fcc95ea1b67ad577dfdd740ffa9988a92b32a2e9c18f4396388bccd388'
 d=json.loads(p.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'}
 assert d['manifest_hash']==e.canonical_hash(core) and d['passed'] is False and d['decision']=='terminal_reject_no_repair'
 assert d['primary']['base']['absolute_return_pct'] < -8 and d['primary']['base']['mean_gross_underlying_bp'] < 0
 assert d['primary']['stress']['absolute_return_pct'] < -12 and d['checks']['absolute_return_positive'] is False and d['checks']['each_calendar_half_positive'] is False
