import hashlib,json
from training import evaluate_gold_volatility_rotation_btc_confirmation_relay_economics as e
def test_gvrcr_train_economics_frozen_terminal():
 p=e.OUTPUTS['train'];assert hashlib.sha256(p.read_bytes()).hexdigest()=='92b9565ba9839a6b40ba4aa545404204259fd6d298f52c27bcce37d459e8b8fe';d=json.loads(p.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'};assert d['manifest_hash']==e.canonical_hash(core) and d['passed'] is False and d['decision']=='terminal_reject_no_repair';assert d['primary']['base']['absolute_return_pct'] < -1 and d['primary']['base']['mean_gross_underlying_bp'] < 4 and d['primary']['stress']['absolute_return_pct'] < -2;assert d['checks']['absolute_return_positive'] is False and d['checks']['each_calendar_half_positive'] is False
