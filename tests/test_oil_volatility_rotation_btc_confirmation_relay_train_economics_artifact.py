import hashlib,json
from training import evaluate_oil_volatility_rotation_btc_confirmation_relay_economics as e
def test_ovrcr_train_economics_frozen_terminal():
 p=e.OUTPUTS['train'];assert hashlib.sha256(p.read_bytes()).hexdigest()=='91623eb8860314f486ca9e7540cff70ebde68022554844548baf55a2a10d41e0';d=json.loads(p.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'};assert d['manifest_hash']==e.canonical_hash(core) and d['passed'] is False and d['decision']=='terminal_reject_no_repair';assert d['primary']['base']['absolute_return_pct'] < -1 and d['primary']['base']['mean_gross_underlying_bp'] < 2 and d['primary']['stress']['absolute_return_pct'] < -2;assert d['checks']['absolute_return_positive'] is False and d['checks']['each_calendar_half_positive'] is False
