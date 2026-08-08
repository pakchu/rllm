import hashlib,json
from training import evaluate_nasdaq_volatility_rotation_btc_confirmation_relay_economics as e
def test_nvxcr_train_economics_frozen_terminal():
 p=e.OUTPUTS['train'];assert hashlib.sha256(p.read_bytes()).hexdigest()=='d44ff08060fdb70d6216c2a59132e3c7f71cd84fc6b6503cc79963fba17d4b58';d=json.loads(p.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'};assert d['manifest_hash']==e.canonical_hash(core) and d['passed'] is False and d['decision']=='terminal_reject_no_repair';assert d['primary']['base']['absolute_return_pct'] < -3 and d['primary']['base']['mean_gross_underlying_bp'] < -14 and d['primary']['stress']['absolute_return_pct'] < -4;assert d['checks']['absolute_return_positive'] is False and d['checks']['each_calendar_half_positive'] is False
