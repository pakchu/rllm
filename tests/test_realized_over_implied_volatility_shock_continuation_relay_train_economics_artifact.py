import hashlib,json
from training import evaluate_realized_over_implied_volatility_shock_continuation_relay_economics as e
def test_rivscr_train_economics_frozen_terminal():
 p=e.OUTPUTS['train'];assert hashlib.sha256(p.read_bytes()).hexdigest()=='4b5a6d75c7a5f3c98f1563ba67aed49e7065db4fe0e6866b3a2a8e6bea2b7b28';d=json.loads(p.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'};assert d['manifest_hash']==e.canonical_hash(core) and d['passed'] is False and d['decision']=='terminal_reject_no_repair';assert d['primary']['base']['absolute_return_pct'] < -1 and d['primary']['base']['mean_gross_underlying_bp'] < 5 and d['primary']['stress']['absolute_return_pct'] < -3;assert d['checks']['absolute_return_positive'] is False and d['checks']['each_calendar_half_positive'] is False
