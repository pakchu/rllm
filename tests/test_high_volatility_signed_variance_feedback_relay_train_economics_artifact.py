import hashlib,json
from pathlib import Path
R=Path('results/high_volatility_signed_variance_feedback_relay_train_economics_2026-08-10.json')
def test_terminal_train_reject_is_frozen():
 p=json.loads(R.read_text());assert p['stage']=='train' and p['passed'] is False and p['advance_to_next_stage'] is False and p['later_stage_outcomes_opened'] is False and p['decision']=='terminal_reject_no_repair';assert p['primary']['base']['absolute_return_pct']==-1.537027929240553 and p['primary']['base']['mean_gross_underlying_bp']==6.515108006673109 and p['primary']['cluster_signflip']['pvalue']==.6428035719642804;assert hashlib.sha256(R.read_bytes()).hexdigest()=='14a1403592aaff696353deab9048e348622e7c1293f8de84b35fd554af0e8ccf'
