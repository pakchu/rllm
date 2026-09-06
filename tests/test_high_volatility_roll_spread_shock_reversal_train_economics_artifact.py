import hashlib,json
from pathlib import Path
R=Path('results/high_volatility_roll_spread_shock_reversal_train_economics_2026-08-10.json')
def test_terminal_train_reject_is_frozen():
 p=json.loads(R.read_text());assert p['stage']=='train' and p['passed'] is False and p['advance_to_next_stage'] is False and p['later_stage_outcomes_opened'] is False and p['decision']=='terminal_reject_no_repair'
 assert p['primary']['base']['absolute_return_pct']==-6.900273683186642 and p['primary']['base']['mean_gross_underlying_bp']==-23.01665669841843 and p['primary']['cluster_signflip']['pvalue']==0.971650283497165
 assert hashlib.sha256(R.read_bytes()).hexdigest()=='022dca5d0b2a64e0371b62c08443dc4b7a4cc9bbcbfd002cff8c07e4ed546089'
