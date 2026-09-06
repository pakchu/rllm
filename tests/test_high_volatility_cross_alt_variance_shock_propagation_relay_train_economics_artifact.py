import hashlib,json
from pathlib import Path
from training import evaluate_high_volatility_cross_alt_variance_shock_propagation_relay_economics as e
RESULT=Path('results/high_volatility_cross_alt_variance_shock_propagation_relay_train_economics_2026-08-13.json');EXPECTED='dab1c8affe5fcaeeb39377e0d4cfea1e1b3cf0b3bbb50d5a7f7adec05c6669ad'
def test_terminal_train_rejection_is_immutable_and_later_stages_absent():
 assert hashlib.sha256(RESULT.read_bytes()).hexdigest()==EXPECTED
 x=json.loads(RESULT.read_text());h=x.pop('manifest_hash');assert e.canonical_hash(x)==h
 assert x['policy_id']=='HVCAVS-8' and x['stage']=='train' and not x['passed'] and x['decision']=='terminal_reject_no_repair' and not x['later_stage_outcomes_opened']
 b=x['primary']['base'];assert b['trades']==42 and b['absolute_return_pct']<0 and b['mean_gross_underlying_bp']<20
 assert x['primary']['calendar_halves']['first']['absolute_return_pct']>0 and x['primary']['calendar_halves']['second']['absolute_return_pct']<0
 for stage in ('test','eval','final'):assert not Path(f'results/high_volatility_cross_alt_variance_shock_propagation_relay_{stage}_economics_2026-08-13.json').exists()
