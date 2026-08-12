import hashlib,json
from pathlib import Path
from training import evaluate_high_volatility_cross_alt_lagged_transfer_consensus_relay_economics as e
RESULT=Path('results/high_volatility_cross_alt_lagged_transfer_consensus_relay_train_economics_2026-08-13.json');EXPECTED='9954cedafb3cc1c737914cdb6460f0b3dd4232103e2ddb488da0ad797093c1c6'
def test_terminal_train_rejection_is_immutable_and_later_stages_absent():
 assert hashlib.sha256(RESULT.read_bytes()).hexdigest()==EXPECTED
 x=json.loads(RESULT.read_text());h=x.pop('manifest_hash');assert e.canonical_hash(x)==h
 assert x['policy_id']=='HVCALT-6' and x['stage']=='train' and not x['passed'] and x['decision']=='terminal_reject_no_repair' and not x['later_stage_outcomes_opened']
 b=x['primary']['base'];assert b['trades']==44 and b['absolute_return_pct']<0 and b['mean_gross_underlying_bp']<0
 assert not x['checks']['absolute_return_positive'] and not x['checks']['mean_gross_move_min_20bp'] and not x['checks']['each_calendar_half_positive']
 for stage in ('test','eval','final'):assert not Path(f'results/high_volatility_cross_alt_lagged_transfer_consensus_relay_{stage}_economics_2026-08-13.json').exists()
