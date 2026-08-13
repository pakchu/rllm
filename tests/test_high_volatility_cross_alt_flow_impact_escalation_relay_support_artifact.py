import hashlib, json
from pathlib import Path
from training import build_high_volatility_cross_alt_flow_impact_escalation_relay_support as s

RESULT=Path('results/high_volatility_cross_alt_flow_impact_escalation_relay_support_2026-08-13.json')
EXPECTED='d0caec6f74a5c83b2f2a02f69150980f6749084016763dd6af274a83c879e2c8'

def test_source_rejection_is_immutable_and_later_evidence_remains_sealed():
 assert hashlib.sha256(RESULT.read_bytes()).hexdigest()==EXPECTED
 x=json.loads(RESULT.read_text());h=x.pop('manifest_hash');assert s.chash(x)==h
 assert x['policy_id']=='HVCAFIE-8' and not x['support_passed'] and x['decision']=='terminal_source_support_reject' and not x['advance_to_gross9_novelty'] and not x['advance_to_economic_outcomes']
 assert not x['postentry_return_pnl_execution_price_opened'] and not x['funding_values_opened'] and not x['gross9_rows_opened']
 assert {k:v['events'] for k,v in x['support'].items()}=={'train':4,'test':16,'eval':20,'final':11}
 assert not x['support_checks']['train_minimum_events'] and not x['support_checks']['train_side_balance'] and not x['support_checks']['test_side_balance']
 assert not Path('results/high_volatility_cross_alt_flow_impact_escalation_relay_gross9_novelty_2026-08-13.json').exists()
 for stage in ('train','test','eval','final'):assert not Path(f'results/high_volatility_cross_alt_flow_impact_escalation_relay_{stage}_economics_2026-08-13.json').exists()
