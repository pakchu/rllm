import hashlib,json
from pathlib import Path
from training import build_high_volatility_em_fx_dollar_stress_breadth_relay_support as s
RESULT=Path('results/high_volatility_em_fx_dollar_stress_breadth_relay_support_2026-08-13.json');EXPECTED='8e93f89fb4965697b27b8c53fcdc12212eb58936e550f37a3d0562df77461c39'
def test_source_rejection_is_immutable_and_outcomes_remain_sealed():
 assert hashlib.sha256(RESULT.read_bytes()).hexdigest()==EXPECTED
 x=json.loads(RESULT.read_text());h=x.pop('manifest_hash');assert s.chash(x)==h
 assert x['policy_id']=='HVEMFX-12' and not x['support_passed'] and x['decision']=='terminal_source_support_reject' and not x['advance_to_gross9_novelty'] and not x['advance_to_economic_outcomes']
 assert not x['postentry_return_pnl_execution_price_opened'] and not x['funding_values_opened'] and not x['gross9_rows_opened']
 assert {k:v['events'] for k,v in x['support'].items()}=={'train':8,'test':13,'eval':0,'final':0};assert not x['support_checks']['train_side_balance'] and not x['support_checks']['eval_minimum_events'] and not x['support_checks']['final_minimum_events']
 assert not Path('results/high_volatility_em_fx_dollar_stress_breadth_relay_gross9_novelty_2026-08-13.json').exists()
 for stage in ('train','test','eval','final'):assert not Path(f'results/high_volatility_em_fx_dollar_stress_breadth_relay_{stage}_economics_2026-08-13.json').exists()
