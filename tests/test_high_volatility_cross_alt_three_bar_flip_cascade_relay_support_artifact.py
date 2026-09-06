import hashlib,json
from pathlib import Path
from training import build_high_volatility_cross_alt_three_bar_flip_cascade_relay_support as s
RESULT=Path('results/high_volatility_cross_alt_three_bar_flip_cascade_relay_support_2026-08-13.json');EXPECTED='8ef99252b5803dae28eb63f8b2a9c7fd24f405b9aca7134abbb8091ab64d60c7'
def test_source_pass_is_immutable_and_outcomes_remain_sealed():
 assert hashlib.sha256(RESULT.read_bytes()).hexdigest()==EXPECTED
 x=json.loads(RESULT.read_text());h=x.pop('manifest_hash');assert s.chash(x)==h
 assert x['policy_id']=='HVCAFC-6' and x['support_passed'] and x['decision']=='pass_to_novelty' and x['advance_to_gross9_novelty'] and not x['advance_to_economic_outcomes']
 assert not x['postentry_return_pnl_execution_price_opened'] and not x['funding_values_opened'] and not x['gross9_rows_opened']
 assert {k:v['events'] for k,v in x['support'].items()}=={'train':33,'test':91,'eval':115,'final':62}
 assert all(v['minority_side_share']>=.2 and v['max_month_share']<=.45 for v in x['support'].values());assert x['clock']['rows']==301
