import hashlib
import json
from pathlib import Path
from training import build_high_volatility_funding_streak_exhaustion_reversal_support as support

RESULT=Path('results/high_volatility_funding_streak_exhaustion_reversal_support_2026-08-13.json')
EXPECTED='3ba98c6c377d8d09295d507ba22019c85b2fa9efad2b96fc50fa8571649b8ee8'

def test_terminal_source_rejection_is_immutable_and_blind():
 assert hashlib.sha256(RESULT.read_bytes()).hexdigest()==EXPECTED
 x=json.loads(RESULT.read_text());h=x.pop('manifest_hash');assert support.chash(x)==h
 assert x['policy_id']=='HVFSE-8' and not x['support_passed'] and x['decision']=='terminal_source_support_reject'
 assert not x['advance_to_gross9_novelty'] and not x['advance_to_economic_outcomes']
 assert not x['postentry_return_pnl_execution_price_opened'] and not x['gross9_rows_opened']
 assert {k:v['events'] for k,v in x['support'].items()}=={'train':14,'test':17,'eval':2,'final':2}
 assert all(v['longs']==0 and v['minority_side_share']==0 for v in x['support'].values())
