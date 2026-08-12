import hashlib,json
from pathlib import Path
from training import build_high_volatility_safehaven_intraday_relative_semivariance_dislocation_relay_support as s
RESULT=Path('results/high_volatility_safehaven_intraday_relative_semivariance_dislocation_relay_support_2026-08-13.json');EXPECTED='3c4744c5126e79c9ee6858baf0ef836db76444cb5a2bdb1dedc380b525067569'
def test_terminal_source_rejection_is_immutable_and_blind():
 assert hashlib.sha256(RESULT.read_bytes()).hexdigest()==EXPECTED
 x=json.loads(RESULT.read_text());h=x.pop('manifest_hash');assert s.chash(x)==h
 assert x['policy_id']=='HVSIRD-8' and not x['support_passed'] and x['decision']=='terminal_source_support_reject'
 assert not x['advance_to_gross9_novelty'] and not x['advance_to_economic_outcomes'] and not x['postentry_return_pnl_execution_price_opened'] and not x['gross9_rows_opened']
 assert {k:v['events'] for k,v in x['support'].items()}=={'train':20,'test':34,'eval':31,'final':12}
 assert x['support']['final']['minority_side_share']==0 and x['clock']['rows']==97
