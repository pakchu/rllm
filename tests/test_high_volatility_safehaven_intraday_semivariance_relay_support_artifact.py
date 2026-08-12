import hashlib,json
from pathlib import Path
from training import build_high_volatility_safehaven_intraday_semivariance_relay_support as s
RESULT=Path('results/high_volatility_safehaven_intraday_semivariance_relay_support_2026-08-13.json');EXPECTED='38cc325981133b5f6ff0df0e9c27ce41da57eb214c14db2c366d0139aeb5bd7e'
def test_terminal_source_rejection_is_immutable_and_blind():
 assert hashlib.sha256(RESULT.read_bytes()).hexdigest()==EXPECTED
 x=json.loads(RESULT.read_text());h=x.pop('manifest_hash');assert s.chash(x)==h
 assert x['policy_id']=='HVSIS-8' and not x['support_passed'] and x['decision']=='terminal_source_support_reject'
 assert not x['advance_to_gross9_novelty'] and not x['advance_to_economic_outcomes'] and not x['postentry_return_pnl_execution_price_opened'] and not x['gross9_rows_opened']
 assert {k:v['events'] for k,v in x['support'].items()}=={'train':21,'test':42,'eval':36,'final':13}
 assert x['support']['final']['minority_side_share']==0 and x['clock']['rows']==112
