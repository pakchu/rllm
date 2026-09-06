import hashlib,json
from pathlib import Path
from training import preregister_high_volatility_leverage_only_wick_rejection_reversal as p
RESULT=Path('results/high_volatility_leverage_only_wick_rejection_reversal_support_2026-08-13.json');EXPECTED='7fcc2c7d647044570a0d076969d709266308dbb0d75b9cd32b687fcb8059a0d1'
def test_source_pass_is_immutable_blind_and_authorizes_only_novelty():
 assert hashlib.sha256(RESULT.read_bytes()).hexdigest()==EXPECTED;x=json.loads(RESULT.read_text());h=x.pop('manifest_hash');assert p.canonical_hash(x)==h;assert x['policy_id']=='HVLOWR-8' and x['support_passed'] and x['decision']=='pass_to_novelty';assert x['advance_to_gross9_novelty'] and not x['advance_to_economic_outcomes'];assert not x['postentry_return_pnl_execution_price_opened'] and not x['gross9_rows_opened'];assert {k:v['events'] for k,v in x['support'].items()}=={'train':50,'test':104,'eval':109,'final':71};assert all(x['support_checks'].values())
