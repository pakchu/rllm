import hashlib,json
from pathlib import Path
from training import preregister_high_volatility_cash_range_sponsorship_relay as p
RESULT=Path('results/high_volatility_cash_range_sponsorship_relay_support_2026-08-13.json');EXPECTED='f812c1abb209a60060f890cd3907944595c9f41e804d75234af39c63c401ee1d'
def test_source_pass_is_immutable_blind_and_authorizes_only_novelty():
 assert hashlib.sha256(RESULT.read_bytes()).hexdigest()==EXPECTED;x=json.loads(RESULT.read_text());h=x.pop('manifest_hash');assert p.canonical_hash(x)==h;assert x['policy_id']=='HVCRS-8' and x['support_passed'] and x['decision']=='pass_to_novelty';assert x['advance_to_gross9_novelty'] and not x['advance_to_economic_outcomes'];assert not x['postentry_return_pnl_execution_price_opened'] and not x['gross9_rows_opened'];assert {k:v['events'] for k,v in x['support'].items()}=={'train':65,'test':157,'eval':148,'final':59};assert all(x['support_checks'].values())
