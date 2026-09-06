import hashlib,json
from pathlib import Path
from training import preregister_high_volatility_mexican_peso_risk_relay as p
RESULT=Path('results/high_volatility_mexican_peso_risk_relay_support_2026-08-13.json');EXPECTED='b2fb3f7cb79b19ea25f90fb90ef2bd5d19bdcb13c6a06049f65f10b1ce21e273'
def test_terminal_source_rejection_is_immutable_and_blind():
 assert hashlib.sha256(RESULT.read_bytes()).hexdigest()==EXPECTED;x=json.loads(RESULT.read_text());h=x.pop('manifest_hash');assert p.canonical_hash(x)==h;assert x['policy_id']=='HVMXRR-8' and not x['support_passed'] and x['decision']=='terminal_source_support_reject';assert not x['advance_to_gross9_novelty'] and not x['advance_to_economic_outcomes'];assert not x['postentry_return_pnl_execution_price_opened'] and not x['gross9_rows_opened'];assert {k:v['events'] for k,v in x['support'].items()}=={'train':12,'test':24,'eval':24,'final':14};assert x['support']['train']['max_month_share']>.45
