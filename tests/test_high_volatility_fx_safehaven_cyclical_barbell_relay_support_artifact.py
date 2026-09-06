import hashlib,json
from pathlib import Path
from training import preregister_high_volatility_fx_safehaven_cyclical_barbell_relay as p
RESULT=Path("results/high_volatility_fx_safehaven_cyclical_barbell_relay_support_2026-08-13.json");EXPECTED="e1170b295b2ef6b8a62d86542725ce89fba7fb8048a526068b8be3b5c959ad4d"
def test_terminal_source_rejection_is_immutable_and_blind():
 assert hashlib.sha256(RESULT.read_bytes()).hexdigest()==EXPECTED;x=json.loads(RESULT.read_text());h=x.pop('manifest_hash');assert p.canonical_hash(x)==h;assert x['policy_id']=='HVFXBR-8' and not x['support_passed'] and x['decision']=='terminal_source_support_reject';assert not x['advance_to_gross9_novelty'] and not x['advance_to_economic_outcomes'];assert not x['postentry_return_pnl_execution_price_opened'] and not x['gross9_rows_opened'];assert {k:v['events'] for k,v in x['support'].items()}=={'train':3,'test':10,'eval':7,'final':2}
