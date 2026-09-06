import hashlib,json
from pathlib import Path
from training import preregister_high_volatility_cross_venue_flow_disagreement_resolution as p
RESULT=Path('results/high_volatility_cross_venue_flow_disagreement_resolution_support_2026-08-13.json');EXPECTED='68af872195648463bfc4d23fe61e17862b59aa4b465730c4e96e053d612d056c'
def test_terminal_source_rejection_is_immutable_and_blind():
 assert hashlib.sha256(RESULT.read_bytes()).hexdigest()==EXPECTED;x=json.loads(RESULT.read_text());h=x.pop('manifest_hash');assert p.canonical_hash(x)==h;assert x['policy_id']=='HVFDR-8' and not x['support_passed'] and x['decision']=='terminal_source_support_reject';assert not x['advance_to_gross9_novelty'] and not x['advance_to_economic_outcomes'];assert not x['postentry_return_pnl_execution_price_opened'] and not x['gross9_rows_opened'];assert all(v['events']==0 for v in x['support'].values())
