import hashlib,json
from pathlib import Path
from training import preregister_high_volatility_daily_cross_alt_flow_consensus_relay as p
RESULT=Path('results/high_volatility_daily_cross_alt_flow_consensus_relay_support_2026-08-13.json');EXPECTED='e271aa75d1d9d1925d28901c6b61b2cc3c4dbd765b2933f371ab8eddba1f646e'
def test_source_rejection_is_immutable_blind_and_terminal():
 assert hashlib.sha256(RESULT.read_bytes()).hexdigest()==EXPECTED;r=json.loads(RESULT.read_text());h=r.pop('manifest_hash');assert p.canonical_hash(r)==h;assert r['policy_id']=='HVDCAFC-12' and r['support_passed'] is False and r['decision']=='terminal_source_support_reject';assert all(r['support'][s]['minority_side_share']==0 for s in ('train','test','eval'));assert not r['advance_to_gross9_novelty'] and not r['advance_to_economic_outcomes'] and not r['postentry_return_pnl_execution_price_opened'] and not r['funding_values_opened'] and not r['gross9_rows_opened']
