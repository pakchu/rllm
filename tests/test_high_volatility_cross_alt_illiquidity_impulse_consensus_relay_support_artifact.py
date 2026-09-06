import hashlib,json
from pathlib import Path
from training import preregister_high_volatility_cross_alt_illiquidity_impulse_consensus_relay as p
RESULT=Path('results/high_volatility_cross_alt_illiquidity_impulse_consensus_relay_support_2026-08-13.json');EXPECTED='7a0474f5df4af997ab1dec6c891420bc6697acffc2200403845a4aed0cb3c1f7'
def test_source_pass_is_immutable_blind_and_authorizes_only_novelty():
 assert hashlib.sha256(RESULT.read_bytes()).hexdigest()==EXPECTED;r=json.loads(RESULT.read_text());h=r.pop('manifest_hash');assert p.canonical_hash(r)==h;assert r['policy_id']=='HVCIIC-8' and r['support_passed'] is True and r['decision']=='pass_to_novelty';assert r['advance_to_gross9_novelty'] and not r['advance_to_economic_outcomes'];assert not r['postentry_return_pnl_execution_price_opened'] and not r['funding_values_opened'] and not r['gross9_rows_opened'];assert {k:v['events'] for k,v in r['support'].items()}=={'train':19,'test':46,'eval':72,'final':16}
