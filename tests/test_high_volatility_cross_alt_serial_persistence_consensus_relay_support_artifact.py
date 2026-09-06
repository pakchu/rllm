import hashlib,json
from pathlib import Path
from training import preregister_high_volatility_cross_alt_serial_persistence_consensus_relay as p
P=Path('results/high_volatility_cross_alt_serial_persistence_consensus_relay_support_2026-08-13.json');EXPECTED='53f401d5ab9a499838c128b7023b4e420925455ca7e858e5a4db5c6e4f83d52e'
def test_source_pass_is_immutable_blind_and_authorizes_only_novelty():
 assert hashlib.sha256(P.read_bytes()).hexdigest()==EXPECTED;r=json.loads(P.read_text());h=r.pop('manifest_hash');assert p.canonical_hash(r)==h;assert r['policy_id']=='HVCASPC-8' and r['support_passed'] and r['decision']=='pass_to_novelty';assert r['advance_to_gross9_novelty'] and not r['advance_to_economic_outcomes'];assert not r['postentry_return_pnl_execution_price_opened'] and not r['funding_values_opened'] and not r['gross9_rows_opened'];assert {k:v['events'] for k,v in r['support'].items()}=={'train':20,'test':53,'eval':51,'final':29}
