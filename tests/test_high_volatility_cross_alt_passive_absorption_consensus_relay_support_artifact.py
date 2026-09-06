import hashlib,json
from pathlib import Path
from training import preregister_high_volatility_cross_alt_passive_absorption_consensus_relay as p
P=Path('results/high_volatility_cross_alt_passive_absorption_consensus_relay_support_2026-08-13.json');EXPECTED='75e95ffec61e3afc04c0ee8250a174a22cb8a1450cc84b91ff89659ce6be2ad7'
def test_source_rejection_is_immutable_blind_and_terminal():
 assert hashlib.sha256(P.read_bytes()).hexdigest()==EXPECTED;r=json.loads(P.read_text());h=r.pop('manifest_hash');assert p.canonical_hash(r)==h;assert r['policy_id']=='HVCAPAC-8' and not r['support_passed'] and r['decision']=='terminal_source_support_reject';assert sum(x['events'] for x in r['support'].values())==10 and sum(x['shorts'] for x in r['support'].values())==0;assert not r['advance_to_gross9_novelty'] and not r['advance_to_economic_outcomes'];assert not r['postentry_return_pnl_execution_price_opened'] and not r['funding_values_opened'] and not r['gross9_rows_opened']
