import hashlib,json
from pathlib import Path
from training import preregister_high_volatility_funding_acceleration_divergence_relay as p
P=Path('results/high_volatility_funding_acceleration_divergence_relay_support_2026-08-13.json');EXPECTED='0358bc42b196edf03463b4d29dbdc205136a1e5010483562c68c8e57575fbd07'
def test_source_rejection_is_immutable_blind_and_terminal():
 assert hashlib.sha256(P.read_bytes()).hexdigest()==EXPECTED;r=json.loads(P.read_text());h=r.pop('manifest_hash');assert p.canonical_hash(r)==h;assert r['policy_id']=='HVFADR-8' and not r['support_passed'] and r['decision']=='terminal_source_support_reject';assert r['support']['final']['events']==11 and r['support']['final']['max_month_share']>.45;assert not r['advance_to_gross9_novelty'] and not r['advance_to_economic_outcomes'];assert not r['postentry_return_pnl_execution_price_opened'] and not r['held_interval_funding_values_opened'] and not r['gross9_rows_opened']
