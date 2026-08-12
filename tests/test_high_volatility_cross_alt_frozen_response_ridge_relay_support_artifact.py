import hashlib,json
from pathlib import Path
from training import preregister_high_volatility_cross_alt_frozen_response_ridge_relay as p
P=Path('results/high_volatility_cross_alt_frozen_response_ridge_relay_support_2026-08-13.json');EXPECTED='51016c26ff6f796006159a9957cbbad6f4e34cea0282a22ef9f0b74bdfa42864'
def test_source_rejection_is_immutable_oos_blind_and_terminal():
 assert hashlib.sha256(P.read_bytes()).hexdigest()==EXPECTED;r=json.loads(P.read_text());h=r.pop('manifest_hash');assert p.canonical_hash(r)==h;assert r['policy_id']=='HVCAFRR-8' and not r['support_passed'] and r['decision']=='terminal_source_support_reject';assert r['support']['train']['minority_side_share']<.2 and r['support']['test']['minority_side_share']<.2 and r['support']['final']['minority_side_share']<.2;assert not r['advance_to_gross9_novelty'] and not r['advance_to_economic_outcomes'];assert r['calibration_labels_opened'] and not r['oos_postentry_return_pnl_execution_price_opened'] and not r['oos_funding_values_opened'] and not r['gross9_rows_opened']
