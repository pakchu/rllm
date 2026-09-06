import hashlib,json
from training import build_spot_perpetual_variance_transfer_asymmetry_support as support

def test_spvta_source_rejection_is_terminal_and_outcome_sealed():
 assert hashlib.sha256(support.RESULT.read_bytes()).hexdigest()=='adf52135221f7c7d45469cad5586682612d53a16b70900a4620a6a2244722d43';r=json.loads(support.RESULT.read_text());assert r['policy_id']=='SPVTA-8';assert r['support_passed'] is False;assert r['decision']=='terminal_source_support_reject';assert r['advance_to_gross9_novelty'] is False;assert r['postentry_return_pnl_execution_price_opened'] is False;assert r['gross9_rows_opened'] is False

def test_spvta_event_counts_are_below_every_split_minimum():
 r=json.loads(support.RESULT.read_text());assert [r['support'][s]['events'] for s in ('train','test','eval','final')]==[3,2,1,3];assert all(r['support_checks'][f'{s}_minimum_events'] is False for s in ('train','test','eval','final'))
