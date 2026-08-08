import hashlib,json
from training import build_stablecoin_quote_count_breadth_impulse_support as support

def test_sqcbi_source_rejection_is_terminal_and_outcome_sealed():
 assert hashlib.sha256(support.RESULT.read_bytes()).hexdigest()=='b31737d78b4250663f39f50b4122e920c4096235d624d242c528f97e4e3be199';r=json.loads(support.RESULT.read_text());assert r['policy_id']=='SQCBI-8';assert r['support_passed'] is False;assert r['decision']=='terminal_source_support_reject';assert r['advance_to_gross9_novelty'] is False;assert r['postentry_return_pnl_execution_price_opened'] is False;assert r['gross9_rows_opened'] is False

def test_sqcbi_fails_train_support_and_month_concentration():
 r=json.loads(support.RESULT.read_text());assert [r['support'][s]['events'] for s in ('train','test','eval','final')]==[4,14,25,21];assert r['support_checks']['train_minimum_events'] is False;assert r['support_checks']['train_month_concentration'] is False;assert r['support_checks']['test_month_concentration'] is False
