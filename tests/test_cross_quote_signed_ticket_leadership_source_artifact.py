import hashlib,json
from training import build_cross_quote_signed_ticket_leadership_support as support

def test_cqstl_source_rejection_is_terminal_and_outcome_sealed():
 assert hashlib.sha256(support.RESULT.read_bytes()).hexdigest()=='698db3c5ff28eb1477e9055136a1044256fe630b6f89d01d8b936d7b7c2d0d5c';r=json.loads(support.RESULT.read_text());assert r['policy_id']=='CQSTL-8';assert r['support_passed'] is False;assert r['decision']=='terminal_source_support_reject';assert r['advance_to_gross9_novelty'] is False;assert r['postentry_return_pnl_execution_price_opened'] is False;assert r['gross9_rows_opened'] is False

def test_cqstl_only_failure_is_train_side_balance():
 r=json.loads(support.RESULT.read_text());assert r['support']['train']=={'events':18,'longs':3,'shorts':15,'minority_side_share':1/6,'max_month_share':1/3};assert sum(r['support_checks'].values())==11;assert r['support_checks']['train_side_balance'] is False
