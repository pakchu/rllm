import hashlib,json
from training import build_hkd_peg_pressure_rotation_relay_support as support

def test_hpprr_source_rejection_is_terminal_and_outcome_sealed():
 assert hashlib.sha256(support.RESULT.read_bytes()).hexdigest()=="45d089ee78fc19d991132ec217f21a4d616df705a21f392d82a555acc8cefcfc"
 r=json.loads(support.RESULT.read_text());assert r["policy_id"]=="HPPRR-12";assert r["support_passed"] is False;assert r["decision"]=="terminal_source_support_reject";assert r["advance_to_gross9_novelty"] is False;assert r["postentry_return_pnl_execution_price_opened"] is False;assert r["gross9_rows_opened"] is False

def test_hpprr_only_failure_is_final_month_concentration():
 r=json.loads(support.RESULT.read_text());assert r["support"]["final"]=={"events":10,"longs":2,"shorts":8,"minority_side_share":.2,"max_month_share":.5};assert sum(r["support_checks"].values())==11;assert r["support_checks"]["final_month_concentration"] is False
