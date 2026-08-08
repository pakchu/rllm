import json
from training import build_fear_greed_extremity_reversal_support as support
def test_fger_source_failure_is_terminal_and_outcome_sealed():
 r=json.loads(support.RESULT.read_text());assert r["policy_id"]=="FGER-24";assert r["support_passed"] is False;assert r["decision"]=="terminal_source_support_reject";assert r["advance_to_gross9_novelty"] is False;assert r["advance_to_economic_outcomes"] is False;assert r["postentry_return_pnl_execution_price_opened"] is False;assert r["gross9_rows_opened"] is False;assert [r["support"][s]["events"] for s in ("train","test","eval","final")]==[1,56,52,53]
def test_fger_fails_frozen_balance_and_train_support():
 r=json.loads(support.RESULT.read_text());assert r["support_checks"]["train_minimum_events"] is False;assert r["support_checks"]["train_side_balance"] is False;assert r["support_checks"]["test_side_balance"] is False;assert r["support_checks"]["eval_side_balance"] is False;assert r["support_checks"]["final_side_balance"] is False
