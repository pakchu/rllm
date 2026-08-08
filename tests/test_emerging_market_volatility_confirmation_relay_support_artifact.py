import json
from training import build_emerging_market_volatility_confirmation_relay_support as support
def test_emvcr_source_failure_is_terminal_and_outcome_sealed():
 r=json.loads(support.RESULT.read_text());assert r["policy_id"]=="EMVCR-8";assert r["support_passed"] is False;assert r["decision"]=="terminal_source_support_reject";assert r["advance_to_gross9_novelty"] is False;assert r["advance_to_economic_outcomes"] is False;assert r["postentry_return_pnl_execution_price_opened"] is False;assert r["gross9_rows_opened"] is False;assert [r["support"][s]["events"] for s in ("train","test","eval","final")]==[11,24,8,16]
def test_emvcr_failure_matches_frozen_gates():
 r=json.loads(support.RESULT.read_text());assert r["support_checks"]["train_side_balance"] is False;assert r["support_checks"]["eval_minimum_events"] is False;assert r["support"]["train"]["minority_side_share"]<.2
