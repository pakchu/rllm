import json
def test_dvbcr_support_artifact_is_terminal_before_outcomes():
 r=json.load(open("results/daily_variance_backloading_continuation_relay_support_2026-08-08.json"));assert r["decision"]=="terminal_source_support_reject";assert r["support_passed"] is False;assert r["advance_to_gross9_novelty"] is False;assert r["advance_to_economic_outcomes"] is False;assert r["postentry_return_pnl_execution_price_opened"] is False;assert r["gross9_rows_opened"] is False
