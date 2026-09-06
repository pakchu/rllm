import hashlib,json
from pathlib import Path
RESULT=Path("results/high_volatility_leverage_to_cash_flow_handoff_relay_support_2026-08-13.json")
def test_terminal_before_novelty_and_outcomes():
 x=json.loads(RESULT.read_text());assert x["support_passed"] is False;assert x["advance_to_gross9_novelty"] is False;assert x["advance_to_economic_outcomes"] is False;assert all(v["events"]==0 for v in x["support"].values());assert x["gross9_rows_opened"] is False;assert x["postentry_return_pnl_execution_price_opened"] is False;assert hashlib.sha256(RESULT.read_bytes()).hexdigest()=="345646b13580c27fb65634e7c21be6d7226e1355f3f2c1aef5b4bf39724fd6e4"
