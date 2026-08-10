import hashlib,json
from pathlib import Path
RESULT=Path("results/high_volatility_wilder_rsi_reversal_support_2026-08-11.json")
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def test_hvwrsi_source_rejection_is_terminal():
 x=json.loads(RESULT.read_text());assert sha(RESULT)=="5fb6d57db7f3b291c6d1f93b56356d402ef57f412df7b913529ba1536284d107"
 assert x["policy_id"]=="HVWRSI-24" and x["support_passed"] is False and x["advance_to_gross9_novelty"] is False and x["decision"]=="terminal_source_support_reject"
 assert x["support"]["train"]=={"events":28,"longs":0,"shorts":28,"minority_side_share":0.0,"max_month_share":12/28}
 assert x["support"]["eval"]["events"]==6 and x["support"]["final"]["shorts"]==0
 assert x["postentry_return_pnl_execution_price_opened"] is False and x["funding_values_opened"] is False and x["gross9_rows_opened"] is False
 assert not Path("results/high_volatility_wilder_rsi_reversal_gross9_novelty_2026-08-11.json").exists()
