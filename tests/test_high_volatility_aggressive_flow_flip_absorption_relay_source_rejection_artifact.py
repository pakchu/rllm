import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_aggressive_flow_flip_absorption_relay_support_2026-08-11.json")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_hvffa_source_rejection_is_terminal_and_outcomes_are_sealed():
 x=json.loads(R.read_text());assert x["support_passed"] is False and x["advance_to_gross9_novelty"] is False and x["advance_to_economic_outcomes"] is False and x["decision"]=="terminal_source_support_reject"
 assert x["postentry_return_pnl_execution_price_opened"] is False and x["funding_values_opened"] is False and x["gross9_rows_opened"] is False
 assert {k:v["events"] for k,v in x["support"].items()}=={"train":22,"test":27,"eval":31,"final":9}
 assert x["support_checks"]["final_month_concentration"] is False and sum(not v for v in x["support_checks"].values())==1
 assert x["clock"]["rows"]==89 and sha(Path(x["clock"]["path"]))==x["clock"]["sha256"]=="f74e5e9ddd6962470375a8b1237733591122df2d202016b578fe94aeef0db599"
 assert sha(R)=="a61018033b3d1de4b9d198fb87af614b95b90d7ae351a6870cbe958ec21feaa0"
 assert not Path("results/high_volatility_aggressive_flow_flip_absorption_relay_gross9_novelty_2026-08-11.json").exists()
