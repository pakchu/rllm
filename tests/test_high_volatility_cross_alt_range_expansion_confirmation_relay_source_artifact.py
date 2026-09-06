import hashlib,json
from pathlib import Path
RESULT=Path("results/high_volatility_cross_alt_range_expansion_confirmation_relay_support_2026-08-11.json")
def test_source_support_pass_artifact():
 x=json.loads(RESULT.read_text());assert x["support_passed"] is True and x["decision"]=="pass_to_novelty" and x["advance_to_economic_outcomes"] is False
 assert {k:v["events"] for k,v in x["support"].items()}=={"train":35,"test":61,"eval":57,"final":24}
 assert all(x["support_checks"].values()) and x["clock"]["rows"]==177
 assert x["postentry_return_pnl_execution_price_opened"] is False and x["funding_values_opened"] is False and x["gross9_rows_opened"] is False
 assert hashlib.sha256(RESULT.read_bytes()).hexdigest()=="81311ea40401c5625e09428ad57894953cd6dcae527d07cdac766324c41df8af"
def test_economics_sealed():
 for stage in ("train","test","eval","final"):assert not Path(f"results/high_volatility_cross_alt_range_expansion_confirmation_relay_{stage}_economics_2026-08-11.json").exists()
