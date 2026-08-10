import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_quarter_hour_order_flow_relay_support_2026-08-11.json")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_hvqhofr_support_pass_is_frozen_blind_reproducible():
 x=json.loads(R.read_text());assert x["support_passed"] is True and x["advance_to_gross9_novelty"] is True and x["advance_to_economic_outcomes"] is False
 assert x["postentry_return_pnl_execution_price_opened"] is False and x["funding_values_opened"] is False and x["gross9_rows_opened"] is False
 assert {k:v["events"] for k,v in x["support"].items()}=={"train":175,"test":351,"eval":354,"final":171} and all(x["support_checks"].values())
 assert x["clock"]["rows"]==1051 and sha(Path(x["clock"]["path"]))==x["clock"]["sha256"]=="46bdb23aad37c6c80fd371b0439644d1acc5a9cb6b2843fd319bb518f08d3a29"
 assert sha(R)=="7fed9e4547f1e7e207904ce76c99b3545de9a2c5f3aa36adc2b04e7341c054bd"
