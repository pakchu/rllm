import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_eth_relative_variation_risk_relay_train_economics_2026-08-10.json")
def test_terminal_reject():
 x=json.loads(R.read_text());assert x["stage"]=="train" and x["passed"] is False and x["advance_to_next_stage"] is False and x["later_stage_outcomes_opened"] is False and x["decision"]=="terminal_reject_no_repair";assert x["primary"]["base"]["absolute_return_pct"]==-6.228932580891877 and x["primary"]["cluster_signflip"]["pvalue"]==.8699113008869911;assert hashlib.sha256(R.read_bytes()).hexdigest()=="b16b64440de693dad2097efcc9ed01a931fd741d2cf218f04b86de491aa90f65"
