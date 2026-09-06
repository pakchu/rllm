import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_trade_count_dispersion_asymmetry_relay_train_economics_2026-08-10.json")
def test_terminal_reject():
 x=json.loads(R.read_text());assert x["stage"]=="train" and x["passed"] is False and x["advance_to_next_stage"] is False and x["later_stage_outcomes_opened"] is False and x["decision"]=="terminal_reject_no_repair";assert x["primary"]["base"]["absolute_return_pct"]==-2.6195494716249046 and x["primary"]["cluster_signflip"]["pvalue"]==.8493015069849301;assert hashlib.sha256(R.read_bytes()).hexdigest()=="b388b0e18d04f211d2ce25c89939eac75a0a2aeaa756b86303fdde297e4c95a6"
