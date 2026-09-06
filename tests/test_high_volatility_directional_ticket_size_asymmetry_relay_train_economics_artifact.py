import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_directional_ticket_size_asymmetry_relay_train_economics_2026-08-10.json")
def test_terminal_reject():
 x=json.loads(R.read_text());assert x["stage"]=="train" and x["passed"] is False and x["advance_to_next_stage"] is False and x["later_stage_outcomes_opened"] is False and x["decision"]=="terminal_reject_no_repair";assert x["primary"]["base"]["absolute_return_pct"]==-6.194062330488381 and x["primary"]["cluster_signflip"]["pvalue"]==.9511304886951131;assert hashlib.sha256(R.read_bytes()).hexdigest()=="d16caa3a3140b458289a4eda1e4a5e9618d75963107748d503d69c046efd9fcb"
