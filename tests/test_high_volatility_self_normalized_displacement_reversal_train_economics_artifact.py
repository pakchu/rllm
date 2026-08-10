import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_self_normalized_displacement_reversal_train_economics_2026-08-10.json")
def test_terminal_reject():
 x=json.loads(R.read_text());assert x["stage"]=="train" and x["passed"] is False and x["advance_to_next_stage"] is False and x["later_stage_outcomes_opened"] is False and x["decision"]=="terminal_reject_no_repair";assert x["primary"]["base"]["absolute_return_pct"]==-2.8924192874867583 and x["primary"]["cluster_signflip"]["pvalue"]==.6529934700652994;assert hashlib.sha256(R.read_bytes()).hexdigest()=="ed57341fe678d3b121710367a2089d36ad24abb65f1fd5064d08c3e9a692f52c"
