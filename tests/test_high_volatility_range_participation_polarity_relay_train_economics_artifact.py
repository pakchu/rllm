import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_range_participation_polarity_relay_train_economics_2026-08-10.json")
def test_terminal_train_reject_frozen_later_stages_sealed():
 x=json.loads(R.read_text());assert x["stage"]=="train" and x["passed"] is False and x["advance_to_next_stage"] is False and x["later_stage_outcomes_opened"] is False and x["decision"]=="terminal_reject_no_repair";assert x["primary"]["base"]["absolute_return_pct"]==-5.842661041257169 and x["primary"]["base"]["mean_gross_underlying_bp"]==-6.785440227628521 and x["primary"]["cluster_signflip"]["pvalue"]==.938430615693843 and x["primary"]["stress"]["absolute_return_pct"]==-8.226841887724545;assert hashlib.sha256(R.read_bytes()).hexdigest()=="f1dcc5c129d70e690cbeaedb073fb44406b2a31ede073dc3cbaac34ac7c3a3b2"
