import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_multifractal_coherence_relay_train_economics_2026-08-10.json")
def test_terminal_train_reject_is_frozen_later_stages_sealed():
 x=json.loads(R.read_text());assert x["stage"]=="train" and x["passed"] is False and x["advance_to_next_stage"] is False and x["later_stage_outcomes_opened"] is False and x["decision"]=="terminal_reject_no_repair";assert x["primary"]["base"]["absolute_return_pct"]==-.38001636749163126 and x["primary"]["base"]["mean_gross_underlying_bp"]==10.48218291110828 and x["primary"]["cluster_signflip"]["pvalue"]==.5456845431545685 and x["primary"]["stress"]["absolute_return_pct"]==-2.081223664770593;assert hashlib.sha256(R.read_bytes()).hexdigest()=="92ff588032ff740d08c79666244708a76af6d067c2b0fde1b123eca486ee1a18"
