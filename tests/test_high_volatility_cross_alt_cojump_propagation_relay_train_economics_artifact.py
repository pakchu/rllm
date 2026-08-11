import hashlib,json
from pathlib import Path
RESULT=Path("results/high_volatility_cross_alt_cojump_propagation_relay_train_economics_2026-08-11.json")
def test_terminal_train_rejection_artifact():
 x=json.loads(RESULT.read_text());assert x["stage"]=="train" and x["passed"] is False and x["decision"]=="terminal_reject_no_repair"
 assert x["primary"]["base"]["absolute_return_pct"]==-2.4029944452313656 and x["primary"]["base"]["mean_gross_underlying_bp"]==4.476515898697564
 assert x["primary"]["cluster_signflip"]["pvalue"]==0.6199638003619964 and x["checks"]["each_calendar_half_positive"] is False and x["later_stage_outcomes_opened"] is False
 assert hashlib.sha256(RESULT.read_bytes()).hexdigest()=="1b432798b134ade4399a58e06d08158413bf1f0bd4679c2277e4524183e84f98"
def test_later_stages_never_opened():
 for stage in ("test","eval","final"):assert not Path(f"results/high_volatility_cross_alt_cojump_propagation_relay_{stage}_economics_2026-08-11.json").exists()
