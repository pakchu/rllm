import hashlib,json
from pathlib import Path
RESULT=Path("results/high_volatility_cross_alt_range_expansion_confirmation_relay_train_economics_2026-08-11.json")
def test_terminal_train_rejection_artifact():
 x=json.loads(RESULT.read_text());assert x["stage"]=="train" and x["passed"] is False and x["decision"]=="terminal_reject_no_repair"
 assert x["primary"]["base"]["absolute_return_pct"]==-5.8656989622919316 and x["primary"]["base"]["mean_gross_underlying_bp"]==-22.05366281508888
 assert x["primary"]["cluster_signflip"]["pvalue"]==0.9578804211957881 and x["checks"]["absolute_return_positive"] is False
 assert x["later_stage_outcomes_opened"] is False and hashlib.sha256(RESULT.read_bytes()).hexdigest()=="22d4b440c2e95fd8cbeab518dd4a259cdc7cfaf3a50f8f5622ce825b63254fbe"
def test_later_stages_never_opened():
 for stage in ("test","eval","final"):assert not Path(f"results/high_volatility_cross_alt_range_expansion_confirmation_relay_{stage}_economics_2026-08-11.json").exists()
