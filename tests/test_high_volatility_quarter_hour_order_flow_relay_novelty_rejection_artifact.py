import hashlib,json
from pathlib import Path
RESULT=Path("results/high_volatility_quarter_hour_order_flow_relay_gross9_novelty_2026-08-11.json")
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def test_hvqhofr_novelty_rejection_is_terminal_and_outcomes_are_sealed():
 x=json.loads(RESULT.read_text());assert x["every_gross9_sleeve_passed"] is False and x["advance_to_economic_outcomes"] is False and x["gross9_novelty_status"]=="failed" and x["evidence_boundary"]["outcomes_opened"] is False
 assert all(v["checks"]["exact_entry_jaccard"] for v in x["gross9_sleeves"].values()) and all(v["checks"]["occupied_5m_bar_jaccard"] for v in x["gross9_sleeves"].values()) and all(v["checks"]["absolute_signed_exposure_pearson"] for v in x["gross9_sleeves"].values())
 assert all(not v["checks"]["one_to_one_6h_max_matched_share"] for v in x["gross9_sleeves"].values())
 assert max(v["metrics"]["one_to_one_6h_max_matched_share"] for v in x["gross9_sleeves"].values())==0.6470588235294118
 assert sha(RESULT)=="4ebc72649ff4843feae649fe5033e745f69bdc803d95426755e83948df88a06c"
 assert not Path("results/high_volatility_quarter_hour_order_flow_relay_train_economics_2026-08-11.json").exists()
