import hashlib,json
from pathlib import Path
RESULT=Path("results/high_volatility_wide_range_body_continuation_relay_gross9_novelty_2026-08-11.json")
def sha256(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def test_hvwrbc_novelty_pass_is_frozen_blind():
 report=json.loads(RESULT.read_text());assert report["every_gross9_sleeve_passed"] is True and report["advance_to_economic_outcomes"] is True and report["evidence_boundary"]["outcomes_opened"] is False
 metrics=[v["metrics"] for v in report["gross9_sleeves"].values()];assert max(v["exact_entry_jaccard"] for v in metrics)<=.1 and max(v["one_to_one_6h_max_matched_share"] for v in metrics)<=.35 and max(v["occupied_5m_bar_jaccard"] for v in metrics)<=.25 and max(v["absolute_signed_exposure_pearson"] for v in metrics)<=.35
 assert sha256(RESULT)=="5a8a794d693409a7e62bb9bf2f3b557a9438b986cd714abdc2c584eb842fa4db"
