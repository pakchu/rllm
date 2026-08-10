import hashlib,json
from pathlib import Path
RESULT=Path("results/high_volatility_closing_drive_breakout_relay_gross9_novelty_2026-08-11.json")
def sha256(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def test_hvcdbr_novelty_pass_is_frozen_blind():
 report=json.loads(RESULT.read_text());assert report["every_gross9_sleeve_passed"] is True and report["advance_to_economic_outcomes"] is True and report["evidence_boundary"]["outcomes_opened"] is False
 metrics=[v["metrics"] for v in report["gross9_sleeves"].values()];assert max(v["exact_entry_jaccard"] for v in metrics)<=.1 and max(v["one_to_one_6h_max_matched_share"] for v in metrics)<=.35 and max(v["occupied_5m_bar_jaccard"] for v in metrics)<=.25 and max(v["absolute_signed_exposure_pearson"] for v in metrics)<=.35
 assert sha256(RESULT)=="0bea13bc9371bf1d56c552ab6bcc5e9d4e54f1ccdff72344ee864d71bd966284"
