import hashlib,json
from pathlib import Path
RESULT=Path("results/high_volatility_directional_excursion_area_dominance_relay_gross9_novelty_2026-08-10.json")
def sha256(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def test_hvdeadr_novelty_pass_is_frozen_blind():
 report=json.loads(RESULT.read_text());assert report["every_gross9_sleeve_passed"] is True and report["advance_to_economic_outcomes"] is True and report["evidence_boundary"]["outcomes_opened"] is False
 metrics=[v["metrics"] for v in report["gross9_sleeves"].values()];assert max(v["exact_entry_jaccard"] for v in metrics)<=.1 and max(v["one_to_one_6h_max_matched_share"] for v in metrics)<=.35 and max(v["occupied_5m_bar_jaccard"] for v in metrics)<=.25 and max(v["absolute_signed_exposure_pearson"] for v in metrics)<=.35
 assert sha256(RESULT)=="724c7def568e3d21901d8d1e3a97c5ce915876d689d4530827e7abb0b22b51d9"
