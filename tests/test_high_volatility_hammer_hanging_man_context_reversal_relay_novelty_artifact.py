import hashlib,json
from pathlib import Path
RESULT=Path("results/high_volatility_hammer_hanging_man_context_reversal_relay_gross9_novelty_2026-08-11.json")
def test_novelty_pass_is_frozen_blind():
 r=json.loads(RESULT.read_text());assert r["every_gross9_sleeve_passed"] is True and r["advance_to_economic_outcomes"] is True and r["evidence_boundary"]["outcomes_opened"] is False
 m=[v["metrics"] for v in r["gross9_sleeves"].values()];assert max(v["exact_entry_jaccard"] for v in m)<=.1 and max(v["one_to_one_6h_max_matched_share"] for v in m)<=.35 and max(v["occupied_5m_bar_jaccard"] for v in m)<=.25 and max(v["absolute_signed_exposure_pearson"] for v in m)<=.35
 assert hashlib.sha256(RESULT.read_bytes()).hexdigest()=="e98e1642110fd3dc8de69c6ca65a0fc7051db298e127a625ebdbf0bb14d49dd5"
