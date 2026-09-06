import hashlib,json
from pathlib import Path
RESULT=Path("results/high_volatility_taker_imbalance_flow_persistence_gross9_novelty_2026-08-13.json")
def test_novelty_pass_is_reproducible_and_blind():
 x=json.loads(RESULT.read_text());assert hashlib.sha256(RESULT.read_bytes()).hexdigest()=="0220610bdec1fe05b1eeb17727f060b2b619e5706857db13df54962ef9b812f4";assert x["policy_id"]=="HVTIFP-8" and x["source_support_passed"] and x["every_gross9_sleeve_passed"] and x["advance_to_economic_outcomes"];assert not x["evidence_boundary"]["outcomes_opened"] and x["evidence_boundary"]["economic_outcome_rows_opened"]==0
def test_every_sleeve_passes_limits():
 x=json.loads(RESULT.read_text());assert all(v["passed"] and all(v["checks"].values()) for v in x["gross9_sleeves"].values());m=[v["metrics"] for v in x["gross9_sleeves"].values()];assert max(v["exact_entry_jaccard"] for v in m)==0 and max(v["one_to_one_6h_max_matched_share"] for v in m)<.118 and max(v["occupied_5m_bar_jaccard"] for v in m)<.049 and max(v["absolute_signed_exposure_pearson"] for v in m)<.041
