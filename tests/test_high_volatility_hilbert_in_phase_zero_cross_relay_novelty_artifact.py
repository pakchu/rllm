import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_hilbert_in_phase_zero_cross_relay_gross9_novelty_2026-08-11.json")
def test_novelty_pass_is_blind_and_frozen():
 x=json.loads(R.read_text());assert x["every_gross9_sleeve_passed"] is True and x["advance_to_economic_outcomes"] is True and x["evidence_boundary"]["outcomes_opened"] is False
 m=[v["metrics"] for v in x["gross9_sleeves"].values()];assert max(v["exact_entry_jaccard"] for v in m)<=.1 and max(v["one_to_one_6h_max_matched_share"] for v in m)<=.35 and max(v["occupied_5m_bar_jaccard"] for v in m)<=.25 and max(v["absolute_signed_exposure_pearson"] for v in m)<=.35
 assert hashlib.sha256(R.read_bytes()).hexdigest()=="5385215a36259f63bdc18fce7601b64907e3fee8b9afc8bfd37bd0c91e993415"
