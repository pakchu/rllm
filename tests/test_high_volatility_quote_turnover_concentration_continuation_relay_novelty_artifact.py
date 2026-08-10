import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_quote_turnover_concentration_continuation_relay_gross9_novelty_2026-08-10.json")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_hvtccr_novelty_pass_is_frozen_and_outcome_blind():
 x=json.loads(R.read_text());assert x["every_gross9_sleeve_passed"] is True and x["gross9_novelty_status"]=="passed" and x["advance_to_economic_outcomes"] is True
 assert x["evidence_boundary"]["outcomes_opened"] is False and x["evidence_boundary"]["btc_execution_rows_opened"]==0 and x["evidence_boundary"]["funding_rows_opened"]==0
 assert max(v["metrics"]["exact_entry_jaccard"] for v in x["gross9_sleeves"].values())<=.1
 assert max(v["metrics"]["one_to_one_6h_max_matched_share"] for v in x["gross9_sleeves"].values())<=.35
 assert max(v["metrics"]["occupied_5m_bar_jaccard"] for v in x["gross9_sleeves"].values())<=.25
 assert max(v["metrics"]["absolute_signed_exposure_pearson"] for v in x["gross9_sleeves"].values())<=.35
 assert sha(R)=="910993f4a70ba770a9e66273630e14ca4df175864ba8f7009cb388c31fd8c28f"
