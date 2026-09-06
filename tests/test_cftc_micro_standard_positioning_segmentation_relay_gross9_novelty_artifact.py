import hashlib,json
from pathlib import Path
R=Path("results/cftc_micro_standard_positioning_segmentation_relay_gross9_novelty_2026-08-11.json")
def test_frozen_novelty_pass():
 assert hashlib.sha256(R.read_bytes()).hexdigest()=="e6085d5ad2a376e79fd97e3f24a10172d8add445ca103def34a8ea7a3cd3123e";x=json.loads(R.read_text());assert x["gross9_novelty_status"]=="passed" and x["every_gross9_sleeve_passed"] is True and x["advance_to_economic_outcomes"] is True;assert x["evidence_boundary"]["outcomes_opened"] is False and x["evidence_boundary"]["economic_outcome_rows_opened"]==0;assert max(v["metrics"]["one_to_one_6h_max_matched_share"] for v in x["gross9_sleeves"].values())<.35
