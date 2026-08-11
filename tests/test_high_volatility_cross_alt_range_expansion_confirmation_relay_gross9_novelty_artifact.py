import hashlib,json
from pathlib import Path
RESULT=Path("results/high_volatility_cross_alt_range_expansion_confirmation_relay_gross9_novelty_2026-08-11.json")
def test_novelty_pass_artifact():
 x=json.loads(RESULT.read_text());assert x["gross9_novelty_status"]=="passed" and x["every_gross9_sleeve_passed"] is True and x["advance_to_economic_outcomes"] is True
 assert x["evidence_boundary"]["hvcarer_clock_rows_opened"]==177 and x["evidence_boundary"]["economic_outcome_rows_opened"]==0 and x["evidence_boundary"]["outcomes_opened"] is False
 assert all(v["passed"] and all(v["checks"].values()) for v in x["gross9_sleeves"].values())
 assert hashlib.sha256(RESULT.read_bytes()).hexdigest()=="ef36829d249400c52570123e5fe7e078276ecee3e89983b051640341e63ef8a3"
def test_economics_still_sealed():
 for stage in ("train","test","eval","final"):assert not Path(f"results/high_volatility_cross_alt_range_expansion_confirmation_relay_{stage}_economics_2026-08-11.json").exists()
