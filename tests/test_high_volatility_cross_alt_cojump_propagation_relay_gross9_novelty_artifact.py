import hashlib,json
from pathlib import Path
RESULT=Path("results/high_volatility_cross_alt_cojump_propagation_relay_gross9_novelty_2026-08-11.json")
def test_novelty_pass_artifact():
 x=json.loads(RESULT.read_text());assert x["gross9_novelty_status"]=="passed" and x["every_gross9_sleeve_passed"] is True and x["advance_to_economic_outcomes"] is True
 assert x["evidence_boundary"]["hvcacjp_clock_rows_opened"]==424 and x["evidence_boundary"]["economic_outcome_rows_opened"]==0 and x["evidence_boundary"]["outcomes_opened"] is False
 assert all(v["passed"] and all(v["checks"].values()) for v in x["gross9_sleeves"].values())
 assert hashlib.sha256(RESULT.read_bytes()).hexdigest()=="83afa31b0bb45f61063e8014426b3c0f26b8552aebb2c5aff7b8ee7908a8cfa5"
def test_economics_sealed():
 for stage in ("train","test","eval","final"):assert not Path(f"results/high_volatility_cross_alt_cojump_propagation_relay_{stage}_economics_2026-08-11.json").exists()
