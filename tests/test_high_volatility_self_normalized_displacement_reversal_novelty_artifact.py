import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_self_normalized_displacement_reversal_gross9_novelty_2026-08-10.json")
def test_pass_blind():
 x=json.loads(R.read_text());assert x["every_gross9_sleeve_passed"] is True and x["advance_to_economic_outcomes"] is True and x["evidence_boundary"]["outcomes_opened"] is False;assert hashlib.sha256(R.read_bytes()).hexdigest()=="5ebcef895bb41ebb77e95523089e2c0e3aa08fc8fff8228a231ca3f7f3298204"
