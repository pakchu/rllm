import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_eth_relative_variation_risk_relay_gross9_novelty_2026-08-10.json")
def test_pass_blind():
 x=json.loads(R.read_text());assert x["every_gross9_sleeve_passed"] is True and x["advance_to_economic_outcomes"] is True and x["evidence_boundary"]["outcomes_opened"] is False;assert hashlib.sha256(R.read_bytes()).hexdigest()=="0e8058616fb206db5be6d191dac8515d760351a2a6da1b39317e185a3acdd73c"
