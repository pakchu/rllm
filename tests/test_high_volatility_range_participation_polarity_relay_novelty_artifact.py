import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_range_participation_polarity_relay_gross9_novelty_2026-08-10.json")
def test_novelty_pass_frozen_blind_complete():
 x=json.loads(R.read_text());assert x["every_gross9_sleeve_passed"] is True and x["advance_to_economic_outcomes"] is True and x["evidence_boundary"]["outcomes_opened"] is False and x["evidence_boundary"]["funding_rows_opened"]==0 and set(x["gross9_sleeves"])==set(x["gross9_structural_clocks"]["complete_roster"]) and all(v["passed"] for v in x["gross9_sleeves"].values());assert hashlib.sha256(R.read_bytes()).hexdigest()=="490c9190479b47896172edceb1892135604e0cf6ff82e870106285b409320432"
