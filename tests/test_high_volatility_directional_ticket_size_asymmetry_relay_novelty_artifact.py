import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_directional_ticket_size_asymmetry_relay_gross9_novelty_2026-08-10.json")
def test_pass_blind():
 x=json.loads(R.read_text());assert x["every_gross9_sleeve_passed"] is True and x["advance_to_economic_outcomes"] is True and x["evidence_boundary"]["outcomes_opened"] is False;assert hashlib.sha256(R.read_bytes()).hexdigest()=="f4e3d5cab4a98371b343266e4cdba4c209fb20163cb677d4d97c9674790b659c"
