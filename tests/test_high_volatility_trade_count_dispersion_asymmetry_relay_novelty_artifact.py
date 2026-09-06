import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_trade_count_dispersion_asymmetry_relay_gross9_novelty_2026-08-10.json")
def test_pass_blind():
 x=json.loads(R.read_text());assert x["every_gross9_sleeve_passed"] is True and x["advance_to_economic_outcomes"] is True and x["evidence_boundary"]["outcomes_opened"] is False;assert hashlib.sha256(R.read_bytes()).hexdigest()=="097ff65f200fc034ee7b1f330bf8a28418e5a8972e942bc7c6517a46f945865d"
