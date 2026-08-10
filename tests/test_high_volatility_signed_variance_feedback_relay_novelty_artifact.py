import hashlib,json
from pathlib import Path
R=Path('results/high_volatility_signed_variance_feedback_relay_gross9_novelty_2026-08-10.json')
def test_novelty_pass_is_frozen_and_blind():
 p=json.loads(R.read_text());assert p['every_gross9_sleeve_passed'] is True and p['advance_to_economic_outcomes'] is True and p['evidence_boundary']['outcomes_opened'] is False and p['evidence_boundary']['funding_rows_opened']==0;assert hashlib.sha256(R.read_bytes()).hexdigest()=='d6cffcf528c6d095123cba430706898473ac150ccff11f05ddac9882efee2b69'
