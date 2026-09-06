import hashlib,json
from pathlib import Path
R=Path('results/high_volatility_roll_spread_shock_reversal_gross9_novelty_2026-08-10.json')
def test_novelty_pass_is_outcome_blind():
 p=json.loads(R.read_text());assert p['every_gross9_sleeve_passed'] is True and p['advance_to_economic_outcomes'] is True
 assert p['evidence_boundary']['outcomes_opened'] is False and p['evidence_boundary']['funding_rows_opened']==0
 assert hashlib.sha256(R.read_bytes()).hexdigest()=='306a59dab9943196d0e3321ca5be50b216ccef4c9c7347b1010cb9897ffe4cff'
