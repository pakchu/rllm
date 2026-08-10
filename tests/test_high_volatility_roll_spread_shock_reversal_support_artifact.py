import hashlib,json
from pathlib import Path
RESULT=Path('results/high_volatility_roll_spread_shock_reversal_support_2026-08-10.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_support_passes_outcome_blind_and_hashes_are_frozen():
 p=json.loads(RESULT.read_text());assert p['support_passed'] is True and p['advance_to_gross9_novelty'] is True and p['advance_to_economic_outcomes'] is False
 assert p['postentry_return_pnl_execution_price_opened'] is False and p['funding_values_opened'] is False and p['gross9_rows_opened'] is False
 assert {k:v['events'] for k,v in p['support'].items()}=={'train':40,'test':82,'eval':67,'final':36}
 assert p['clock']['rows']==225 and sha(Path(p['clock']['path']))==p['clock']['sha256']=='d1e5c102e486abd9dd41ce42b822113ced3103d3eb892008a97b201f5ba1574f'
 assert sha(RESULT)=='2f531003ff623c316d70edc8b4bd9b20d03be3916b6d5b4eddbe4dd5f19f2cd1'
