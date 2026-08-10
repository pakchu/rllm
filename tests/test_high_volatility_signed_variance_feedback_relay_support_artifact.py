import hashlib,json
from pathlib import Path
R=Path('results/high_volatility_signed_variance_feedback_relay_support_2026-08-10.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_support_pass_is_frozen_and_blind():
 p=json.loads(R.read_text());assert p['support_passed'] is True and p['advance_to_gross9_novelty'] is True and p['advance_to_economic_outcomes'] is False and p['postentry_return_pnl_execution_price_opened'] is False and p['funding_values_opened'] is False and p['gross9_rows_opened'] is False
 assert {k:v['events'] for k,v in p['support'].items()}=={'train':55,'test':87,'eval':92,'final':39};assert p['clock']['rows']==273 and sha(Path(p['clock']['path']))==p['clock']['sha256']=='4b584277f27fc8791f7ca4bc3902099b477d97f69cf9d7d36ab2e414ae6b287d';assert sha(R)=='e0566e354f2ade805201db19d6c81a7c717447937ab993a962ad9954cc1f471b'
