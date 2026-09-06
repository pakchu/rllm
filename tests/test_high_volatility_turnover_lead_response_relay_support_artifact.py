import hashlib,json
from pathlib import Path
R=Path('results/high_volatility_turnover_lead_response_relay_support_2026-08-10.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_hvtlrr_support_pass_is_frozen_blind_reproducible():
 x=json.loads(R.read_text());assert x['support_passed'] is True and x['advance_to_gross9_novelty'] is True and x['advance_to_economic_outcomes'] is False and x['postentry_return_pnl_execution_price_opened'] is False and x['funding_values_opened'] is False and x['gross9_rows_opened'] is False
 assert {k:v['events'] for k,v in x['support'].items()}=={'train':15,'test':48,'eval':44,'final':27};assert x['clock']['rows']==134 and sha(Path(x['clock']['path']))==x['clock']['sha256']=='392f972c3df7e0e9aaaa447729066235c0c5c50ae403e42fae6126a9dedb4e74';assert sha(R)=='c7cd0be2e2c20958181138ec92429d52968a80035c9f1d841384c5d5740bf493'
