import hashlib,json
from pathlib import Path
R=Path('results/high_volatility_turnover_lead_response_relay_gross9_novelty_2026-08-10.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_hvtlrr_novelty_pass_is_frozen_blind():
 x=json.loads(R.read_text());assert x['every_gross9_sleeve_passed'] is True and x['advance_to_economic_outcomes'] is True and x['evidence_boundary']['outcomes_opened'] is False
 assert max(v['metrics']['one_to_one_6h_max_matched_share'] for v in x['gross9_sleeves'].values())<=.35 and max(v['metrics']['occupied_5m_bar_jaccard'] for v in x['gross9_sleeves'].values())<=.25 and max(v['metrics']['absolute_signed_exposure_pearson'] for v in x['gross9_sleeves'].values())<=.35
 assert sha(R)=='a247a4ffb47437d8f1a67ce8bdc66f0c2487d7bc6ec8d15860f48ab33c98fdda'
