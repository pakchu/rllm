import hashlib, json
from pathlib import Path
from training import evaluate_high_volatility_cross_alt_path_ordered_excursion_dominance_relay_gross9_novelty as n

RESULT=Path('results/high_volatility_cross_alt_path_ordered_excursion_dominance_relay_gross9_novelty_2026-08-13.json')
EXPECTED='e55bcd0c48f4c3d111859034c070a6343f5a09fec3cbf9c27e59ef8b4c5a4830'

def test_novelty_pass_is_immutable_and_economics_remained_sealed():
 assert hashlib.sha256(RESULT.read_bytes()).hexdigest()==EXPECTED
 x=json.loads(RESULT.read_text());h=x.pop('manifest_hash');assert n.canonical_hash(x)==h
 assert x['policy_id']=='HVCAPO-8' and x['every_gross9_sleeve_passed'] and x['advance_to_economic_outcomes']
 e=x['evidence_boundary'];assert e['btc_execution_rows_opened']==e['btc_price_or_return_rows_opened']==e['funding_rows_opened']==e['economic_outcome_rows_opened']==0 and not e['outcomes_opened']
 assert max(v['metrics']['exact_entry_jaccard'] for v in x['gross9_sleeves'].values())==0
 assert max(v['metrics']['one_to_one_6h_max_matched_share'] for v in x['gross9_sleeves'].values())<.242
 assert max(v['metrics']['occupied_5m_bar_jaccard'] for v in x['gross9_sleeves'].values())<.066
 assert max(v['metrics']['absolute_signed_exposure_pearson'] for v in x['gross9_sleeves'].values())<.089
