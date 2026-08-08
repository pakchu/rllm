import hashlib,json
from training import evaluate_deribit_expansion_partial_absorption_gross9_novelty as n
def test_depar_novelty_is_frozen_pass_before_economics():
 assert hashlib.sha256(n.OUTPUT.read_bytes()).hexdigest()=='ba5f01607d87369f274df96743f7528b3f4c07e1ace4bde2f7fc18411e62b654'
 d=json.loads(n.OUTPUT.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'}
 assert d['manifest_hash']==n.chash(core) and d['gross9_novelty_status']=='passed'
 assert d['every_gross9_sleeve_passed'] is True and d['advance_to_economic_outcomes'] is True
 assert d['evidence_boundary']['outcomes_opened'] is False
 assert max(x['metrics']['one_to_one_6h_max_matched_share'] for x in d['gross9_sleeves'].values())<.15
