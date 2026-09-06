import hashlib,json
from training import evaluate_deribit_led_shock_deceleration_gross9_novelty as n
def test_dlsdr_novelty_is_frozen_pass_before_economics():
 assert hashlib.sha256(n.OUTPUT.read_bytes()).hexdigest()=='b66d95bceaefd22de512c4b6c19ed4f890a7e839d1a6fd26dc7085bf3bd290d3'
 d=json.loads(n.OUTPUT.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'}
 assert d['manifest_hash']==n.chash(core)
 assert d['gross9_novelty_status']=='passed' and d['every_gross9_sleeve_passed'] is True
 assert d['advance_to_economic_outcomes'] is True and d['evidence_boundary']['outcomes_opened'] is False
 assert max(x['metrics']['one_to_one_6h_max_matched_share'] for x in d['gross9_sleeves'].values())<.07
