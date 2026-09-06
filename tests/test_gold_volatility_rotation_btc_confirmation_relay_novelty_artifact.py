import hashlib,json
from training import evaluate_gold_volatility_rotation_btc_confirmation_relay_gross9_novelty as n
def test_gvrcr_novelty_is_frozen_pass_before_economics():
 assert hashlib.sha256(n.OUTPUT.read_bytes()).hexdigest()=='dc62cd0c646ae12a14cec82f3b7899750b46e157518c5893c077e473c13230cb';d=json.loads(n.OUTPUT.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'};assert d['manifest_hash']==n.canonical_hash(core)=='5017eada666c7524f7f29e35050f900c0dcf6185734dcd1e28aa11a2bf02f13d';assert d['every_gross9_sleeve_passed'] is True and d['advance_to_economic_outcomes'] is True and d['evidence_boundary']['outcomes_opened'] is False
