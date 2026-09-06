import hashlib,json
from training import evaluate_oil_volatility_rotation_btc_confirmation_relay_gross9_novelty as n
def test_ovrcr_novelty_is_frozen_pass_before_economics():
 assert hashlib.sha256(n.OUTPUT.read_bytes()).hexdigest()=='52f32511295adcbf2963218999a10130ff96127dab4d613846b4ef9b27c006d7';d=json.loads(n.OUTPUT.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'};assert d['manifest_hash']==n.canonical_hash(core)=='9fff86ac664a2c00185366bcb343c90bdb5fd53fb731158d43a10903d3fbd820';assert d['every_gross9_sleeve_passed'] is True and d['advance_to_economic_outcomes'] is True and d['evidence_boundary']['outcomes_opened'] is False
