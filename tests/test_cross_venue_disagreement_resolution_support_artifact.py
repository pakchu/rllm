import hashlib,json
from pathlib import Path
from training import build_cross_venue_disagreement_resolution_support as s
P=Path('results/cross_venue_disagreement_resolution_relay_support_2026-08-08.json')
def digest(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def test_cvdr_support_is_frozen_pass_before_novelty_and_economics():
 assert digest(P)=='3272082665dff86554133a6cfa4e27184322f0978ac5dd4e70b96ea3ffa8a006'
 d=json.loads(P.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'}
 assert d['manifest_hash']==s.chash(core)=='31c99b8d29622d866abdc0a31d5aa0a7363d8b0c690dc74d8adccefa77c17f6e'
 assert d['clock']['sha256']==digest(s.CLOCK)
 assert d['support_passed'] is True and all(d['support_checks'].values())
 assert d['advance_to_gross9_novelty'] is True and d['advance_to_economic_outcomes'] is False
