import hashlib,json
from pathlib import Path
from training import evaluate_cross_venue_disagreement_resolution_gross9_novelty as n
P=n.OUTPUT
def test_cvdr_novelty_is_frozen_terminal_before_economics():
 assert hashlib.sha256(P.read_bytes()).hexdigest()=='5fa234f4be045241e65541f1b77c3f52f8912daddb182d9e5776172a61687fc4'
 d=json.loads(P.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'}
 assert d['manifest_hash']==n.chash(core)
 assert d['gross9_novelty_status']=='failed' and d['advance_to_economic_outcomes'] is False
 assert d['gross9_sleeves']['cand_rex_veto_7']['checks']['one_to_one_6h_max_matched_share'] is False
 assert d['gross9_sleeves']['rex_taker_low_range_position']['checks']['one_to_one_6h_max_matched_share'] is False
 assert all(v['metrics']['exact_entry_jaccard']<=.01 for v in d['gross9_sleeves'].values())
 assert d['evidence_boundary']['outcomes_opened'] is False
