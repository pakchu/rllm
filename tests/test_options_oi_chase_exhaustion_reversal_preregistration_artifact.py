import hashlib,json
from pathlib import Path
from training import preregister_options_oi_chase_exhaustion_reversal as p
P=Path('results/options_oi_chase_exhaustion_reversal_preregistration_2026-08-08.json')
def test_artifact_is_frozen_before_incidence_or_postentry_returns():
 assert hashlib.sha256(P.read_bytes()).hexdigest()=='0dabee6a3528d6e3fbeb765bbc1628e376dd8b12e6bac81c2c5257f0d111162c'
 d=json.loads(P.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'};assert d['manifest_hash']==p.chash(core)
 assert d['research_boundary']['oicer_candidate_incidence_opened'] is False and d['research_boundary']['oicer_post_entry_return_or_pnl_opened'] is False
