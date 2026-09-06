import hashlib,json
from pathlib import Path
from training import export_gross9_structural_clocks as g
P=Path('results/high_volatility_cross_alt_path_efficiency_consensus_relay_gross9_novelty_2026-08-13.json');EXPECTED='a6b7520af009af370a31eacb1f10e96b38306421f658f47a53dfd5c21f8061de'
def test_gross9_pass_is_immutable_blind_and_complete():
 assert hashlib.sha256(P.read_bytes()).hexdigest()==EXPECTED;r=json.loads(P.read_text());h=r.pop('manifest_hash');assert g.canonical_hash(r)==h;assert r['policy_id']=='HVCAPEC-8' and r['every_gross9_sleeve_passed'] and r['advance_to_economic_outcomes'];assert set(r['gross9_sleeves'])==set(g.EXPECTED_WEIGHTS);assert not r['evidence_boundary']['outcomes_opened'] and r['evidence_boundary']['economic_outcome_rows_opened']==0
