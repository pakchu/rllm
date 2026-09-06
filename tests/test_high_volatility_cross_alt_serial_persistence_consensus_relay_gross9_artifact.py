import hashlib,json
from pathlib import Path
from training import export_gross9_structural_clocks as g
P=Path('results/high_volatility_cross_alt_serial_persistence_consensus_relay_gross9_novelty_2026-08-13.json');EXPECTED='61c7fe10f3b331528315e8d7625aff7c8fb8ff1c82b6c553aa34f19ad89832f3'
def test_gross9_pass_is_immutable_blind_and_complete():
 assert hashlib.sha256(P.read_bytes()).hexdigest()==EXPECTED;r=json.loads(P.read_text());h=r.pop('manifest_hash');assert g.canonical_hash(r)==h;assert r['policy_id']=='HVCASPC-8' and r['every_gross9_sleeve_passed'] and r['advance_to_economic_outcomes'];assert set(r['gross9_sleeves'])==set(g.EXPECTED_WEIGHTS);assert not r['evidence_boundary']['outcomes_opened'] and r['evidence_boundary']['economic_outcome_rows_opened']==0
