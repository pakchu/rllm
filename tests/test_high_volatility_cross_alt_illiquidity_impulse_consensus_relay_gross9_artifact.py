import hashlib,json
from pathlib import Path
from training import export_gross9_structural_clocks as g
P=Path('results/high_volatility_cross_alt_illiquidity_impulse_consensus_relay_gross9_novelty_2026-08-13.json');EXPECTED='b3050b9ba8d3085722cebc3ec6abda39a38feb905a4326acf88a324d5386418b'
def test_gross9_pass_is_immutable_blind_and_complete():
 assert hashlib.sha256(P.read_bytes()).hexdigest()==EXPECTED;r=json.loads(P.read_text());h=r.pop('manifest_hash');assert g.canonical_hash(r)==h;assert r['policy_id']=='HVCIIC-8' and r['every_gross9_sleeve_passed'] and r['advance_to_economic_outcomes'];assert set(r['gross9_sleeves'])==set(g.EXPECTED_WEIGHTS);assert not r['evidence_boundary']['outcomes_opened'] and r['evidence_boundary']['economic_outcome_rows_opened']==0
