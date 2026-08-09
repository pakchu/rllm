import json
from training import evaluate_cross_asset_ticket_leadership_relay_gross9_novelty as novelty
def test_novelty_artifact_passes_and_economics_authorized():
 p=json.loads(novelty.OUTPUT.read_text());assert p['policy_id']=='CATLR-12';assert p['source_support_passed'];assert p['every_gross9_sleeve_passed'];assert p['gross9_novelty_status']=='passed';assert p['advance_to_economic_outcomes'];assert p['evidence_boundary']['economic_outcome_rows_opened']==0;h=p.pop('manifest_hash');assert novelty.canonical_hash(p)==h
