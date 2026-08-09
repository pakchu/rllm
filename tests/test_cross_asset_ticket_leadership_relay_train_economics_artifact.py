import json
from training import evaluate_cross_asset_ticket_leadership_relay_economics as economics
def test_terminal_train_artifact_is_bound_and_later_stages_closed():
 p=json.loads(economics.OUTPUTS['train'].read_text());assert p['policy_id']=='CATLR-12';assert p['stage']=='train';assert not p['passed'];assert not p['advance_to_next_stage'];assert not p['later_stage_outcomes_opened'];assert p['primary']['base']['trades']==42;assert p['primary']['base']['absolute_return_pct']>0;assert p['primary']['calendar_halves']['first']['absolute_return_pct']>0;assert p['primary']['calendar_halves']['second']['absolute_return_pct']>0;h=p.pop('manifest_hash');assert economics.canonical_hash(p)==h
