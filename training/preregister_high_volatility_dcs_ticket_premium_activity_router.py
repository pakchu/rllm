"""Outcome-blind preregistration for HVDCSATPAPAR-8."""
from __future__ import annotations
import argparse,copy,json
from pathlib import Path
from training import preregister_high_volatility_caspc_ehlers_premium_activity_router as template
POLICY_ID='HVDCSATPAPAR-8';DEFAULT_OUTPUT=Path('results/high_volatility_dcs_ticket_premium_activity_router_preregistration_2026-08-16.json')
BASE={'preregistration':{'path':'results/high_volatility_directional_change_scarcity_ticket_participation_acceleration_relay_preregistration_2026-08-16.json','sha256':'8003732efa6d369c474ec08b71178729a4ba557f43ec7325142bec01a203f153'},'support':{'path':'results/high_volatility_directional_change_scarcity_ticket_participation_acceleration_relay_support_2026-08-16.json','sha256':'5abaebfb9999921fd9834ade8a928c55326d0a0ca3cb09bb22d920c491bbefe6'},'gross9':{'path':'results/high_volatility_directional_change_scarcity_ticket_participation_acceleration_relay_gross9_novelty_2026-08-16.json','sha256':'28bed590847c8c3582424875b9a54a8d02bba54cb65f5bd5ecf6c3132c71dd44'},'clock':{'path':'data/high_volatility_directional_change_scarcity_ticket_participation_acceleration_relay_clocks_2023_2026.csv.gz','sha256':'089a077cfe89b63540a4b0b0af1a18bf9b5818ac269875e2e9e688c2c432cef0'}}
canonical_hash=template.canonical_hash;sha=template.sha
def build():
 v=copy.deepcopy(template.build());v.pop('manifest_hash');v['protocol_version']='high_volatility_dcs_ticket_premium_activity_router_v1';v['policy_id']=POLICY_ID;v['candidate_family']=[POLICY_ID];v['base_artifacts']=BASE
 v['construction'].update({'base':'immutable HVDCSATPA-8 event times, entry and hold','state_grid':'every exact 00:00/08:00/16:00 UTC using [D-8h,D) completed data','router':'relative activity rank<=0.50 keeps immutable HVDCSATPA side; rank>0.50 flips it'})
 v['mechanism']={'claim':'A high-variation directional-change-scarcity path with rising ticket size and participation should continue under quiet relative premium repricing, but reverse when active derivative premium repricing indicates crowded sponsorship.','why_low_gross9_overlap_is_plausible':'the sparse ticket-sponsored DCS incidence and premium-state polarity are absent from Gross9'}
 v['clock']['decision']='immutable base 00:00/08:00/16:00 UTC'
 v['research_boundary']={'HVDCSATPA_train_pass_test_fail_known':True,'prior_premium_activity_router_outcomes_known':True,'exact_HVDCSATPA_premium_router_incidence_or_outcomes_known':False,'source_incidence_opened':False,'postentry_return_or_pnl_opened':False,'classification':'exploratory discovery; not fresh confirmatory evidence','repair_of_prior_candidate':False,'selection_basis':'relative premium crowding polarity applied to a distinct immutable ticket-sponsored DCS base'}
 v['stopping_rule']='source support, Gross9, train/test/eval/final; stop first failure; no rank, median, direction, base, side, clock, hold, subset, substitution, or control repair.'
 return {**v,'manifest_hash':canonical_hash(v)}
def validate(v):
 if v['manifest_hash']!=canonical_hash({k:x for k,x in v.items() if k!='manifest_hash'}):raise RuntimeError('manifest drift')
 for a in BASE.values():
  if sha(a['path'])!=a['sha256']:raise RuntimeError('base drift')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();v=build();validate(v);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(v,indent=2)+"\n");print(a.output)
