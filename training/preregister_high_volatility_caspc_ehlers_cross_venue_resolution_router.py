"""Outcome-blind preregistration for HVCELVCVDR-8."""
from __future__ import annotations
import argparse,copy,json
from pathlib import Path
from training import preregister_high_volatility_caspc_ehlers_cash_side_router as template
POLICY_ID='HVCELVCVDR-8';DEFAULT_OUTPUT=Path('results/high_volatility_caspc_ehlers_cross_venue_resolution_router_preregistration_2026-08-16.json');canonical_hash=template.canonical_hash;sha=template.sha
CVDR={'preregistration':{'path':'results/cross_venue_disagreement_resolution_relay_preregistration_2026-08-08.json','sha256':'bc5a567c2387f175c3c521a7c00b0df10f28229ea473c8e261fcb0ae5e9024cf'},'support':{'path':'results/cross_venue_disagreement_resolution_relay_support_2026-08-08.json','sha256':'3272082665dff86554133a6cfa4e27184322f0978ac5dd4e70b96ea3ffa8a006'},'clock':{'path':'data/cross_venue_disagreement_resolution_relay_clocks_2023_2026.csv.gz','sha256':'3bb871777420669159c4b4d28338968ef0641d3b14c311a83bb04f6dc12e3525'}}
def build():
 v=copy.deepcopy(template.build());v.pop('manifest_hash');v['protocol_version']='high_volatility_caspc_ehlers_cross_venue_resolution_router_v1';v['policy_id']=POLICY_ID;v['candidate_family']=[POLICY_ID];v['routing_component_artifacts']=CVDR
 v['construction']={'base':'immutable HVCELV-8 event times, entry and hold','routing_component':'frozen source-passed CVDR-6 clock','active_join':'unique CVDR entry<=HVCELV entry<CVDR exit; timestamp tolerance none','router':'active CVDR side overrides HVCELV side; no active CVDR leaves immutable HVCELV side','multiple_active':'hard failure','activation_gate':'at least one override in train and at least three over all splits','additional_or_tuned_thresholds':'none','entry':'immutable HVCELV D+5m','hold':'8 elapsed hours'}
 v['mechanism']={'claim':'Cross-alt persistence that survives an Ehlers veto should defer to a causally active cross-venue disagreement-resolution state because venue-price conflict resolution identifies the contemporaneous cash discovery direction.','why_low_gross9_overlap_is_plausible':'sparse HVCELV incidence is retained while asynchronous cross-venue states alter signed exposure independently of Gross9'}
 v['source_plan']={'sources':['immutable HVCELV-8 clock','immutable source-passed CVDR-6 clock'],'read_after_preregistration':True,'postentry_prices_returns_pnl':'sealed'}
 v['research_boundary']={'HVCELV_train_pass_test_positive_but_terminal_known':True,'CVDR_source_pass_gross9_fail_known':True,'CVDR_economic_outcomes_unopened':True,'exact_combined_incidence_or_outcomes_known':False,'source_incidence_opened':False,'postentry_return_or_pnl_opened':False,'classification':'exploratory discovery; not fresh confirmatory evidence','repair_of_prior_candidate':False,'selection_basis':'asynchronous cross-venue resolution as an independent causal side authority'}
 v['stopping_rule']='source support including activation, Gross9, train/test/eval/final; stop first failure; no join, override, base, component, side, clock, hold, substitution, or control repair.'
 return {**v,'manifest_hash':canonical_hash(v)}
def validate(v):
 if v['manifest_hash']!=canonical_hash({k:x for k,x in v.items() if k!='manifest_hash'}):raise RuntimeError('manifest drift')
 for arts in (template.BASE,CVDR):
  for a in arts.values():
   if sha(a['path'])!=a['sha256']:raise RuntimeError('component drift')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();v=build();validate(v);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(v,indent=2)+"\n");print(a.output)
