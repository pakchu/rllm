"""Outcome-blind preregistration for HVCASPCPQA-8."""
from __future__ import annotations
import argparse,copy,json
from pathlib import Path
from training import preregister_high_volatility_caspc_ehlers_premium_activity_router as template
POLICY_ID='HVCASPCPQA-8';DEFAULT_OUTPUT=Path('results/high_volatility_caspc_quiet_premium_activity_preregistration_2026-08-16.json')
BASE={'preregistration':{'path':'results/high_volatility_cross_alt_serial_persistence_consensus_relay_preregistration_2026-08-13.json','sha256':'be77dfa8d83dfcb54171da4cf3263c57336fc629dd8f7b63ff8e8b33a860784a'},'support':{'path':'results/high_volatility_cross_alt_serial_persistence_consensus_relay_support_2026-08-13.json','sha256':'53f401d5ab9a499838c128b7023b4e420925455ca7e858e5a4db5c6e4f83d52e'},'gross9':{'path':'results/high_volatility_cross_alt_serial_persistence_consensus_relay_gross9_novelty_2026-08-13.json','sha256':'61c7fe10f3b331528315e8d7625aff7c8fb8ff1c82b6c553aa34f19ad89832f3'},'clock':{'path':'data/high_volatility_cross_alt_serial_persistence_consensus_relay_clocks_2023_2026.csv.gz','sha256':'bffadb70bea5d96fc779c641dbe3a1f50ba94fdf76a4e950d778a32a2b64b085'}}
canonical_hash=template.canonical_hash;sha=template.sha
def build():
 v=copy.deepcopy(template.build());v.pop('manifest_hash');v['protocol_version']='high_volatility_caspc_quiet_premium_activity_v1';v['policy_id']=POLICY_ID;v['candidate_family']=[POLICY_ID];v['base_artifacts']=BASE
 v['construction']={'base':'immutable HVCASPC-8 event times, side, entry and hold','state_grid':'every exact 03:00/11:00/19:00 UTC using [D-8h,D) completed data','premium_activity':'sum absolute one-minute bars_binance_premium close differences','btc_activity':'sqrt(sum squared BTCUSDT one-minute log-close returns)','relative_premium_activity':'premium_activity/btc_activity; both strict positive','rank':'strict-prior midrank over at most 270 valid state-grid blocks, minimum 180, current excluded','gate':'relative activity rank<=0.50; greater, missing or invalid is ineligible','additional_or_tuned_thresholds':'none; fixed median quiet-state gate','entry':'immutable D+5m','hold':'8 elapsed hours'}
 v['mechanism']={'claim':'Cross-alt serial persistence is more durable when derivative premium repricing remains quiet relative to the volatile BTC path, indicating spot-broad participation rather than crowded derivative leadership.','why_low_gross9_overlap_is_plausible':'the sparse quiet-premium subset of cross-alt serial persistence is absent from Gross9'}
 v['research_boundary']={'HVCASPC_train_pass_test_fail_known':True,'prior_HVCELV_premium_router_train_failure_known':True,'exact_HVCASPC_quiet_premium_incidence_or_outcomes_known':False,'source_incidence_opened':False,'postentry_return_or_pnl_opened':False,'classification':'exploratory discovery; not fresh confirmatory evidence','repair_of_prior_candidate':False,'selection_basis':'quiet relative premium activity applied to a distinct immutable CASPC base'}
 v['stopping_rule']='source support, Gross9, train/test/eval/final; stop first failure; no rank, median, side, base, clock, hold, subset, substitution, or control repair.'
 return {**v,'manifest_hash':canonical_hash(v)}
def validate(v):
 if v['manifest_hash']!=canonical_hash({k:x for k,x in v.items() if k!='manifest_hash'}):raise RuntimeError('manifest drift')
 for a in BASE.values():
  if sha(a['path'])!=a['sha256']:raise RuntimeError('base drift')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();v=build();validate(v);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(v,indent=2)+"\n");print(a.output)
