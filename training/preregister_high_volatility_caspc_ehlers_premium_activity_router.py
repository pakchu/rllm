"""Outcome-blind preregistration for HVCELVPAR-8."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_cross_structure_action_vote as contract
POLICY_ID='HVCELVPAR-8';DEFAULT_OUTPUT=Path('results/high_volatility_caspc_ehlers_premium_activity_router_preregistration_2026-08-16.json')
BASE={'preregistration':{'path':'results/high_volatility_caspc_ehlers_active_veto_preregistration_2026-08-16.json','sha256':'7dc8f6228fa32eba7d5aded586d389855a47fbf12917800c43537e5e18c6244f'},'support':{'path':'results/high_volatility_caspc_ehlers_active_veto_support_2026-08-16.json','sha256':'9b7d2230588465a73dd9646387d3d7000b3f26d98daeef4e25cf51f4118c6375'},'gross9':{'path':'results/high_volatility_caspc_ehlers_active_veto_gross9_novelty_2026-08-16.json','sha256':'b7d94871688fce546a6a187594cf9665856193d7129cba4c2ca6e69fa9aa3b17'},'clock':{'path':'data/high_volatility_caspc_ehlers_active_veto_clocks_2023_2026.csv.gz','sha256':'1f8512a74906921169b115ec917c80c76b714c5a2f5fda25698feb6b4b026294'}}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical_hash(v:Any):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def build():
 c=contract.build();core={'protocol_version':'high_volatility_caspc_ehlers_premium_activity_router_v1','policy_id':POLICY_ID,'as_of_date':'2026-08-16','exploratory_discovery':True,'fresh_confirmatory_evidence':False,'source_incidence_opened':False,'outcomes_opened':False,'gross9_rows_opened':False,'singleton':True,'candidate_family':[POLICY_ID],'candidate_family_size':1,'base_artifacts':BASE,
 'construction':{'base':'immutable HVCELV-8 event times, entry and hold','state_grid':'every exact 03:00/11:00/19:00 UTC using [D-8h,D) completed data','premium_activity':'sum of absolute one-minute bars_binance_premium close differences','btc_activity':'sqrt(sum squared BTCUSDT one-minute log-close returns)','relative_premium_activity':'premium_activity/btc_activity; both strict positive','rank':'strict-prior midrank over at most 270 valid state-grid blocks, minimum 180, current excluded','router':'relative activity rank<=0.50 keeps immutable HVCELV side; rank>0.50 flips it','missing':'ineligible','additional_or_tuned_thresholds':'none; fixed median split','entry':'immutable D+5m','hold':'8 elapsed hours'},
 'mechanism':{'claim':'Cross-alt persistence that survives an Ehlers veto should continue when derivative premium repricing is quiet relative to BTC path variation, but reverse when premium repricing is unusually active and signals crowded derivative-led movement.','why_low_gross9_overlap_is_plausible':'sparse HVCELV events retain their irregular incidence while premium-state polarity changes signed exposure independently of Gross9'},
 'clock':{'decision':'immutable base 03:00/11:00/19:00 UTC','entry':'D+5m','hold':'8 elapsed hours','funding':'not a signal; exact held settlements sealed until economics'},'stages':c['stages'],'source_support_gates':c['source_support_gates'],'gross9_novelty_gates':c['gross9_novelty_gates'],'economic_gates':c['economic_gates'],
 'source_plan':{'tables':['bars_binance','bars_binance_premium'],'symbol':'BTCUSDT','interval':'1m','columns':['ts','open','high','low','close'],'window':['2023-01-01T00:00:00Z','2026-08-01T00:00:00Z'],'read_after_preregistration':True,'postentry_prices_returns_pnl':'sealed'},
 'research_boundary':{'HVCELV_train_pass_test_positive_but_terminal_known':True,'prior_premium_activity_candidate_outcomes_known':True,'exact_HVCELV_premium_router_incidence_or_outcomes_known':False,'source_incidence_opened':False,'postentry_return_or_pnl_opened':False,'classification':'exploratory discovery; not fresh confirmatory evidence','repair_of_prior_candidate':False,'selection_basis':'relative premium repricing activity as an independent causal crowding polarity state'},
 'stopping_rule':'source support, Gross9, train/test/eval/final; stop first failure; no rank, median, direction, base, side, clock, hold, subset, substitution, or control repair.'}
 return {**core,'manifest_hash':canonical_hash(core)}
def validate(v):
 if v['manifest_hash']!=canonical_hash({k:x for k,x in v.items() if k!='manifest_hash'}):raise RuntimeError('manifest drift')
 for a in BASE.values():
  if sha(a['path'])!=a['sha256']:raise RuntimeError('base drift')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();v=build();validate(v);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(v,indent=2)+"\n");print(a.output)
