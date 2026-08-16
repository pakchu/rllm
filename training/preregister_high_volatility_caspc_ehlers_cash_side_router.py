"""Outcome-blind preregistration for HVCELVCSR-8."""
from __future__ import annotations
import argparse,copy,json
from pathlib import Path
from training import preregister_high_volatility_caspc_ehlers_premium_activity_router as template
POLICY_ID='HVCELVCSR-8';DEFAULT_OUTPUT=Path('results/high_volatility_caspc_ehlers_cash_side_router_preregistration_2026-08-16.json');BASE=template.BASE;canonical_hash=template.canonical_hash;sha=template.sha
def build():
 v=copy.deepcopy(template.build());v.pop('manifest_hash');v['protocol_version']='high_volatility_caspc_ehlers_cash_side_router_v1';v['policy_id']=POLICY_ID;v['candidate_family']=[POLICY_ID]
 v['construction']={'base':'immutable HVCELV-8 event times, entry and hold','cash_source':'480 exact coherent BTCUSDT bars_binance_spot interval=1m rows [D-8h,D)','cash_return':'log(last spot close/first spot open), strict nonzero','router':'emit sign(cash_return); equivalently keep immutable HVCELV side on cash agreement and flip it on disagreement','missing':'ineligible','additional_or_tuned_thresholds':'none','entry':'immutable D+5m','hold':'8 elapsed hours'}
 v['mechanism']={'claim':'Cross-alt persistence that survives an Ehlers veto is most causally grounded when trade direction follows the completed cash BTC auction; cash disagreement identifies derivative-led crowding and overrides the frozen side.','why_low_gross9_overlap_is_plausible':'HVCELV sparse incidence is retained while independent cash-market direction changes signed exposure relative to Gross9'}
 v['source_plan']={'table':'bars_binance_spot','symbol':'BTCUSDT','interval':'1m','columns':['ts','open','high','low','close'],'window':['2023-07-01T00:00:00Z','2026-08-01T00:00:00Z'],'read_after_preregistration':True,'postentry_prices_returns_pnl':'sealed'}
 v['research_boundary']={'HVCELV_train_pass_test_positive_but_terminal_known':True,'prior_spot_led_candidate_outcomes_known':True,'exact_HVCELV_cash_router_incidence_or_outcomes_known':False,'source_incidence_opened':False,'postentry_return_or_pnl_opened':False,'classification':'exploratory discovery; not fresh confirmatory evidence','repair_of_prior_candidate':False,'selection_basis':'cash-market completed return as an independent causal side authority on immutable HVCELV events'}
 v['stopping_rule']='source support, Gross9, train/test/eval/final; stop first failure; no cash formula, direction, base, side, clock, hold, subset, substitution, or control repair.'
 return {**v,'manifest_hash':canonical_hash(v)}
def validate(v):
 if v['manifest_hash']!=canonical_hash({k:x for k,x in v.items() if k!='manifest_hash'}):raise RuntimeError('manifest drift')
 for a in BASE.values():
  if sha(a['path'])!=a['sha256']:raise RuntimeError('base drift')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();v=build();validate(v);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(v,indent=2)+"\n");print(a.output)
