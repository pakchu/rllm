"""Source-only asynchronous CVDR router support for HVCELVCVDR-8."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from typing import Any
import pandas as pd
from training import preregister_high_volatility_caspc_ehlers_cross_venue_resolution_router as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
PREREG_SHA='a6b24af32d36aad76ebc0ae0b453790234e70a7862a257f9fb2c10886f58f3e1';BASE=Path('data/high_volatility_caspc_ehlers_active_veto_clocks_2023_2026.csv.gz');CVDR=Path('data/cross_venue_disagreement_resolution_relay_clocks_2023_2026.csv.gz');CLOCK=Path('data/high_volatility_caspc_ehlers_cross_venue_resolution_router_clocks_2023_2026.csv.gz');CONTROL_DIR=Path('data/high_volatility_caspc_ehlers_cross_venue_resolution_router_controls_2023_2026');RESULT=Path('results/high_volatility_caspc_ehlers_cross_venue_resolution_router_support_2026-08-16.json');MINIMUM_EVENTS={'train':8,'test':12,'eval':12,'final':8}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical_hash(v:Any):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def load(path):
 d=pd.read_csv(path)
 if 'control' in d:d=d[d.control.eq('primary')].copy()
 for c in ('decision_time','feature_available_time','entry_time','exit_time'):d[c]=pd.to_datetime(d[c],utc=True)
 return d.sort_values('entry_time').reset_index(drop=True)
def route(base,cvdr):
 rows=[];overrides={k:0 for k in ('train','test','eval','final')};active_total=0
 for b in base.itertuples(index=False):
  active=cvdr[(cvdr.entry_time<=b.entry_time)&(cvdr.exit_time>b.entry_time)]
  if len(active)>1:raise RuntimeError('multiple active CVDR states')
  side=int(b.side);cvdr_side=0;cvdr_entry=pd.NaT
  if len(active)==1:
   r=active.iloc[0];cvdr_side=int(r.side);cvdr_entry=r.entry_time;active_total+=1
   if cvdr_side!=side:overrides[b.split]+=1
   side=cvdr_side
  rows.append({'candidate':prereg.POLICY_ID,'control':'primary','split':b.split,'selected_action':b.selected_action,'decision_time':b.decision_time,'feature_available_time':b.feature_available_time,'entry_time':b.entry_time,'exit_time':b.exit_time,'side':side,'active_action_count':b.active_action_count,'base_side':int(b.side),'cvdr_active':len(active)==1,'cvdr_side':cvdr_side,'cvdr_entry_time':cvdr_entry})
 return pd.DataFrame(rows),{'active_total':active_total,'overrides_by_split':overrides,'overrides_total':sum(overrides.values())}
def stats(d,k):
 x=d[d.split.eq(k)];n=len(x);l=int(x.side.eq(1).sum());s=int(x.side.eq(-1).sum());return {'events':n,'longs':l,'shorts':s,'minority_side_share':min(l,s)/n if n else 0.,'max_month_share':float(x.entry_time.dt.strftime('%Y-%m').value_counts().max()/n) if n else 0.}
def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError('prereg drift')
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);base,cvdr=load(BASE),load(CVDR);clock,activation=route(base,cvdr);CLOCK.parent.mkdir(parents=True,exist_ok=True);_write_gzip_csv(clock,CLOCK);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(base,CONTROL_DIR/'immutable_base_side.csv.gz')
 support={k:stats(clock,k) for k in MINIMUM_EVENTS};checks={}
 for k,v in support.items():checks.update({f'{k}_minimum_events':v['events']>=MINIMUM_EVENTS[k],f'{k}_side_balance':v['minority_side_share']>=.2,f'{k}_month_concentration':v['max_month_share']<=.45})
 checks['activation_train_override_min_1']=activation['overrides_by_split']['train']>=1;checks['activation_full_override_min_3']=activation['overrides_total']>=3;passed=all(checks.values());core={'protocol_version':'hvcelvcvdr_8_source_support_v1','policy_id':prereg.POLICY_ID,'preregistration':{'path':str(prereg.DEFAULT_OUTPUT),'sha256':PREREG_SHA,'manifest_hash':reg['manifest_hash']},'completed_preentry_sources_opened':True,'postentry_return_pnl_execution_price_opened':False,'held_interval_funding_values_opened':False,'gross9_rows_opened':False,'component_clock_rows_opened':{'HVCELV-8':len(base),'CVDR-6':len(cvdr)},'activation':activation,'clock':{'path':str(CLOCK),'sha256':sha(CLOCK),'rows':len(clock)},'controls':{'immutable_base_side':{'path':str(CONTROL_DIR/'immutable_base_side.csv.gz'),'sha256':sha(CONTROL_DIR/'immutable_base_side.csv.gz'),'rows':len(base),'promotion_authorized':False}},'support':support,'support_checks':checks,'support_passed':passed,'advance_to_gross9_novelty':passed,'advance_to_economic_outcomes':False,'decision':'pass_to_novelty' if passed else 'terminal_source_support_reject'}
 r={**core,'manifest_hash':canonical_hash(core)};RESULT.write_text(json.dumps(r,indent=2,allow_nan=False,default=str)+"\n");return r
if __name__=='__main__':
 r=run();print(json.dumps({'passed':r['support_passed'],'activation':r['activation'],'support':r['support']},indent=2))
