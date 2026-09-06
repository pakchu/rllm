"""Source-only asynchronous HVCVARIR opposite-side veto for HVCASPCVAIV-8."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from typing import Any
import pandas as pd
from training import preregister_high_volatility_caspc_variance_acceleration_inventory_opposite_veto as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
PREREG_SHA='eb4b40c5c79ff6ba5173b7ce927b5fa8f29a40187dec8dff3715a718f47523bb';BASE=Path('data/high_volatility_cross_alt_serial_persistence_consensus_relay_clocks_2023_2026.csv.gz');VAI=Path('data/high_volatility_causal_variance_acceleration_inventory_relay_clocks_2020_2026.csv.gz');CLOCK=Path('data/high_volatility_caspc_variance_acceleration_inventory_opposite_veto_clocks_2023_2026.csv.gz');CONTROL_DIR=Path('data/high_volatility_caspc_variance_acceleration_inventory_opposite_veto_controls_2023_2026');RESULT=Path('results/high_volatility_caspc_variance_acceleration_inventory_opposite_veto_support_2026-08-17.json');MINIMUM_EVENTS={'train':8,'test':12,'eval':12,'final':8}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical_hash(v:Any):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def load(path):
 d=pd.read_csv(path)
 if 'control' in d:d=d[d.control.eq('primary')].copy()
 for c in ('decision_time','feature_available_time','entry_time','exit_time'):d[c]=pd.to_datetime(d[c],utc=True)
 return d.sort_values('entry_time').reset_index(drop=True)
def route(base,vai):
 rows=[];vetoes={k:0 for k in ('train','test','eval','final')};active_total=0;same_side_total=0
 for b in base.itertuples(index=False):
  active=vai[(vai.entry_time<=b.entry_time)&(vai.exit_time>b.entry_time)]
  if len(active)>1:raise RuntimeError('multiple active HVCVARIR states')
  vai_side=0;vai_entry=pd.NaT
  if len(active)==1:
   r=active.iloc[0];vai_side=int(r.side);vai_entry=r.entry_time;active_total+=1
   if vai_side==-int(b.side):
    vetoes[b.split]+=1
    continue
   same_side_total+=1
  rows.append({'candidate':prereg.POLICY_ID,'control':'primary','split':b.split,'decision_time':b.decision_time,'feature_available_time':b.feature_available_time,'entry_time':b.entry_time,'exit_time':b.exit_time,'side':int(b.side),'base_side':int(b.side),'vai_active':len(active)==1,'vai_side':vai_side,'vai_entry_time':vai_entry})
 return pd.DataFrame(rows),{'active_total':active_total,'same_side_total':same_side_total,'opposite_vetoes_by_split':vetoes,'opposite_vetoes_total':sum(vetoes.values())}
def stats(d,k):
 x=d[d.split.eq(k)];n=len(x);l=int(x.side.eq(1).sum());s=int(x.side.eq(-1).sum());return {'events':n,'longs':l,'shorts':s,'minority_side_share':min(l,s)/n if n else 0.,'max_month_share':float(x.entry_time.dt.strftime('%Y-%m').value_counts().max()/n) if n else 0.}
def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError('prereg drift')
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);base,vai=load(BASE),load(VAI);clock,activation=route(base,vai);CLOCK.parent.mkdir(parents=True,exist_ok=True);_write_gzip_csv(clock,CLOCK);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(base,CONTROL_DIR/'immutable_base_side.csv.gz')
 support={k:stats(clock,k) for k in MINIMUM_EVENTS};checks={}
 for k,v in support.items():checks.update({f'{k}_minimum_events':v['events']>=MINIMUM_EVENTS[k],f'{k}_side_balance':v['minority_side_share']>=.2,f'{k}_month_concentration':v['max_month_share']<=.45})
 checks['activation_train_opposite_veto_min_1']=activation['opposite_vetoes_by_split']['train']>=1;checks['activation_full_opposite_veto_min_3']=activation['opposite_vetoes_total']>=3;passed=all(checks.values());core={'protocol_version':'hvcaspcvaiv_8_source_support_v1','policy_id':prereg.POLICY_ID,'preregistration':{'path':str(prereg.DEFAULT_OUTPUT),'sha256':PREREG_SHA,'manifest_hash':reg['manifest_hash']},'completed_preentry_sources_opened':True,'postentry_return_pnl_execution_price_opened':False,'held_interval_funding_values_opened':False,'gross9_rows_opened':False,'component_clock_rows_opened':{'HVCASPC-8':len(base),'HVCVARIR-8':len(vai)},'activation':activation,'clock':{'path':str(CLOCK),'sha256':sha(CLOCK),'rows':len(clock)},'controls':{'immutable_base_side':{'path':str(CONTROL_DIR/'immutable_base_side.csv.gz'),'sha256':sha(CONTROL_DIR/'immutable_base_side.csv.gz'),'rows':len(base),'promotion_authorized':False}},'support':support,'support_checks':checks,'support_passed':passed,'advance_to_gross9_novelty':passed,'advance_to_economic_outcomes':False,'decision':'pass_to_novelty' if passed else 'terminal_source_support_reject'}
 r={**core,'manifest_hash':canonical_hash(core)};RESULT.write_text(json.dumps(r,indent=2,allow_nan=False,default=str)+"\n");return r
if __name__=='__main__':
 r=run();print(json.dumps({'passed':r['support_passed'],'activation':r['activation'],'support':r['support']},indent=2))
